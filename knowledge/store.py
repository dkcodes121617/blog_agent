"""The uniqueness knowledge base.

A local, file-based vector store — zero API cost, no external DB, runs anywhere
(including a free Render cron box). Every published post is embedded with a small
local sentence-transformer and stored alongside its metadata. New topics and
finished drafts are cosine-compared against it so the agent never repeats itself.

Files (all under knowledge_base/, gitignored):
  - embeddings.npy       float32 matrix, one row per ledger entry (same order)
  - content_ledger.json  list of {slug, title, description, tags, text_hash, added}

The embedding model is loaded lazily so importing this module stays cheap (torch
is heavy); the graph only pays for it when it actually embeds.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import CONFIG

log = logging.getLogger("agent.kb")

_EMB_PATH = CONFIG.kb_dir / "embeddings.npy"
_LEDGER_PATH = CONFIG.kb_dir / "content_ledger.json"

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy singleton for the sentence-transformer."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer  # heavy import

                log.info("loading embedding model %s", CONFIG.embed_model)
                _model = SentenceTransformer(CONFIG.embed_model)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings (so dot product == cosine similarity)."""
    model = _get_model()
    vecs = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.astype(np.float32)


@dataclass
class LedgerEntry:
    slug: str
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    text_hash: str = ""
    added: str = ""


class KnowledgeBase:
    def __init__(self):
        self.ledger: list[LedgerEntry] = []
        self.matrix: np.ndarray | None = None
        self._load()

    # ── persistence ──
    def _load(self):
        if _LEDGER_PATH.exists():
            raw = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
            self.ledger = [LedgerEntry(**e) for e in raw]
        if _EMB_PATH.exists():
            self.matrix = np.load(_EMB_PATH)
        if self.matrix is not None and len(self.ledger) != len(self.matrix):
            log.warning(
                "KB out of sync (ledger=%d, matrix=%s) — will rebuild on next ingest",
                len(self.ledger), None if self.matrix is None else len(self.matrix),
            )

    def _save(self):
        _LEDGER_PATH.write_text(
            json.dumps([asdict(e) for e in self.ledger], indent=2), encoding="utf-8"
        )
        if self.matrix is not None:
            np.save(_EMB_PATH, self.matrix)

    # ── mutation ──
    def has_slug(self, slug: str) -> bool:
        return any(e.slug == slug for e in self.ledger)

    def add(self, *, slug: str, title: str, description: str, tags: list[str], body_text: str):
        """Embed and append a post. Idempotent on slug."""
        if self.has_slug(slug):
            log.info("KB already has slug=%s, skipping", slug)
            return
        combined = f"{title}\n{description}\n{body_text}"
        vec = embed([combined])  # (1, d)
        entry = LedgerEntry(
            slug=slug,
            title=title,
            description=description,
            tags=tags,
            text_hash=hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16],
            added=datetime.now(timezone.utc).isoformat(),
        )
        self.ledger.append(entry)
        self.matrix = vec if self.matrix is None else np.vstack([self.matrix, vec])
        self._save()
        log.info("KB added slug=%s (now %d entries)", slug, len(self.ledger))

    def prune_to(self, live_slugs: set[str]) -> list[str]:
        """Drop every entry whose slug is no longer a real post. Returns what went.

        The KB only ever grew. `add()` is idempotent and `ingest_existing` skipped
        slugs it already had, so a post that was renamed or deleted stayed in the
        ledger forever — and `known_slugs` comes straight from `all_slugs()`.

        That is not hypothetical. The ledger held
        `what-custom-software-actually-costs-in-2025` and
        `react-native-vs-flutter-2025` long after both were republished as -2026.
        The MDX validator checks every /blog/ link against `known_slugs`, so a
        draft linking to a phantom passed validation and would 404 on publish —
        which is verbatim the incident its own comment documents.

        The matrix rows are positional, so they are filtered with the ledger in one
        pass; rebuilding them separately is how the two fall out of alignment.
        """
        keep = [i for i, e in enumerate(self.ledger) if e.slug in live_slugs]
        dropped = [e.slug for e in self.ledger if e.slug not in live_slugs]
        if not dropped:
            return []

        self.ledger = [self.ledger[i] for i in keep]
        if self.matrix is not None and len(self.matrix):
            self.matrix = self.matrix[keep] if keep else None
        self._save()
        log.warning("KB pruned %d stale slug(s): %s", len(dropped), ", ".join(dropped))
        return dropped

    # ── queries ──
    def max_similarity(self, text: str) -> tuple[float, str | None]:
        """Highest cosine similarity of `text` vs anything in the KB, + its slug."""
        if self.matrix is None or len(self.ledger) == 0:
            return 0.0, None
        q = embed([text])[0]                 # (d,)
        sims = self.matrix @ q               # (n,) — cosine, since both normalised
        idx = int(np.argmax(sims))
        return float(sims[idx]), self.ledger[idx].slug

    def top_related(self, text: str, k: int = 3, exclude_slug: str | None = None) -> list[tuple[str, float]]:
        """Return up to k (slug, similarity) of the most related existing posts.

        Reused by the outliner to build a genuine topical-cluster internal-link
        plan (the enhancement we agreed on) — the same embedding search that
        powers uniqueness, for free.
        """
        if self.matrix is None or len(self.ledger) == 0:
            return []
        q = embed([text])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for i in order:
            slug = self.ledger[int(i)].slug
            if slug == exclude_slug:
                continue
            out.append((slug, float(sims[int(i)])))
            if len(out) >= k:
                break
        return out

    def all_slugs(self) -> list[str]:
        return [e.slug for e in self.ledger]

    def __len__(self):
        return len(self.ledger)
