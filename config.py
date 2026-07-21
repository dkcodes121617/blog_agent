"""Central configuration for the WizCodes blog agent.

All tunables come from environment variables (loaded from `.env` locally, or set
as Render service env vars in production). Kept in one place so the rest of the
codebase never reads os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the agent root (no-op if the file is absent, e.g. on Render).
AGENT_ROOT = Path(__file__).resolve().parent
load_dotenv(AGENT_ROOT / ".env")


def _clean(raw: str | None) -> str:
    """Strip whitespace and any trailing inline '# comment' (dotenv keeps those)."""
    if raw is None:
        return ""
    return raw.split("#", 1)[0].strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_clean(os.getenv(name)) or default)
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_clean(os.getenv(name)) or default)
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # ── Proxy / LLM ──
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "https://api3.claudestore.store")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4.6")
    small_model: str = os.getenv("ANTHROPIC_SMALL_FAST_MODEL", "claude-haiku-4.5")
    # Topic choice and outline set the shape of the whole post, so they get the
    # strongest model; the per-section writing stays on the default. Probed and
    # confirmed available on this proxy.
    strategy_model: str = os.getenv("ANTHROPIC_STRATEGY_MODEL", "claude-opus-4-8")

    # ── GitHub publishing ──
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_repo: str = os.getenv("GITHUB_REPO", "dkcodes121617/wizcodes_main_website")
    github_branch: str = os.getenv("GITHUB_BRANCH", "main")
    # IndexNow key for pinging Bing on publish. The matching <key>.txt must be served
    # from the site root (public/) or the ping is rejected. Optional: unset = no ping.
    indexnow_key: str = os.getenv("INDEXNOW_KEY", "2df4018dbb1444e6bc48faf84fc0ff39")
    git_author_name: str = os.getenv("GIT_AUTHOR_NAME", "WizCodes Blog Bot")
    git_author_email: str = os.getenv("GIT_AUTHOR_EMAIL", "business@wizcodes.site")

    # ── Cadence ──
    max_posts_per_day: int = _int("MAX_POSTS_PER_DAY", 1)
    avg_posts_per_day: float = _float("AVG_POSTS_PER_DAY", 0.3)
    publish_window_start: int = _int("PUBLISH_WINDOW_START", 8)
    publish_window_end: int = _int("PUBLISH_WINDOW_END", 22)
    min_gap_hours: int = _int("MIN_GAP_HOURS", 4)
    schedule_tz: str = os.getenv("SCHEDULE_TZ", "Asia/Kolkata")

    # ── Uniqueness ──
    topic_sim_threshold: float = _float("TOPIC_SIM_THRESHOLD", 0.82)
    body_sim_threshold: float = _float("BODY_SIM_THRESHOLD", 0.86)
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # ── Behaviour ──
    dry_run: bool = _bool("DRY_RUN", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Paths ──
    root: Path = AGENT_ROOT
    kb_dir: Path = AGENT_ROOT / "knowledge_base"
    output_dir: Path = AGENT_ROOT / "output"
    # Where the wizcodes site repo is checked out for publishing (kept local to agent).
    site_repo_dir: Path = AGENT_ROOT / "site_repo"
    # Where to read the site content from when running locally against the sibling folder.
    # In production the agent clones the repo into site_repo_dir; locally we can point at
    # the checked-out sibling so `ingest`/`facts` work without a network clone.
    local_site_dir: Path = field(default=AGENT_ROOT.parent / "wizcodes_next")

    # Derived content paths, relative to whichever site dir is in use.
    posts_registry_rel: str = "src/content/blog/posts.ts"
    blog_content_rel: str = "src/content/blog"
    site_config_rel: str = "src/config/site.ts"
    data_dir_rel: str = "src/data"

    def validate_for_publish(self) -> list[str]:
        """Return a list of missing-config problems that would block real publishing."""
        problems = []
        if not self.anthropic_api_key:
            problems.append("ANTHROPIC_API_KEY is not set")
        if not self.dry_run and not self.github_token:
            problems.append("GITHUB_TOKEN is not set (required when DRY_RUN=0)")
        return problems


CONFIG = Config()

# Ensure runtime dirs exist.
CONFIG.kb_dir.mkdir(parents=True, exist_ok=True)
CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
