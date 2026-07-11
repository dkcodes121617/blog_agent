"""Human-like publishing scheduler for a FREE Render Cron Job.

Render's free tier has no always-on worker, so we don't keep a process alive.
Instead Render invokes the script on a fixed cadence (hourly) and we decide, each
run, whether a post is due *right now*. The randomness lives in a per-day plan:

  - Once per day we roll how many posts today (0/1/2, weighted so the long-run
    average is ~AVG_POSTS_PER_DAY) and a random target time for each, inside the
    publish window, at least MIN_GAP_HOURS apart.
  - Each hourly invocation publishes any planned slot whose time has arrived and
    isn't done yet.

The plan is persisted (knowledge_base/daily_plan.json) so a restart or an extra
cron fire can't double-post.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from config import CONFIG

log = logging.getLogger("agent.scheduler")

_PLAN_PATH = CONFIG.kb_dir / "daily_plan.json"


@dataclass
class Slot:
    time_iso: str          # local-tz ISO datetime the post should go live
    done: bool = False


@dataclass
class DailyPlan:
    day: str               # YYYY-MM-DD (local)
    slots: list[Slot] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"day": self.day, "slots": [asdict(s) for s in self.slots]}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "DailyPlan":
        d = json.loads(text)
        return cls(day=d["day"], slots=[Slot(**s) for s in d["slots"]])


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(CONFIG.schedule_tz)
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


def _roll_post_count() -> int:
    """0, 1, or 2 posts, weighted so the mean ≈ AVG_POSTS_PER_DAY (capped at max)."""
    avg = min(CONFIG.avg_posts_per_day, CONFIG.max_posts_per_day)
    # Simple weighting: P(2) grows with avg; keep an occasional zero-day for realism.
    if CONFIG.max_posts_per_day >= 2:
        p_two = max(0.0, min(0.9, avg - 1.0)) if avg > 1 else 0.0
        p_zero = 0.15 if avg <= 1.4 else 0.08
        r = random.random()
        if r < p_zero:
            return 0
        if r < p_zero + p_two:
            return 2
        return 1
    # max 1/day
    return 0 if random.random() < 0.15 else 1


def _random_times(day: datetime, n: int) -> list[datetime]:
    """n random datetimes within the publish window, MIN_GAP_HOURS apart."""
    start_h, end_h = CONFIG.publish_window_start, CONFIG.publish_window_end
    gap = timedelta(hours=CONFIG.min_gap_hours)
    for _ in range(40):  # retry until spacing satisfied
        picks = sorted(
            day.replace(
                hour=random.randint(start_h, max(start_h, end_h - 1)),
                minute=random.randint(0, 59), second=0, microsecond=0,
            )
            for _ in range(n)
        )
        if all(picks[i + 1] - picks[i] >= gap for i in range(len(picks) - 1)):
            return picks
    return picks  # give up on spacing after many tries (rare)


def _make_plan(now_local: datetime) -> DailyPlan:
    n = _roll_post_count()
    times = _random_times(now_local, n)
    plan = DailyPlan(day=now_local.date().isoformat(),
                     slots=[Slot(time_iso=t.isoformat()) for t in times])
    log.info("new daily plan for %s: %d post(s) at %s",
             plan.day, n, [s.time_iso[11:16] for s in plan.slots])
    return plan


def load_or_make_plan(now_local: datetime | None = None) -> DailyPlan:
    tz = _tz()
    now_local = now_local or datetime.now(tz)
    today = now_local.date().isoformat()
    if _PLAN_PATH.exists():
        plan = DailyPlan.from_json(_PLAN_PATH.read_text(encoding="utf-8"))
        if plan.day == today:
            return plan
    plan = _make_plan(now_local)
    _PLAN_PATH.write_text(plan.to_json(), encoding="utf-8")
    return plan


def due_slot(now_local: datetime | None = None) -> Slot | None:
    """Return the earliest not-done slot whose time has passed, else None."""
    tz = _tz()
    now_local = now_local or datetime.now(tz)
    plan = load_or_make_plan(now_local)
    for slot in plan.slots:
        if slot.done:
            continue
        slot_time = datetime.fromisoformat(slot.time_iso)
        if slot_time.tzinfo is None:
            slot_time = slot_time.replace(tzinfo=tz)
        if now_local >= slot_time:
            return slot
    return None


def mark_slot_done(slot: Slot):
    if not _PLAN_PATH.exists():
        return
    plan = DailyPlan.from_json(_PLAN_PATH.read_text(encoding="utf-8"))
    for s in plan.slots:
        if s.time_iso == slot.time_iso:
            s.done = True
    _PLAN_PATH.write_text(plan.to_json(), encoding="utf-8")
    log.info("marked slot %s done", slot.time_iso[11:16])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Simulate a month to sanity-check the cadence distribution.
    tz = _tz()
    from collections import Counter
    counts = Counter()
    base = datetime.now(tz)
    for d in range(60):
        day = base + timedelta(days=d)
        random.seed(day.toordinal())
        counts[_roll_post_count()] += 1
    total = sum(k * v for k, v in counts.items())
    print("60-day simulation:", dict(sorted(counts.items())),
          f"-> avg {total/60:.2f} posts/day (target {CONFIG.avg_posts_per_day})")
