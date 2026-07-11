"""Stateless, human-like publishing scheduler.

Designed for ephemeral runners (GitHub Actions / free serverless): there is NO
saved plan file. Everything the scheduler needs is derived from two things that
are always available:

  1. Today's DATE — used to seed the daily plan deterministically, so every run
     of the day computes the *same* number of posts and the *same* target times.
  2. How many posts are ALREADY published today — counted from the site repo's
     posts.ts (the real source of truth). This makes double-posting impossible
     without persisting any local state.

Decision each run: if (posts already published today) < (planned slots whose time
has passed), a post is due now.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import CONFIG

log = logging.getLogger("agent.scheduler")


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(CONFIG.schedule_tz)
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


def _rng_for(day: datetime) -> random.Random:
    """A Random seeded by the calendar date, so all runs of the day agree."""
    return random.Random(day.date().toordinal())


def _roll_post_count(rng: random.Random) -> int:
    """0, 1, or 2 posts, weighted so the mean ~= AVG_POSTS_PER_DAY."""
    avg = min(CONFIG.avg_posts_per_day, CONFIG.max_posts_per_day)
    if CONFIG.max_posts_per_day >= 2:
        p_two = max(0.0, min(0.9, avg - 1.0)) if avg > 1 else 0.0
        p_zero = 0.15 if avg <= 1.4 else 0.08
        r = rng.random()
        if r < p_zero:
            return 0
        if r < p_zero + p_two:
            return 2
        return 1
    return 0 if rng.random() < 0.15 else 1


def plan_times(now_local: datetime | None = None) -> list[datetime]:
    """The deterministic target times for today (sorted, spaced >= MIN_GAP_HOURS)."""
    tz = _tz()
    now_local = now_local or datetime.now(tz)
    day = now_local
    rng = _rng_for(day)
    n = _roll_post_count(rng)
    if n == 0:
        return []

    start_h, end_h = CONFIG.publish_window_start, CONFIG.publish_window_end
    gap = timedelta(hours=CONFIG.min_gap_hours)
    picks: list[datetime] = []
    for _ in range(60):
        cand = sorted(
            day.replace(
                hour=rng.randint(start_h, max(start_h, end_h - 1)),
                minute=rng.randint(0, 59), second=0, microsecond=0,
            )
            for _ in range(n)
        )
        if all(cand[i + 1] - cand[i] >= gap for i in range(len(cand) - 1)):
            picks = cand
            break
        picks = cand
    return picks


def slots_due(now_local: datetime | None = None) -> int:
    """How many of today's planned posts should already be live by now."""
    tz = _tz()
    now_local = now_local or datetime.now(tz)
    return sum(1 for t in plan_times(now_local) if now_local >= t)


def is_post_due(published_today: int, now_local: datetime | None = None) -> bool:
    """A post is due if fewer have been published today than slots that are due."""
    due = slots_due(now_local)
    log.info("scheduler: %d slot(s) due so far today, %d already published",
             due, published_today)
    return published_today < due and published_today < CONFIG.max_posts_per_day


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tz = _tz()
    from collections import Counter
    counts = Counter()
    base = datetime.now(tz)
    for d in range(60):
        counts[len(plan_times(base + timedelta(days=d)))] += 1
    total = sum(k * v for k, v in counts.items())
    print("60-day plan simulation:", dict(sorted(counts.items())),
          f"-> avg {total/60:.2f} posts/day (target {CONFIG.avg_posts_per_day})")
    print("today's plan times:", [t.strftime('%H:%M') for t in plan_times()])
