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


# ─── When to publish ───
# The audience is founders and business owners in the US, UK, Canada and Europe
# (details.md 2.5), but the schedule runs in Asia/Kolkata. Publishing at a uniformly
# random hour inside an 08:00-22:00 IST window put a lot of posts out at 03:00 UTC,
# when every target market is asleep.
#
# Publishing time matters far LESS for a blog than for social - search traffic builds
# over weeks and there is no feed decaying by the hour. What it does buy is the
# initial engagement signal, the first crawl, and a consistent rhythm. So these are
# WEIGHTS, not rules: a low-weight hour can still be chosen, just rarely.
#
# IST is UTC+5:30. The overlap where US-East morning, UK afternoon and CET afternoon
# are all awake is roughly 12:00-16:00 UTC, which is 17:30-21:30 IST.
_HOUR_WEIGHTS: dict[int, int] = {
    8: 1, 9: 1, 10: 1, 11: 1,     # 02:30-05:30 UTC - US asleep, EU asleep
    12: 2, 13: 2, 14: 3,          # 06:30-08:30 UTC - EU waking
    15: 4, 16: 5,                 # 09:30-10:30 UTC - UK/EU morning
    17: 8, 18: 8, 19: 8,          # 11:30-13:30 UTC - UK/EU afternoon + US-East morning
    20: 7, 21: 7,                 # 14:30-15:30 UTC - US-East mid-morning, US-West early
    22: 3,                        # 16:30 UTC - US-West morning, EU done for the day
}

# B2B readership is strongly mid-week. Again soft: weekends get fewer posts, not none,
# because an unbroken publishing rhythm is itself worth something.
_WEEKDAY_WEIGHT: dict[int, float] = {
    0: 0.85,   # Mon - inbox catch-up day
    1: 1.00,   # Tue
    2: 1.00,   # Wed
    3: 1.00,   # Thu
    4: 0.70,   # Fri - attention drops after midday
    5: 0.30,   # Sat
    6: 0.35,   # Sun - some evening reading
}


def _weighted_hour(rng: random.Random) -> int:
    """An hour inside the publish window, biased toward the global overlap."""
    lo, hi = CONFIG.publish_window_start, CONFIG.publish_window_end
    hours = [h for h in range(lo, max(lo + 1, hi))]
    # Hours outside the curve still get a small weight so a widened window keeps working.
    weights = [_HOUR_WEIGHTS.get(h, 2) for h in hours]
    return rng.choices(hours, weights=weights, k=1)[0]


# Normalised so the weights REDISTRIBUTE volume across the week without reducing it.
# Without this, multiplying by an average weight of ~0.74 quietly cut the cadence from
# 1.3 to 1.0 posts/day - a schedule change should move posts, not delete them.
_WEEKDAY_MEAN = sum(_WEEKDAY_WEIGHT.values()) / len(_WEEKDAY_WEIGHT)


# Normalised so the weekday weights REDISTRIBUTE volume across the week rather than
# reducing it.
_WEEKDAY_MEAN = sum(_WEEKDAY_WEIGHT.values()) / len(_WEEKDAY_WEIGHT)


def _rng_for_week(day: datetime) -> random.Random:
    """A Random seeded by the ISO week, so every run in a week agrees on its plan."""
    iso = day.isocalendar()
    return random.Random(iso[0] * 100 + iso[1])


def _posts_this_week(rng: random.Random) -> int:
    """How many posts this week — centred on the weekly target.

    Rolled PER WEEK rather than per day. Independent daily coin-flips hit the right
    long-run mean but clustered badly: at ~0.3/day a simulated two years produced 12%
    completely empty weeks and a 20-day silence. A weekly roll keeps the same average
    while guaranteeing the gap between posts stays sane, which is what actually matters
    for crawl rhythm and for looking like a publication rather than a cron job.
    """
    target = CONFIG.avg_posts_per_day * 7
    lo = max(1, int(target))          # never a silent week
    # Distribute the fractional part so the long-run mean matches the target exactly.
    return lo + (1 if rng.random() < (target - lo) else 0)


def _publish_days(day: datetime) -> dict[int, int]:
    """Map of {weekday -> posts} for the week containing `day`.

    Days are drawn without replacement, weighted toward mid-week, so the week's posts
    land on different days and cluster Tue-Thu.
    """
    rng = _rng_for_week(day)
    n = min(_posts_this_week(rng), 7 * CONFIG.max_posts_per_day)

    weekdays = list(range(7))
    weights = [_WEEKDAY_WEIGHT.get(d, 1.0) for d in weekdays]
    chosen: dict[int, int] = {}
    for _ in range(n):
        available = [(d, w) for d, w in zip(weekdays, weights)
                     if chosen.get(d, 0) < CONFIG.max_posts_per_day]
        if not available:
            break
        days_, ws = zip(*available)
        pick = rng.choices(days_, weights=ws, k=1)[0]
        chosen[pick] = chosen.get(pick, 0) + 1
    return chosen


def plan_times(now_local: datetime | None = None) -> list[datetime]:
    """The deterministic target times for today (sorted, spaced >= MIN_GAP_HOURS).

    The week decides WHICH days publish; the day decides WHAT TIME. Both are seeded
    from the calendar, so a stateless runner reaches the same answer on every run.
    """
    tz = _tz()
    now_local = now_local or datetime.now(tz)
    day = now_local
    n = _publish_days(day).get(day.weekday(), 0)
    if n == 0:
        return []

    rng = _rng_for(day)
    gap = timedelta(hours=CONFIG.min_gap_hours)
    picks: list[datetime] = []
    for _ in range(60):
        cand = sorted(
            day.replace(
                hour=_weighted_hour(rng),
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
