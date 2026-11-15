"""Fixed-seed synthetic customers and activity events (no download, no real data).

Story: every customer has a daily activity rate. Some customers churn in the
first or second 30-day window after the cutoff: most fade out over about 8
weeks first (gradual), some stop with no warning (sudden). Noise on purpose:
some customers fade but stay, some pause for a month and come back, some
signed up only weeks ago, a few are nearly dormant. That keeps metrics
realistic (AUC well below 1.0).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .layout import LABEL_DAYS, OBS_DAYS

PLANS = ("free", "basic", "pro")
FADE_DAYS = 56


def generate_data(
    n_customers: int,
    seed: int = 42,
    cutoff: date = date(2026, 6, 1),
    obs_days: int = OBS_DAYS,
    label_days: int = LABEL_DAYS,
):
    """Return (customer_rows, event_rows) as plain Python tuples.

    The timeline runs from `cutoff - obs_days` to `cutoff + 2 * label_days`, so the
    month after the cutoff is the training label window and the month after that
    is what `score` looks at.
    """
    rng = np.random.default_rng(seed)
    n = n_customers
    total_days = obs_days + 2 * label_days
    start = cutoff - timedelta(days=obs_days)
    day = np.arange(total_days)[None, :]  # day index; the cutoff is day `obs_days`
    cut = obs_days

    base_rate = rng.uniform(0.04, 0.5, n)
    dormant = rng.random(n) < 0.03
    base_rate[dormant] = rng.uniform(0.0005, 0.004, dormant.sum())
    spend = rng.uniform(5, 80, n)

    # Who churns in which window, who pauses for a month (and comes back).
    u = rng.random(n)
    churn_window = np.full(n, -1)
    churn_window[u < 0.07] = 0
    churn_window[(u >= 0.07) & (u < 0.14)] = 1
    pause_window = np.full(n, -1)
    pause_window[(u >= 0.14) & (u < 0.16)] = 0
    pause_window[(u >= 0.16) & (u < 0.18)] = 1
    sudden = rng.random(n) < 0.2
    fader_stays = (churn_window < 0) & (pause_window < 0) & (rng.random(n) < 0.12)
    new_signup = rng.random(n) < 0.06

    # Last active day of a churner: just around the start of its window.
    window_start = cut + label_days * np.clip(churn_window, 0, None)
    stop_gradual = window_start + rng.integers(-7, 4, n)
    stop_sudden = window_start + rng.integers(-3, label_days - 5, n)
    stop = np.where(sudden, stop_sudden, stop_gradual)

    mult = np.ones((n, total_days))
    churner = churn_window >= 0
    # gradual churners: rate falls linearly to 15% over FADE_DAYS, then stops
    ramp = 1 - 0.85 * (day - (stop[:, None] - FADE_DAYS)) / FADE_DAYS
    fade = np.clip(ramp, 0.15, 1.0)
    gradual_mask = churner & ~sudden
    mult = np.where(gradual_mask[:, None], fade, mult)
    mult = np.where(churner[:, None] & (day >= stop[:, None]), 0.0, mult)
    # fade-but-stay: looks like a churner before the cutoff, keeps 30% of activity after
    stay_ramp = np.clip(1 - 0.7 * (day - (cut - FADE_DAYS)) / FADE_DAYS, 0.3, 1.0)
    mult = np.where(fader_stays[:, None], stay_ramp, mult)
    # pause: zero activity for a whole 30-day window, then back to normal
    p_start = cut + label_days * np.clip(pause_window, 0, None)
    paused = (pause_window[:, None] >= 0) & (day >= p_start[:, None]) & (day < p_start[:, None] + label_days)
    mult = np.where(paused, 0.0, mult)
    # new customers: nothing before they signed up (1-40 days before the cutoff)
    signup_idx = np.where(new_signup, cut - rng.integers(1, 41, n), 0)
    mult = np.where(day < signup_idx[:, None], 0.0, mult)

    hits = rng.random((n, total_days)) < base_rate[:, None] * mult
    cust_idx, day_idx = np.nonzero(hits)
    amounts = np.round(spend[cust_idx] * rng.uniform(0.5, 1.5, cust_idx.size), 2)

    signup_dates = [
        start + timedelta(days=int(signup_idx[i])) if new_signup[i]
        else start - timedelta(days=int(rng.integers(30, 700)))
        for i in range(n)
    ]
    plans = rng.choice(PLANS, n, p=[0.5, 0.35, 0.15])
    customers = [(i, signup_dates[i], str(plans[i])) for i in range(n)]
    events = [
        (int(c), start + timedelta(days=int(d)), float(a))
        for c, d, a in zip(cust_idx, day_idx, amounts)
    ]
    return customers, events


CUSTOMER_SCHEMA = "customer_id int, signup_date date, plan string"
EVENT_SCHEMA = "customer_id int, event_date date, amount double"


def make_frames(spark, n_customers: int, seed: int = 42, cutoff: date = date(2026, 6, 1)):
    """Generate the data and return (customers_df, events_df) Spark DataFrames."""
    customers, events = generate_data(n_customers, seed, cutoff)
    return (
        spark.createDataFrame(customers, CUSTOMER_SCHEMA),
        spark.createDataFrame(events, EVENT_SCHEMA),
    )
