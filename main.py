#!/usr/bin/env python3
"""Forecast the top 5 new Solana coins by market cap at the house's answer reads.

Model (v2; everything about the window is read from task.json at runtime, every
data-driven quantity is estimated from the history at run time):

  * The leaderboard is a race in log10(market cap in SOL).  Every coin that
    counts is a "dud" (pinned near its launch cap ~28 SOL, log ~1.45), a
    "curve" coin still climbing its bonding curve, or a "pool" coin that has
    graduated.  Graduated pool coins hold essentially every top places, so a
    coin's venue and its current log size are the two facts that matter.
  * Each named coin's log size a horizon h out is a mixture of: keep growing,
    fade out of the race, or - if it is still on its curve - jump when it
    graduates.  A leader's growth is its own recent momentum (recency
    weighted, shrunk toward the population's leader growth), saturating with
    the horizon, not one fixed prior curve.
  * A curve coin graduates with a per-mark hazard that grows with its
    distance to the completion level, and lands either at the breakout level
    or at a dead level, as the history shows.
  * Coins born after the cut are an inhomogeneous Poisson process of
    breakouts: a rate per minute measured from the marks' graduated counters
    and the observed crossings of the contention level, times the
    distribution of the log size a breakout coin reaches at a given age.
    This mass is left unnamed on purpose, exactly as the rules reward.
  * The place distributions are the order statistics of that race, computed
    once per answer read and averaged over the reads (the arithmetic mean
    maximises the mean of the log scores).  The per-place size bins are the
    race's own bins blended with the measured per-place size distribution:
    its level growth and its gaps to the leader.
"""

import json
import math
import os
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# model constants (priors, fixed in advance; the data-driven quantities are
# estimated from the history at runtime)
# --------------------------------------------------------------------------
DUD_LOG = 1.45           # log10(SOL) a coin with no traction sits at
DUD_SD = 0.06
LEADER_LOG = 2.50        # log10(SOL) at which a coin has clearly broken out
BREAK_LOG = 3.00         # log10(SOL) a coin must reach to contend for the top 5
FADE_DROP = 0.25         # log a leader loses when it fades out of the race
FADE_SD = 0.16
FADE_DUD = 0.30          # share of fades that go all the way down to the dud
GRAD_LO = 2.00           # curve completion ramp: hazard 0 here ...
GRAD_HI = 2.32           # ... and saturated from here
GRAD_P_MAX = 0.92        # per-mark graduation hazard at/above GRAD_HI
GRAD_HIT = 2.78          # pool entry level of a graduation that matters
GRAD_HIT_SD = 0.16
GRAD_MISS = 1.65         # pool entry level of a graduation that dies
SIGMA_MAX = 0.40         # cap on the log-size spread at long horizons
MIN_ROW_P = 0.0004       # do not bother naming a coin below this
X_LO, X_HI, X_STEP = 1.00, 6.40, 0.02
AGE_STEP = 5.0           # minutes, newcomer age grid (fine near launch)
AGE_STEP_FAR = 15.0
AGE_FAR_FROM = 120.0

# --- momentum (per-coin recent growth) ---
MOM_TAU = 120.0          # minutes: momentum saturates over this scale
MOM_K = 3.0              # shrinkage of a coin's own slope toward the population
MOM_CAP = 0.16           # cap on the slope used for the drift (log/10min)
MOM_FLOOR = -0.03
MOM_DECAY = 0.55         # recency decay of a coin's sightings (per 10 min)

# --- newcomer rate ---
RATE_PRIOR = 0.34        # per-minute breakout rate believed on weak evidence
RATE_PRIOR_W = 3.0       # weight of that prior, in 10-minute bins
RATE_LO, RATE_HI = 0.08, 0.70
BREAK_PRIOR = (3.10, 0.18)   # (mean, sd) of a breakout's log size
BREAK_K = 3.0            # shrinkage of the measured breakout size

# --- fade (leaving the race) ---
FADE_PRIOR = 0.18        # per-mark fade probability believed on weak evidence
FADE_PRIOR_W = 1.0
FADE_BETA = 0.60         # the cumulative hazard grows sub-linearly in marks
FADE_LO, FADE_HI = 0.12, 0.34
STALE_W = 0.40           # discount for a coin missing from the last mark
LOW_POOL_W = 0.35        # discount for a pool coin already below the leaders

# --- size calibration ---
W_EMP = 0.50             # weight of the measured per-place size distribution
ANCHOR_W = 0.50          # weight of the measured level vs the race's own mean
LEVEL_PRIOR = 0.13       # per-10-minute growth of the place-1 cap
LEVEL_PRIOR_W = 1.0      # weight of that prior, in transitions
LEVEL_LO, LEVEL_HI = 0.04, 0.22
LEVEL_TAU = 90.0         # minutes: the level growth saturates over this scale
LEVEL_CAP = 0.38         # cap on the cumulative level growth
GAP_PRIOR = (0.0, 0.18, 0.19, 0.23, 0.27)   # place1 - place k, believed prior
EMP_SD_PRIOR = (0.10, 0.08, 0.08, 0.09, 0.10)
EMP_SD_FLOOR = 0.07      # tightest per-place size spread we will believe

SOL_LAMPORTS = 1_000_000_000.0


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def norm_cdf(z):
    return 0.5 * math.erfc(-z * 0.7071067811865476)


def norm_sf(z):
    return 0.5 * math.erfc(z * 0.7071067811865476)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_ndjson(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def wmean(pairs, default):
    """Weighted mean of (value, weight); falls back to default."""
    sw = sum(w for _, w in pairs)
    if sw <= 0.0:
        return default
    return sum(v * w for v, w in pairs) / sw


def wvar(pairs, mean):
    sw = sum(w for _, w in pairs)
    if sw <= 0.0:
        return 0.0
    return sum(w * (v - mean) ** 2 for v, w in pairs) / sw


def sat_scale(h, tau):
    """Mean-reverting cumulative factor: 1 at short h, -> tau/10 as h grows."""
    h = max(0.0, h)
    if h <= 0.0 or tau <= 0.0:
        return 0.0
    return (1.0 - math.exp(-h / tau)) / (h / tau)


# --------------------------------------------------------------------------
# history -> per-coin state
# --------------------------------------------------------------------------
class Coin(object):
    __slots__ = ("mint", "log", "age", "venue", "rank", "last_mark", "stale",
                 "slope")

    def __init__(self, mint, log, age, venue, rank, last_mark, stale, slope):
        self.mint = mint
        self.log = log
        self.age = age
        self.venue = venue
        self.rank = rank
        self.last_mark = last_mark
        self.stale = stale
        self.slope = slope


def coin_slope(hist, mu_pop):
    """Recency-weighted log-growth rate (log/10min) of one coin.

    hist is the coin's sightings as (t_minutes, log) pairs.  The fitted slope
    is shrunk toward the population's leader growth, so a coin seen twice is
    not trusted on its own two points.
    """
    if len(hist) < 2:
        return mu_pop
    pts = hist[-4:]
    n = len(pts)
    ws = [MOM_DECAY ** (n - 1 - i) for i in range(n)]
    sw = sum(ws)
    mx = sum(w * p[0] for w, p in zip(ws, pts)) / sw
    my = sum(w * p[1] for w, p in zip(ws, pts)) / sw
    sxx = sum(w * (p[0] - mx) ** 2 for w, p in zip(ws, pts))
    if sxx <= 1e-9:
        return mu_pop
    sxy = sum(w * (p[0] - mx) * (p[1] - my) for w, p in zip(ws, pts))
    g = sxy / sxx
    g = (g * sw + mu_pop * MOM_K) / (sw + MOM_K)
    return max(MOM_FLOOR, min(MOM_CAP, g))


def build_coins(marks, mu_pop):
    """Latest known state of every coin the history names, plus momentum."""
    t0 = parse_ts(marks[0]["at"])
    hist = {}
    latest = {}
    n = len(marks)
    for i, mark in enumerate(marks):
        at = parse_ts(mark["at"])
        tmin = (at - t0).total_seconds() / 60.0
        for row in mark.get("rows") or []:
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap <= 0:
                continue
            mint = row["mint"]
            log = math.log10(cap)
            hist.setdefault(mint, []).append((tmin, log))
            latest[mint] = (log, (at - parse_ts(row["created_at"])).total_seconds() / 60.0,
                            row.get("venue"), int(row.get("rank", 999)), i, i < n - 1)
    coins = {}
    for mint in sorted(latest):
        log, age, venue, rank, last_mark, stale = latest[mint]
        coins[mint] = Coin(mint, log, age, venue, rank, last_mark, stale,
                           coin_slope(hist[mint], mu_pop))
    return coins


# --------------------------------------------------------------------------
# data-driven parameters
# --------------------------------------------------------------------------
def transitions(marks):
    """(t_cur, span, prev_rows, cur_rows) for each mark pair."""
    out = []
    for i in range(1, len(marks)):
        t_prev = parse_ts(marks[i - 1]["at"])
        t_cur = parse_ts(marks[i]["at"])
        span = (t_cur - t_prev).total_seconds() / 60.0
        if span <= 0:
            continue
        out.append((t_cur, span, marks[i - 1].get("rows") or [],
                    marks[i].get("rows") or []))
    return out


def estimate_params(marks):
    """Leader growth and spread, fade rate, level growth, breakout rate/size."""
    trs = transitions(marks)
    t_last = parse_ts(marks[-1]["at"]) if marks else None

    # --- leader growth: the coins that held a top-5 place across a transition
    lead = []       # (delta per 10min, recency weight)
    level = []      # place-1 cap growth per 10 min
    persist = []    # (kept, total) of the previous mark's top 5, pool era only
    for t_cur, span, prev_rows, cur_rows in trs:
        age = 0.0 if t_last is None else max(0.0, (t_last - t_cur).total_seconds() / 600.0)
        w = 0.62 ** age

        def logs(rows):
            out = {}
            for row in rows:
                cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
                if cap > 0:
                    out[row["mint"]] = math.log10(cap)
            return out

        plog = logs(prev_rows)
        clog = logs(cur_rows)
        for row in prev_rows[:5]:
            mint = row["mint"]
            if mint in clog and any(r["mint"] == mint for r in cur_rows[:5]):
                lead.append(((clog[mint] - plog[mint]) * (10.0 / span), w))
        if prev_rows and cur_rows:
            level.append(((clog[cur_rows[0]["mint"]] - plog[prev_rows[0]["mint"]])
                          * (10.0 / span), w))

        def pool_era(rows):
            if len(rows) < 5:
                return False
            for row in rows[:5]:
                cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
                if cap <= 0:
                    return False
                if row.get("venue") != "pool" and math.log10(cap) < BREAK_LOG:
                    return False
            return True

        if pool_era(prev_rows) and pool_era(cur_rows):
            top5 = {r["mint"] for r in prev_rows[:5]}
            kept = sum(1 for r in cur_rows[:5] if r["mint"] in top5)
            persist.append((kept, 5, w))

    if lead:
        mu = wmean(lead, 0.07)
        sd = math.sqrt(max(0.0, wvar(lead, mu)))
        sw = sum(w for _, w in lead)
        mu = (mu * sw + 0.07 * 1.5) / (sw + 1.5)
        sd = math.sqrt(max(0.0, (sd * sd * sw + 0.09 ** 2 * 1.5) / (sw + 1.5)))
    else:
        mu, sd = 0.07, 0.09
    mu = max(0.0, min(0.15, mu))
    sd = max(0.05, min(0.30, sd))

    lvl_m = wmean(level, LEVEL_PRIOR)
    lvl_sw = sum(w for _, w in level)
    lvl = (lvl_m * lvl_sw + LEVEL_PRIOR * LEVEL_PRIOR_W) / (lvl_sw + LEVEL_PRIOR_W)
    lvl = max(LEVEL_LO, min(LEVEL_HI, lvl))

    if persist:
        sw = sum(w for _, _, w in persist)
        keep = sum(k * w for k, _, w in persist) / sw
        tot = sum(t * w for _, t, w in persist) / sw
        fade_m = 1.0 - keep / max(tot, 1e-9)
        fade_m = (fade_m * sw + FADE_PRIOR * FADE_PRIOR_W) / (sw + FADE_PRIOR_W)
    else:
        fade_m = FADE_PRIOR
    fade = max(FADE_LO, min(FADE_HI, fade_m))

    # --- breakout rate ---
    # first sighting at/above the contention level, per coin, recency weighted
    seen_first = {}
    for i, mark in enumerate(marks):
        for row in mark.get("rows") or []:
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap <= 0:
                continue
            mint = row["mint"]
            log = math.log10(cap)
            prev = seen_first.get(mint)
            if prev is None:
                seen_first[mint] = (i, log)
            elif prev[1] < BREAK_LOG <= log:
                seen_first[mint] = (i, log)
    cross_w = 0.0
    cross_sizes = []
    for mint in sorted(seen_first):
        i, log = seen_first[mint]
        if log < BREAK_LOG:
            continue
        at = parse_ts(marks[i]["at"])
        age = 0.0 if t_last is None else max(0.0, (t_last - at).total_seconds() / 600.0)
        w = 0.55 ** age
        cross_w += w
        cross_sizes.append((log, w))
    if t_last is not None and len(marks) >= 2:
        span = (t_last - parse_ts(marks[0]["at"])).total_seconds() / 60.0 + 10.0
        nbins = int(span / 10.0) + 1
        tw = 10.0 * sum(0.55 ** b for b in range(nbins))
    else:
        tw = 0.0
    rate_cross = cross_w / tw if tw > 0 else 0.0

    # graduations counted by the recorder itself, recency weighted
    grad_w = 0.0
    grad_total = 0.0
    prev_g = None
    for i, mark in enumerate(marks):
        g = (mark.get("counts") or {}).get("graduated")
        if g is None:
            continue
        if prev_g is not None:
            inc = g - prev_g
            if inc > 0:
                grad_total += inc
                span = (parse_ts(marks[i]["at"]) - parse_ts(marks[i - 1]["at"])).total_seconds() / 60.0
                if span > 0:
                    at = parse_ts(marks[i]["at"])
                    age = 0.0 if t_last is None else max(0.0, (t_last - at).total_seconds() / 600.0)
                    grad_w += (inc * (10.0 / span)) * (0.62 ** age)
        prev_g = g
    # weighted mean of the per-10-minute increments, turned into per-minute
    grad_rate = 0.0
    if grad_total > 0:
        sw = 0.0
        prev_g = None
        for i, mark in enumerate(marks):
            g = (mark.get("counts") or {}).get("graduated")
            if g is None:
                continue
            if prev_g is not None and g - prev_g > 0:
                at = parse_ts(marks[i]["at"])
                age = 0.0 if t_last is None else max(0.0, (t_last - at).total_seconds() / 600.0)
                sw += 0.62 ** age
            prev_g = g
        grad_rate = (grad_w / sw / 10.0) if sw > 0 else 0.0
    # the share of graduations that reach the contention level
    ratio = 1.0 if grad_total <= 0 else max(0.30, min(1.0, len(cross_sizes) / grad_total))
    rate_raw = rate_cross
    if grad_rate > 0.0:
        rate_raw = 0.5 * rate_cross + 0.5 * ratio * grad_rate

    n_eff = tw / 10.0
    rate = (rate_raw * n_eff + RATE_PRIOR * RATE_PRIOR_W) / (n_eff + RATE_PRIOR_W)
    rate = max(RATE_LO, min(RATE_HI, rate))

    if cross_sizes:
        bmean = wmean(cross_sizes, BREAK_PRIOR[0])
        bvar = wvar(cross_sizes, bmean)
        sw = cross_w
        bmean = (bmean * sw + BREAK_PRIOR[0] * BREAK_K) / (sw + BREAK_K)
        bsd = math.sqrt(max(0.0, (bvar * sw + BREAK_PRIOR[1] ** 2 * BREAK_K) / (sw + BREAK_K)))
    else:
        bmean, bsd = BREAK_PRIOR
    bsd = max(0.10, min(0.30, bsd))

    # --- per-place size structure of the pool era (for the size calibration)
    # only marks whose whole top 5 has graduated describe the sizes that
    # matter; earlier marks still have curve coins in the lower places
    era = []
    for mark in marks[-3:]:
        rows = mark.get("rows") or []
        if len(rows) < 5:
            continue
        caps = []
        ok = True
        for row in rows[:5]:
            c = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if c <= 0:
                ok = False
                break
            lg = math.log10(c)
            if row.get("venue") != "pool" and lg < BREAK_LOG:
                ok = False
                break
            caps.append(lg)
        if ok:
            era.append(caps)
    if era:
        gaps = []
        sds = []
        for k in range(5):
            gs = [caps[0] - caps[k] for caps in era]
            gaps.append(max(0.0, sum(gs) / len(gs)))
            if len(era) >= 2:
                m = sum(caps[k] for caps in era) / len(era)
                sds.append(math.sqrt(sum((caps[k] - m) ** 2 for caps in era) / len(era)))
            else:
                sds.append(EMP_SD_PRIOR[k])
    else:
        gaps = list(GAP_PRIOR)
        sds = list(EMP_SD_PRIOR)

    return {
        "mu": mu, "sd": sd, "rate": rate, "bmean": bmean, "bsd": bsd,
        "fade": fade, "level": lvl, "gaps": gaps, "emp_sd": sds,
    }


# --------------------------------------------------------------------------
# the race
# --------------------------------------------------------------------------
def mom_drift(coin, params, horizon):
    """Saturating cumulative growth from the coin's own momentum."""
    h = max(0.0, horizon)
    return coin.slope * (h / 10.0) * sat_scale(h, MOM_TAU)


def level_growth(params, horizon):
    """Saturating cumulative growth of the place-1 cap level."""
    h = max(0.0, horizon)
    return min(LEVEL_CAP, params["level"] * (h / 10.0) * sat_scale(h, LEVEL_TAU))


def coin_components(coin, params, horizon, cadence):
    """Mixture components (weight, mean, sd) of a named coin's future log size."""
    h = max(0.0, horizon)
    marks = h / cadence if cadence > 0 else h / 10.0
    drift = mom_drift(coin, params, h)
    sd = min(SIGMA_MAX, params["sd"] * math.sqrt(max(h, 0.0) / 10.0))
    # a coin that has already survived the early churn is the robust kind, so
    # the cumulative hazard grows sub-linearly in the number of marks
    fade = 1.0 - math.exp(-((max(marks, 0.0) ** FADE_BETA)) * params["fade"])
    stale_w = STALE_W if coin.stale else 1.0
    comps = []
    if coin.venue == "pool" or coin.log >= LEADER_LOG:
        low_w = LOW_POOL_W if coin.log < LEADER_LOG else 1.0
        stay = (1.0 - fade) * stale_w * low_w
        fade_w = 1.0 - stay
        comps.append((stay, coin.log + drift, sd))
        comps.append((fade_w * (1.0 - FADE_DUD),
                      max(DUD_LOG + 0.1, coin.log - FADE_DROP) + drift * 0.3,
                      max(FADE_SD, sd * 0.8)))
        comps.append((fade_w * FADE_DUD, DUD_LOG, DUD_SD))
    elif coin.log >= 1.9:
        # still on its curve: the graduation hazard grows with the distance
        # to the completion level, and a graduation lands either at the
        # breakout level or at a dead one
        ramp = max(0.0, min(1.0, (coin.log - GRAD_LO) / (GRAD_HI - GRAD_LO)))
        p1 = ramp * GRAD_P_MAX
        p_grad = 1.0 - (1.0 - p1) ** max(marks, 0.0)
        gw = p_grad * stale_w
        comps.append((gw * 0.5, GRAD_HIT + drift * 0.6, max(GRAD_HIT_SD, sd * 0.7)))
        comps.append((gw * 0.5, GRAD_MISS, 0.12))
        keep = (1.0 - p_grad) * (1.0 - fade) * stale_w
        comps.append((keep, coin.log + drift * 0.3, sd))
        comps.append((1.0 - gw - keep, DUD_LOG, DUD_SD))
    else:
        # dud: a small chance it is the next breakout, fading with its age
        surv = (1.0 - fade) * math.exp(-max(0.0, coin.age) / 60.0)
        jump_p = 0.02 * surv
        comps.append((jump_p, params["bmean"] + drift * 0.6, max(0.20, sd)))
        comps.append((1.0 - jump_p, DUD_LOG, DUD_SD))
    # keep the mixture proper
    tot = sum(w for w, _, _ in comps)
    if tot <= 0.0:
        return [(1.0, coin.log, max(0.05, sd))]
    if abs(tot - 1.0) > 1e-12:
        comps = [(w / tot, m, s) for w, m, s in comps]
    return comps


def mix_cdf(comps, x):
    tot = 0.0
    for w, mu, sd in comps:
        if w <= 0.0:
            continue
        if sd <= 0:
            tot += w * (1.0 if x >= mu else 0.0)
        else:
            tot += w * norm_cdf((x - mu) / sd)
    return tot


def mix_sf(comps, x):
    tot = 0.0
    for w, mu, sd in comps:
        if w <= 0.0:
            continue
        if sd <= 0:
            tot += w * (1.0 if x < mu else 0.0)
        else:
            tot += w * norm_sf((x - mu) / sd)
    return tot


def newcomer_intensity(params, horizon):
    """Expected number of post-cut breakouts above x, on the race grid."""
    xs = []
    x = X_LO
    while x < X_HI - 1e-9:
        xs.append(round(x, 6))
        x += X_STEP
    xs.append(X_HI)
    h = max(0.0, horizon)
    ages = []
    a = 1.0
    while a < min(h, AGE_FAR_FROM):
        ages.append(a)
        a += AGE_STEP
    a = max(a, AGE_FAR_FROM)
    while a < h:
        ages.append(a)
        a += AGE_STEP_FAR
    if h > 0 and (not ages or ages[-1] < h):
        ages.append(h)
    if not ages:
        ages = [1.0]
    out = []
    for x in xs:
        # trapezoidal integral over age (minutes) of the survival function
        integ = 0.0
        for i, a in enumerate(ages):
            aa = max(a - 2.0, 0.0)
            drift = params["mu"] * (aa / 10.0) * sat_scale(aa, MOM_TAU)
            sd = math.sqrt(params["bsd"] ** 2 +
                           (min(SIGMA_MAX, params["sd"] * math.sqrt(aa / 10.0))) ** 2)
            sf = norm_sf((x - (params["bmean"] + drift)) / sd)
            # it also has to still be in the race at the read, which bounds
            # the population of old breakouts instead of letting it grow for
            # ever.  The newcomer pool is not pre-selected, so it faces the
            # plain per-mark hazard.
            sf *= math.exp(-(a / 10.0) * params["fade"])
            lo = 0.0 if i == 0 else (ages[i - 1] + a) / 2.0
            hi = h if i == len(ages) - 1 else (a + ages[i + 1]) / 2.0
            integ += sf * max(0.0, hi - lo)
        out.append(params["rate"] * integ)
    return xs, out


def race(named, params, horizon, places, chart_size, cadence, edges):
    """Per-place named-coin probabilities, newcomer mass and size bins."""
    # keep the plausible candidates only: leaders plus recent movers
    named = sorted(named, key=lambda c: (-c.log, c.rank, c.mint))[:400]
    comps = {}
    for c in named:
        comps[c.mint] = coin_components(c, params, horizon, cadence)

    xs, lam = newcomer_intensity(params, horizon)
    n = len(xs)

    # size bin lookup: bin index for a grid point
    nbins = len(edges) + 1
    bin_of = []
    for x in xs:
        b = 0
        while b < len(edges) and x >= edges[b]:
            b += 1
        bin_of.append(b)

    # Poisson(mean) truncated at 5 (mass above goes to index 5)
    pois_cache = {}

    def poisson(mean):
        key = round(mean, 6)
        if key in pois_cache:
            return pois_cache[key]
        arr = [0.0] * 6
        if mean <= 0:
            arr[0] = 1.0
        else:
            p = math.exp(-mean)
            arr[0] = p
            for j in range(1, 6):
                p = p * mean / j
                arr[j] = p
            arr[5] = max(0.0, 1.0 - sum(arr[:5]))
        pois_cache[key] = arr
        return arr

    # per-place results
    place_coin = [dict() for _ in range(places)]
    place_new = [0.0] * places
    place_bins = [[0.0] * nbins for _ in range(places)]

    for xi in range(n):
        x = xs[xi]
        xnext = xs[xi + 1] if xi + 1 < n else X_HI + 1.0
        # distribution of the number of competitors above x
        dist = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for c in named:
            p = mix_sf(comps[c.mint], x)
            if p <= 1e-12:
                continue
            q = 1.0 - p
            new = [0.0] * 6
            for j in range(6):
                if dist[j] == 0.0:
                    continue
                new[j] += dist[j] * q
                if j + 1 < 6:
                    new[j + 1] += dist[j] * p
            dist = new
        lam_now = lam[xi]
        lam_next = lam[xi + 1] if xi + 1 < n else 0.0
        po = poisson(lam_now)
        # convolve
        conv = [0.0] * 6
        for a in range(6):
            if dist[a] == 0.0:
                continue
            for b in range(6 - a):
                conv[a + b] += dist[a] * po[b]
        for k in range(places):
            dp = conv[k] if k < 5 else 0.0
            if dp <= 0.0:
                continue
            # newcomer mass in this slice
            dlam = max(0.0, lam_now - lam_next)
            place_new[k] += dp * dlam
            b = bin_of[xi]
            place_bins[k][b] += dp * dlam
            for c in named:
                f0 = mix_cdf(comps[c.mint], x)
                f1 = mix_cdf(comps[c.mint], xnext)
                d = f1 - f0
                if d <= 1e-15:
                    continue
                place_coin[k][c.mint] = place_coin[k].get(c.mint, 0.0) + dp * d
                place_bins[k][b] += dp * d

    # normalise each place over the grid
    out_coin = []
    out_new = []
    out_bins = []
    for k in range(places):
        tot = sum(place_coin[k].values()) + place_new[k]
        if tot <= 0:
            out_coin.append({})
            out_new.append(1.0 / max(1, chart_size))
            out_bins.append([1.0 / nbins] * nbins)
            continue
        s = 1.0 / tot
        out_coin.append({m: v * s for m, v in place_coin[k].items()})
        out_new.append(place_new[k] * s)
        out_bins.append([v * s for v in place_bins[k]])
    return out_coin, out_new, out_bins


# --------------------------------------------------------------------------
# forecast assembly
# --------------------------------------------------------------------------
def emp_size_bins(params, anchor, places, edges):
    """Measured per-place size distribution: gaps around the anchor level."""
    nbins = len(edges) + 1
    out = []
    for k in range(places):
        mean = anchor - params["gaps"][k]
        sd = max(EMP_SD_FLOOR, params["emp_sd"][k])
        row = []
        for b in range(nbins):
            lo = edges[b - 1] if b >= 1 else -1e9
            hi = edges[b] if b < nbins - 1 else 1e9
            p = norm_cdf((hi - mean) / sd) - norm_cdf((lo - mean) / sd)
            row.append(max(0.0, p))
        out.append(row)
    return out


def build_forecast(task, marks):
    places = int(task.get("places", 5))
    chart_size = int(task.get("chart_size", 100))
    edges = [float(e) for e in task["size"]["edges"]]
    cadence = float(task.get("cadence_minutes", 10) or 10)
    reads = [parse_ts(r) for r in task.get("reads", [])]
    if not reads:
        reads = [parse_ts(task["window"]["answer_at"])]
    params = estimate_params(marks)
    coins = build_coins(marks, params["mu"])
    last_at = parse_ts(marks[-1]["at"]) if marks else parse_ts(task["t0"])
    nbins = len(edges) + 1

    # only the plausible candidates are worth naming
    cand = sorted(coins.values(), key=lambda c: (-c.log, c.rank, c.mint))
    cand = [c for c in cand if c.log >= 1.9 or c.venue == "pool"][:400]

    # anchor of the measured size distribution: the last mark's leader cap,
    # when the pool era has begun; otherwise the race's own place-1 mean
    last_rows = marks[-1].get("rows") or [] if marks else []
    anchor_emp = None
    if len(last_rows) >= 5:
        cap1 = float(last_rows[0]["market_cap_lamports"]) / SOL_LAMPORTS
        if cap1 > 0:
            lg1 = math.log10(cap1)
            if last_rows[0].get("venue") == "pool" or lg1 >= BREAK_LOG:
                anchor_emp = lg1

    # log-cap centre of every size bin (for reading a distribution's level)
    centers = []
    for b in range(nbins):
        if b == 0:
            centers.append(edges[0] - 0.05)
        elif b == nbins - 1:
            centers.append(edges[-1] + 0.05)
        else:
            centers.append((edges[b - 1] + edges[b]) / 2.0)

    acc_coin = [dict() for _ in range(places)]
    acc_new = [0.0] * places
    acc_bins = [[0.0] * nbins for _ in range(places)]
    acc_emp = [[0.0] * nbins for _ in range(places)]
    nreads = 0
    for r in reads:
        horizon = (r - last_at).total_seconds() / 60.0
        if horizon < 0:
            horizon = 0.0
        pc, pn, pb = race(cand, params, horizon, places, chart_size, cadence, edges)
        nreads += 1
        for k in range(places):
            for m, v in pc[k].items():
                acc_coin[k][m] = acc_coin[k].get(m, 0.0) + v
            acc_new[k] += pn[k]
            for b, v in enumerate(pb[k]):
                acc_bins[k][b] += v
        # anchor of the measured distribution: the last mark's leader cap
        # grown at the measured level, blended with the race's own place-1
        # mean (the race knows about newcomers the anchor does not)
        tot = sum(pb[0])
        model_p1 = (sum(c * pb[0][b] for b, c in enumerate(centers)) / tot) if tot > 0 else 0.0
        if anchor_emp is not None:
            anchor = (ANCHOR_W * (anchor_emp + level_growth(params, horizon)) +
                      (1.0 - ANCHOR_W) * model_p1)
        else:
            anchor = model_p1
        emp = emp_size_bins(params, anchor, places, edges)
        for k in range(places):
            for b, v in enumerate(emp[k]):
                acc_emp[k][b] += v
    inv = 1.0 / max(1, nreads)
    for k in range(places):
        for m in list(acc_coin[k].keys()):
            acc_coin[k][m] *= inv

    # ---- probabilities ----
    probabilities = []
    for k in range(places):
        rows = sorted(acc_coin[k].items(), key=lambda kv: (-kv[1], kv[0]))
        rows = [(m, v) for m, v in rows if v >= MIN_ROW_P][:100]
        for mint, p in rows:
            probabilities.append({"mint": mint, "place": k + 1, "p": round(p, 9)})

    # ---- sizes: the race's bins blended with the measured distribution ----
    sizes = []
    for k in range(places):
        bins = [(1.0 - W_EMP) * acc_bins[k][b] * inv +
                W_EMP * acc_emp[k][b] * inv for b in range(nbins)]
        tot = sum(bins)
        if tot > 1.0:
            bins = [b / tot for b in bins]
        for b, p in enumerate(bins):
            if p >= 1e-6:
                sizes.append({"place": k + 1, "bin": b, "p": round(p, 9)})

    # ---- top5: maximise expected agreement ----
    top5 = choose_top5(acc_coin, places, coins)
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def choose_top5(acc_coin, places, coins):
    pool = {}
    for k in range(places):
        for m, v in acc_coin[k].items():
            pool[m] = max(pool.get(m, 0.0), v)
    order = sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    mints = [m for m, _ in order]
    while len(mints) < places:
        extra = sorted(coins.keys())
        for m in extra:
            if m not in mints:
                mints.append(m)
                break
        if len(mints) >= places:
            break
    mints = mints[:max(places, len(mints))]

    def score(assign):
        # expected top-5 agreement of this ordered guess
        tot = 0.0
        for idx, mint in enumerate(assign):
            k = idx + 1
            for j in range(1, places + 1):
                pj = acc_coin[j - 1].get(mint, 0.0)
                if pj > 0.0:
                    tot += pj * (places + 1 - abs(j - k))
        return tot

    best = None
    best_s = -1e18
    n = len(mints)

    def rec(chosen, used):
        nonlocal best, best_s
        if len(chosen) == places:
            s = score(chosen)
            if s > best_s:
                best_s = s
                best = list(chosen)
            return
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            chosen.append(mints[i])
            rec(chosen, used)
            chosen.pop()
            used[i] = False

    rec([], [False] * n)
    return best if best else mints[:places]


# --------------------------------------------------------------------------
def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    task = read_json(os.path.join(in_dir, "task.json"))
    marks = read_ndjson(os.path.join(in_dir, "history.ndjson"))
    marks = [m for m in marks if m.get("rows")]
    forecast = build_forecast(task, marks)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, task.get("output", "forecast.json"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(forecast, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
