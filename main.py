#!/usr/bin/env python3
"""Forecast the top 5 new Solana coins by market cap at the house's answer reads.

Model (derived from the practice history, nothing copied from any starter):

  * The leaderboard is a race in log10(market cap in SOL).  Every coin that
    counts is either a "dud" (pinned near its launch cap ~28 SOL, log ~1.45),
    a "curve" coin still climbing its bonding curve, or a "pool" coin that has
    graduated.  Graduated pool coins hold essentially every top place, so a
    coin's venue and its current log size are the two facts that matter.
  * Each named coin's log size a horizon h out is a 3-component mixture:
    keep growing (a saturating drift + diffusion), fall back to its dud level
    (the market's attention moves on), or - if it is still on its curve - jump
    to the breakout level when it graduates.
  * Coins born after the cut are modelled as an inhomogeneous Poisson process
    of breakouts: a rate per minute (measured from the history) times the
    distribution of the log size a breakout coin reaches at a given age.  This
    mass is left unnamed on purpose, exactly as the rules reward.
  * The place distributions and the per-place size-bin distributions are the
    order statistics of that race, computed once per answer read and averaged
    over the reads (the arithmetic mean maximises the mean of the log scores).

Everything about the window (deadline, reads, places, size edges, chart size)
is read from task.json at runtime.
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
BREAK_LOG = 2.80         # log10(SOL) a "breakout" coin is first seen at
GROW_TOTAL = 0.40        # extra log10 a broken-out coin gains in the long run
GROW_TAU = 240.0         # minutes: time constant of that growth
SIGMA_MAX = 0.45         # cap on the log-size spread at long horizons
DEATH_LAMBDA = 0.11      # leader disappearance rate per 10-minute mark
DEATH_BETA = 0.60        # the cumulative hazard grows sub-linearly in marks
CURVE_JUMP_P = 0.30      # chance a curve coin near the top graduates upward
MIN_ROW_P = 0.0004       # do not bother naming a coin below this
X_LO, X_HI, X_STEP = 1.00, 6.40, 0.02
AGE_STEP = 5.0           # minutes, newcomer age grid (fine near launch)
AGE_STEP_FAR = 15.0
AGE_FAR_FROM = 120.0

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


# --------------------------------------------------------------------------
# history -> per-coin state
# --------------------------------------------------------------------------
class Coin(object):
    __slots__ = ("mint", "log", "age", "venue", "rank", "last_mark", "stale")

    def __init__(self, mint, log, age, venue, rank, last_mark, stale):
        self.mint = mint
        self.log = log
        self.age = age
        self.venue = venue
        self.rank = rank
        self.last_mark = last_mark
        self.stale = stale


def build_coins(marks):
    """Latest known state of every coin the history names."""
    coins = {}
    n = len(marks)
    for i, mark in enumerate(marks):
        at = parse_ts(mark["at"])
        for row in mark.get("rows") or []:
            mint = row["mint"]
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap <= 0:
                continue
            age = (at - parse_ts(row["created_at"])).total_seconds() / 60.0
            cur = coins.get(mint)
            if cur is None:
                coins[mint] = Coin(mint, math.log10(cap), age, row.get("venue"),
                                   int(row.get("rank", 999)), i, False)
            else:
                if i >= cur.last_mark:
                    cur.log = math.log10(cap)
                    cur.age = age
                    cur.venue = row.get("venue")
                    cur.rank = int(row.get("rank", 999))
                    cur.last_mark = i
                    cur.stale = (i < n - 1)
    return coins


# --------------------------------------------------------------------------
# data-driven parameters
# --------------------------------------------------------------------------
def estimate_params(marks, coins):
    """Drift/spread of the leaders, the breakout rate and size, death rate."""
    deltas = []
    for i in range(1, len(marks)):
        t_prev = parse_ts(marks[i - 1]["at"])
        t_cur = parse_ts(marks[i]["at"])
        span = (t_cur - t_prev).total_seconds() / 60.0
        if span <= 0:
            continue
        prev = {}
        for row in marks[i - 1].get("rows") or []:
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap > 0:
                prev[row["mint"]] = math.log10(cap)
        for row in marks[i].get("rows") or []:
            mint = row["mint"]
            if mint not in prev:
                continue
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap <= 0:
                continue
            if prev[mint] < LEADER_LOG:
                continue
            deltas.append((math.log10(cap) - prev[mint]) * (10.0 / span))
    if len(deltas) >= 4:
        mu = sum(deltas) / len(deltas)
        var = sum((d - mu) ** 2 for d in deltas) / len(deltas)
        sd = math.sqrt(var)
    else:
        mu, sd = 0.085, 0.14
    mu = max(0.0, min(0.35, mu))
    sd = max(0.05, min(0.45, sd))

    # breakout rate: distinct coins first seen at/above BREAK_LOG, per minute
    seen = {}
    for i, mark in enumerate(marks):
        for row in mark.get("rows") or []:
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap <= 0:
                continue
            log = math.log10(cap)
            if log < BREAK_LOG:
                continue
            mint = row["mint"]
            if mint not in seen or i < seen[mint][0]:
                seen[mint] = (i, log)
    first_mark = None
    for mark in marks:
        if mark.get("rows"):
            first_mark = mark
            break
    if first_mark is not None and len(marks) >= 2:
        t0 = parse_ts(first_mark["at"])
        t1 = parse_ts(marks[-1]["at"])
        cadence = 10.0
        window = (t1 - t0).total_seconds() / 60.0 + cadence
    else:
        window = 0.0
    sizes = [v[1] for v in seen.values()]
    if window > 0 and sizes:
        rate = len(sizes) / window
        bmean = sum(sizes) / len(sizes)
        bvar = sum((s - bmean) ** 2 for s in sizes) / len(sizes)
        bsd = math.sqrt(bvar)
    else:
        rate, bmean, bsd = 0.0, 3.10, 0.20
    rate = max(0.02, min(1.5, rate))
    bsd = max(0.10, min(0.45, bsd))

    # death rate of the leader cohort
    alive = gone = 0
    for i in range(len(marks) - 1):
        cur = {}
        for row in marks[i].get("rows") or []:
            cap = float(row["market_cap_lamports"]) / SOL_LAMPORTS
            if cap > 0 and math.log10(cap) >= LEADER_LOG:
                cur[row["mint"]] = True
        nxt = set()
        for row in marks[i + 1].get("rows") or []:
            nxt.add(row["mint"])
        for mint in cur:
            alive += 1
            if mint not in nxt:
                gone += 1
    death = (gone / alive) if alive >= 8 else DEATH_LAMBDA
    # never believe a leader is immortal, however clean the history looks
    death = max(0.12, min(0.30, death))

    return {
        "mu": mu, "sd": sd, "rate": rate, "bmean": bmean, "bsd": bsd,
        "death": death,
    }


# --------------------------------------------------------------------------
# the race
# --------------------------------------------------------------------------
def growth(horizon):
    """Saturating drift and spread of a broken-out coin's log size."""
    h = max(0.0, horizon)
    drift = GROW_TOTAL * (1.0 - math.exp(-h / GROW_TAU))
    return drift


def coin_components(coin, params, horizon, cadence):
    """Mixture components (weight, mean, sd) of a named coin's future log size."""
    h = max(0.0, horizon)
    marks = h / cadence if cadence > 0 else h / 10.0
    # a coin that has already survived the early churn is the robust kind, so
    # the cumulative hazard grows sub-linearly in the number of marks
    surv = math.exp(-(marks ** DEATH_BETA) * params["death"])
    drift = growth(h)
    sd = min(SIGMA_MAX, params["sd"] * math.sqrt(max(h, 0.0) / 10.0))
    comps = []
    if coin.log >= LEADER_LOG:
        w = surv * (0.35 if coin.stale else 1.0)
        comps.append((w, coin.log + drift, sd))
        comps.append((1.0 - w, DUD_LOG, DUD_SD))
    elif coin.log >= 1.9:
        # still on its curve but with real traction: may graduate upward
        jump_p = CURVE_JUMP_P * surv
        comps.append((jump_p, params["bmean"] + drift * 0.6, max(0.18, sd * 0.7)))
        keep = (1.0 - jump_p) * (0.45 if coin.stale else 1.0)
        comps.append((keep, coin.log + drift * 0.5, sd))
        comps.append((1.0 - jump_p - keep, DUD_LOG, DUD_SD))
    else:
        # dud: a small chance it is the next breakout
        jump_p = 0.02 * surv
        comps.append((jump_p, params["bmean"] + drift * 0.6, max(0.20, sd)))
        comps.append((1.0 - jump_p, DUD_LOG, DUD_SD))
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
            drift = GROW_TOTAL * (1.0 - math.exp(-max(a - 2.0, 0.0) / GROW_TAU))
            sd = math.sqrt(params["bsd"] ** 2 +
                           (min(SIGMA_MAX, params["sd"] * math.sqrt(max(a - 2.0, 0.0) / 10.0))) ** 2)
            sf = norm_sf((x - (params["bmean"] + drift)) / sd)
            # it also has to still be alive at the read, which bounds the
            # population of old breakouts instead of letting it grow for ever.
            # The newcomer pool is not pre-selected, so it faces the plain
            # per-mark hazard; the coins the history names are the ones that
            # already survived the early churn (see coin_components).
            sf *= math.exp(-(a / 10.0) * params["death"])
            lo = 0.0 if i == 0 else (ages[i - 1] + a) / 2.0
            hi = h if i == len(ages) - 1 else (a + ages[i + 1]) / 2.0
            integ += sf * max(0.0, hi - lo)
        out.append(params["rate"] * integ)
    return xs, out


def race(named, params, horizon, places, chart_size, cadence, edges):
    """Per-place named-coin probabilities, newcomer mass and size bins."""
    # keep the plausible candidates only: leaders plus recent movers
    named.sort(key=lambda c: (-c.log, c.rank))
    named = named[:400]
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
def build_forecast(task, marks):
    places = int(task.get("places", 5))
    chart_size = int(task.get("chart_size", 100))
    edges = [float(e) for e in task["size"]["edges"]]
    cadence = float(task.get("cadence_minutes", 10) or 10)
    reads = [parse_ts(r) for r in task.get("reads", [])]
    if not reads:
        reads = [parse_ts(task["window"]["answer_at"])]
    coins = build_coins(marks)
    params = estimate_params(marks, coins)
    last_at = parse_ts(marks[-1]["at"]) if marks else parse_ts(task["t0"])

    # only the plausible candidates are worth naming
    cand = sorted(coins.values(), key=lambda c: (-c.log, c.rank))
    cand = [c for c in cand if c.log >= 1.9 or c.venue == "pool"][:400]

    acc_coin = [dict() for _ in range(places)]
    acc_new = [0.0] * places
    acc_bins = [[0.0] * (len(edges) + 1) for _ in range(places)]
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

    # ---- sizes ----
    sizes = []
    for k in range(places):
        bins = [v * inv for v in acc_bins[k]]
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
