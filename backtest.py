#!/usr/bin/env python3
"""Backtest: train on the earlier marks, score the later ones.

For each cut c the forecast is built from marks 0..c only and scored against
every later mark as if it were an answer read (its top-100 chart and its top-5
market caps).  Baselines scored on the very same reads:

  * do-nothing  : the last mark's order holds, p=1 on it, and p=1 on the bin
                  the last mark's cap at each place falls in.
  * house       : the house baseline's fixed numbers (0.30 on the then #1 and
                  0.10 on the then #2 at place 1; 0.30 on the then #k and 0.10
                  on #k-1 and #k+1 at place k; normal sizes, sd 0.35).
"""

import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M  # noqa: E402
import scorer as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(in_dir):
    task = json.load(open(os.path.join(in_dir, "task.json")))
    marks = []
    with open(os.path.join(in_dir, "history.ndjson")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                marks.append(json.loads(line))
    return task, [m for m in marks if m.get("rows")]


def pseudo_answer(marks, idx_from, places):
    reads = []
    for m in marks[idx_from:]:
        rows = m["rows"]
        reads.append({
            "at": m["at"],
            "chart": [r["mint"] for r in rows[:100]],
            "caps": [int(r["market_cap_lamports"]) for r in rows[:places]],
        })
    return {"reads": reads}


def baseline_nothing(marks, places, edges):
    last = marks[-1]["rows"]
    probs = []
    for k in range(1, places + 1):
        if k - 1 < len(last):
            probs.append({"mint": last[k - 1]["mint"], "place": k, "p": 1.0})
    sizes = []
    for k in range(1, places + 1):
        if k - 1 < len(last):
            cap = int(last[k - 1]["market_cap_lamports"])
            log = math.log10(cap / 1e9)
            b = 0
            while b < len(edges) and log >= edges[b]:
                b += 1
            sizes.append({"place": k, "bin": b, "p": 1.0})
    return {
        "probabilities": probs,
        "top5": [r["mint"] for r in last[:places]],
        "sizes": sizes,
    }


def baseline_house(marks, places, edges):
    last = marks[-1]["rows"]
    probs = []
    for k in range(1, places + 1):
        if k == 1:
            if len(last) >= 1:
                probs.append({"mint": last[0]["mint"], "place": 1, "p": 0.30})
            if len(last) >= 2:
                probs.append({"mint": last[1]["mint"], "place": 1, "p": 0.10})
        else:
            if len(last) >= k:
                probs.append({"mint": last[k - 1]["mint"], "place": k, "p": 0.30})
            if len(last) >= k - 1:
                probs.append({"mint": last[k - 2]["mint"], "place": k, "p": 0.10})
            if len(last) >= k + 1:
                probs.append({"mint": last[k]["mint"], "place": k, "p": 0.10})
    sizes = []
    for k in range(1, places + 1):
        if k - 1 >= len(last):
            continue
        cap = int(last[k - 1]["market_cap_lamports"])
        mu = math.log10(cap / 1e9)
        sd = 0.35
        acc = []
        tot = 0.0
        for b in range(len(edges) + 1):
            lo = edges[b - 1] if b >= 1 else -1e9
            hi = edges[b] if b < len(edges) else 1e9
            p = S_norm_cdf((hi - mu) / sd) - S_norm_cdf((lo - mu) / sd)
            p = math.floor(p * 1e6) / 1e6
            acc.append((b, p))
            tot += p
        if tot > 1.0:
            acc = [(b, p / tot) for b, p in acc]
        for b, p in acc:
            if p > 0:
                sizes.append({"place": k, "bin": b, "p": p})
    return {
        "probabilities": probs,
        "top5": [r["mint"] for r in last[:places]],
        "sizes": sizes,
    }


def S_norm_cdf(z):
    return 0.5 * math.erfc(-z * 0.7071067811865476)


def run(in_dir="input", cuts=None, verbose=True):
    task, marks = load(in_dir)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    if cuts is None:
        # every cut that leaves at least one later mark to score against
        cuts = [c for c in range(1, len(marks) - 1)]
    rows = []
    for c in cuts:
        train = marks[: c + 1]
        sub_task = dict(task)
        sub_task["reads"] = [m["at"] for m in marks[c + 1:]]
        if not sub_task["reads"]:
            continue
        fc = M.build_forecast(sub_task, train)
        ans = pseudo_answer(marks, c + 1, places)
        mine = S.score(fc, ans, places, edges)
        noth = S.score(baseline_nothing(train, places, edges), ans, places, edges)
        house = S.score(baseline_house(train, places, edges), ans, places, edges)
        rows.append((c, mine, noth, house))
        if verbose:
            print("cut %d (%d marks, %d test reads): mine order %+.3f sizes %+.3f log %+.3f agr %d | "
                  "nothing %+.3f | house %+.3f"
                  % (c, len(train), mine["reads"], mine["order"], mine["sizes"],
                     mine["log"], mine["agreement"], noth["log"], house["log"]))
    if not rows:
        return None
    agg = {}
    for key in ("order", "sizes", "log", "agreement"):
        agg[key] = sum(r[1][key] for r in rows) / len(rows)
    n_h = sum(r[3]["log"] for r in rows) / len(rows)
    n_n = sum(r[2]["log"] for r in rows) / len(rows)
    print("\nmean over %d cuts: mine order %+.3f sizes %+.3f log %+.3f agreement %.1f"
          % (len(rows), agg["order"], agg["sizes"], agg["log"], agg["agreement"]))
    print("mean over %d cuts: do-nothing log %+.3f | house-baseline log %+.3f"
          % (len(rows), n_n, n_h))
    print("house baseline's published practice score: -40.408345 (order -21.380389, sizes -19.027956)")
    return {"mine": agg, "nothing": n_n, "house": n_h, "rows": rows}


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "input"
    run(d)
