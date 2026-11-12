#!/usr/bin/env python3
"""Local scorer: the make.fun log score exactly as RULES-ref.txt describes it.

order score of a read = sum over places k of ln(max(0.0001, p given to the coin
actually at place k)), where p is the forecast's row for that coin at that
place, else the place's unnamed share divided by the number of coins on the
read's chart the forecast did not name at that place.

size score of a read = the same over the size bins.

log score = mean over reads of (order + size); agreement = sum over reads of
the top-5 agreement.
"""

import json
import math
import os
from datetime import datetime, timezone

FLOOR = 0.0001


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_answer(path):
    """answer.json: {"reads": [{"at": ..., "chart": [mint,...], "caps": [lamports,...]}]}"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def score_read(forecast, chart, caps, places, edges):
    probs = {}
    for row in forecast.get("probabilities", []):
        probs[(row["place"], row["mint"])] = row["p"]
    sizes = {}
    for row in forecast.get("sizes", []):
        sizes[(row["place"], row["bin"])] = row["p"]
    chart_set = set(chart)
    named_at = {k: set() for k in range(1, places + 1)}
    for (place, mint) in probs:
        if place in named_at:
            named_at[place].add(mint)

    order = 0.0
    detail = []
    for k in range(1, places + 1):
        mint = chart[k - 1] if k - 1 < len(chart) else None
        p = probs.get((k, mint), None) if mint is not None else None
        if p is None:
            s = sum(probs.get((k, m), 0.0) for m in named_at[k])
            denom = sum(1 for m in chart if m not in named_at[k])
            if denom <= 0:
                denom = 1
            p = max(0.0, 1.0 - s) / denom
        p = max(p, FLOOR)
        order += math.log(p)
        detail.append((k, mint, p))

    sz = 0.0
    for k in range(1, places + 1):
        cap = caps[k - 1] if k - 1 < len(caps) else None
        if cap is None:
            p = 1.0 / (len(edges) + 1)
        else:
            log = math.log10(float(cap) / 1e9)
            b = 0
            while b < len(edges) and log >= edges[b]:
                b += 1
            p = sizes.get((k, b), None)
            if p is None:
                s = sum(sizes.get((k, bb), 0.0) for bb in range(len(edges) + 1))
                p = max(0.0, 1.0 - s) / (len(edges) + 1)
        p = max(p, FLOOR)
        sz += math.log(p)
    return order, sz, detail


def agreement(forecast, chart, places):
    top5 = forecast.get("top5", [])
    tot = 0
    for i, mint in enumerate(top5[:places]):
        if mint in chart:
            j = chart.index(mint)
            if j < places:
                tot += places + 1 - abs(i - j)
    return tot


def score(forecast, answer, places, edges):
    reads = answer["reads"]
    tot_o = tot_s = 0.0
    tot_a = 0
    n = 0
    for r in reads:
        chart = r["chart"][:100]
        caps = r["caps"]
        o, s, _ = score_read(forecast, chart, caps, places, edges)
        tot_o += o
        tot_s += s
        tot_a += agreement(forecast, chart, places)
        n += 1
    if n == 0:
        return None
    return {
        "reads": n,
        "order": tot_o / n,
        "sizes": tot_s / n,
        "log": (tot_o + tot_s) / n,
        "agreement": tot_a,
    }


def main():
    import sys
    fx = sys.argv[1]
    ax = sys.argv[2]
    tx = sys.argv[3] if len(sys.argv) > 3 else "input/task.json"
    forecast = json.load(open(fx))
    answer = load_answer(ax)
    task = json.load(open(tx))
    res = score(forecast, answer, int(task["places"]), task["size"]["edges"])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
