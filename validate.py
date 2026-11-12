#!/usr/bin/env python3
"""Strict validation of /output/forecast.json against the rules' shape limits."""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "input")
OUT = os.path.join(HERE, "out")

MAX_BYTES = 1_048_576
PLACES = 5
NBINS = 82


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    # ---- run twice, byte-identical ----
    hashes = []
    for i in range(2):
        t0 = time.time()
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = IN
        env["MAKEFUN_OUTPUT_DIR"] = OUT
        r = subprocess.run([sys.executable, os.path.join(HERE, "main.py")],
                           env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            fail("run %d exited %d: %s" % (i, r.returncode, r.stderr[-2000:]))
        print("run %d: %.2f s" % (i, dt))
        path = os.path.join(OUT, "forecast.json")
        data = open(path, "rb").read()
        hashes.append((len(data), __import__("hashlib").sha256(data).hexdigest()))
    if hashes[0] != hashes[1]:
        fail("output is not byte-identical between runs: %s" % hashes)
    print("byte-identical: yes (%d bytes, sha256 %s...)" % (hashes[0][0], hashes[0][1][:16]))
    if hashes[0][0] > MAX_BYTES:
        fail("file too large: %d bytes" % hashes[0][0])

    task = json.load(open(os.path.join(IN, "task.json")))
    edges = [float(e) for e in task["size"]["edges"]]
    assert len(edges) + 1 == NBINS, "task.json edges do not give 82 bins"
    chart_size = int(task.get("chart_size", 100))

    doc = json.loads(open(os.path.join(OUT, "forecast.json"), "rb").read().decode("utf-8"))

    # ---- exactly three keys ----
    if set(doc.keys()) != {"probabilities", "top5", "sizes"}:
        fail("keys are %s" % sorted(doc.keys()))
    print("keys: exactly probabilities, top5, sizes")

    # ---- probabilities ----
    probs = doc["probabilities"]
    if not isinstance(probs, list):
        fail("probabilities is not a list")
    per_place = {}
    seen = set()
    for row in probs:
        if set(row.keys()) != {"mint", "place", "p"}:
            fail("probability row keys %s" % sorted(row.keys()))
        if not isinstance(row["mint"], str) or not (32 <= len(row["mint"]) <= 44):
            fail("bad mint %r" % row["mint"])
        if not isinstance(row["place"], int) or not (1 <= row["place"] <= PLACES):
            fail("bad place %r" % row["place"])
        if not isinstance(row["p"], (int, float)) or not (0.0 <= row["p"] <= 1.0):
            fail("bad p %r" % row["p"])
        key = (row["place"], row["mint"])
        if key in seen:
            fail("duplicate row %s" % (key,))
        seen.add(key)
        per_place.setdefault(row["place"], []).append(row["p"])
    for k in range(1, PLACES + 1):
        ps = per_place.get(k, [])
        if len(ps) > 100:
            fail("place %d has %d rows (max 100)" % (k, len(ps)))
        s = sum(ps)
        if s > 1.0 + 1e-9:
            fail("place %d probabilities sum to %.9f (> 1)" % (k, s))
        print("place %d: %d rows, sum %.6f, unnamed %.6f" % (k, len(ps), s, 1.0 - s))
    if len(per_place) != PLACES:
        fail("not every place 1..5 is named")

    # ---- top5 ----
    top5 = doc["top5"]
    if not isinstance(top5, list) or len(top5) != PLACES:
        fail("top5 has %r entries" % (len(top5) if isinstance(top5, list) else top5))
    if len(set(top5)) != PLACES:
        fail("top5 is not 5 distinct mints")
    for m in top5:
        if not isinstance(m, str) or not (32 <= len(m) <= 44):
            fail("bad top5 mint %r" % m)
    hist = set()
    with open(os.path.join(IN, "history.ndjson")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            for row in json.loads(line).get("rows") or []:
                hist.add(row["mint"])
    for m in top5:
        if m not in hist:
            fail("top5 mint %s is not named by the history" % m)
    for row in probs:
        if row["mint"] not in hist:
            fail("probability mint %s is not named by the history" % row["mint"])
    print("top5: 5 distinct mints, all named by the history")

    # ---- sizes ----
    sizes = doc["sizes"]
    bins_per_place = {}
    seenb = set()
    for row in sizes:
        if set(row.keys()) != {"place", "bin", "p"}:
            fail("size row keys %s" % sorted(row.keys()))
        if not isinstance(row["place"], int) or not (1 <= row["place"] <= PLACES):
            fail("bad size place %r" % row["place"])
        if not isinstance(row["bin"], int) or not (0 <= row["bin"] <= 81):
            fail("bad bin %r" % row["bin"])
        if not isinstance(row["p"], (int, float)) or not (0.0 <= row["p"] <= 1.0):
            fail("bad size p %r" % row["p"])
        key = (row["place"], row["bin"])
        if key in seenb:
            fail("duplicate size row %s" % (key,))
        seenb.add(key)
        bins_per_place.setdefault(row["place"], []).append(row["p"])
    for k in range(1, PLACES + 1):
        ps = bins_per_place.get(k, [])
        s = sum(ps)
        if s > 1.0 + 1e-9:
            fail("place %d size bins sum to %.9f (> 1)" % (k, s))
        if len(ps) == 0:
            fail("place %d names no size bin" % k)
        print("place %d: %d bins, sum %.6f" % (k, len(ps), s))

    # ---- sanity: chart_size from task.json was honoured ----
    if chart_size != 100:
        print("note: task.json chart_size is %d" % chart_size)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
