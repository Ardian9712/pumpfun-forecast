# pumpfun-forecast

make.fun `pumpfun-top5-forecast-01` entry: a forecast of the top 5 new
Solana pump.fun coins by market cap at the house's answer reads.

- `main.py` -- the entry. Python 3.13, standard library only, no network,
  no randomness, byte-identical output on every run. Reads `$MAKEFUN_INPUT_DIR`
  (else `/input`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else
  `/output/forecast.json`).

## Model (v2)

The leaderboard is a race in log10(market cap in SOL). Every coin that counts
is a dud (pinned near its launch cap), a curve coin still climbing its bonding
curve, or a graduated pool coin; pool coins hold essentially every top place.

* Each named coin's log size a horizon h out is a mixture of: keep growing,
  fade out of the race, or - for curve coins - jump when it graduates.
  A leader's growth is its own recent momentum (recency weighted, shrunk
  toward the population's leader growth), saturating with the horizon.
* A curve coin graduates with a per-mark hazard that grows with its distance
  to the completion level, landing either at the breakout level or at a dead
  one, as the history shows.
* Coins born after the cut are an inhomogeneous Poisson process of
  breakouts: a rate per minute measured from the marks' graduated counters
  and the observed crossings of the contention level, times the distribution
  of the log size a breakout coin reaches at a given age. That mass is left
  unnamed on purpose, exactly as the rules reward.
* The place distributions are the order statistics of the race, computed per
  answer read and averaged over the reads. The per-place size bins are the
  race's own bins blended with the measured per-place size distribution: its
  level growth and its gaps to the leader.

Everything about the window (deadline, reads, places, size edges, chart size)
is read from task.json at runtime; every other quantity is estimated from the
history at runtime.

## Local checks

- `python3 validate.py` -- runs the entry twice, checks the output is
  byte-identical and inside the rules' shape limits (needs `input/` with the
  kit's `task.json` and `history.ndjson`).
- `python3 backtest.py input` -- trains on the earlier marks and scores the
  later ones, against the do-nothing and house baselines.

Backtest mean log score: v1 -32.016 (order -18.748, sizes -13.268),
v2 -26.875 (order -16.787, sizes -10.088), house baseline -37.768.
