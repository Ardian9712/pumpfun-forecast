# pumpfun-forecast

make.fun `pumpfun-top5-forecast-01` entry: a forecast of the top 5 new
Solana pump.fun coins by market cap at the house's answer reads.

- `main.py` -- the entry. Python 3.13, standard library only, no network,
  no randomness, byte-identical output on every run. Reads `$MAKEFUN_INPUT_DIR`
  (else `/input`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else
  `/output/forecast.json`).

## Model (v3)

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
  answer read and averaged over the reads. **The per-place size bins are those
  same order statistics and nothing else**: the race that says who takes each
  place also says how big that coin is, so the order half and the size half of
  the answer come from one single model instead of two fitted separately.

v3 keeps v2's order engine unchanged (per-coin momentum, distance-driven
graduation hazard, measured fade, newcomer breakout rate, contention level
3.00) and drops v2's second, separately-fitted size distribution, which was
blended 50/50 into the race's bins. That blend was estimated from the last
three marks only, behind an "is this the pool era yet" heuristic, and it made
the size answer depend on quantities the order model never sees.

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
v2 -26.875 (order -16.787, sizes -10.088),
v3 -27.003 (order -16.787, sizes -10.215), house baseline -37.768.

Read those three numbers honestly: **v3 beats v1 at every cut but does not
beat v2's mean** -- it is 0.128 worse, all of it in the sizes. The local
backtest is only 3 cuts and 6 answer reads, and v2 already taught us that a
backtest gain there did not carry into the practice window, so a 0.5% mean
difference is not evidence either way. What v3 buys is one model instead of
two, with no per-place calibration fitted on three marks. An extra variant
was tried and rejected: shrinking the race's bins toward the place-pooled
marginal made things worse at every weight (w=0.10 ... 0.40, mean -27.09 to
-27.48), because the per-place level gaps are real signal and pooling them
away destroys it.
