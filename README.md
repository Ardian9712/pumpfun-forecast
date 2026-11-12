# pumpfun-forecast

make.fun `pumpfun-top5-forecast-01` entry: a forecast of the top 5 new
Solana pump.fun coins by market cap, 24 hours out.

- `main.py` -- the entry. Python 3.13, standard library only, no network,
  no randomness, byte-identical output on every run. Reads `$MAKEFUN_INPUT_DIR`
  (else `/input`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else
  `/output/forecast.json`).
