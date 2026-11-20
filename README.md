# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Once-debug/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Once-debug/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Node 22 program that forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the earn rate Exophase will print for it one week after launch — the percent of tracked players who have earned it — as five quantiles (5/25/50/75/95%). Its "Second Week Curve" model fits each trophy's earn-rate share with a saturating exponential that may rise toward a ceiling or fall toward a floor, blends that fit on the logit scale with a damped log-time trend and with persistence, and gives every trophy its own spread, widened by the horizon and by how jumpy its recorded rates have been. The model was calibrated on rehearsal worlds built like this contest's data — saturating player growth, per-trophy penetration curves, step refreshes, a weekend surge — where it out-scores both house baselines; the assumptions are written out in the header of `main.ts`.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit input dir> MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6fmsan4y&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## What it reads and writes

`main.ts` (Node 22, standard library only, no network) reads every `*.in` case file in the
input directory (`$MAKEFUN_INPUT_DIR`, else `/input`) in name order — one JSON object,
schema `makefun/gta6-trophies-case/v1`: the case's trophies and every recorded read of
the game's Exophase trophy page before the checkpoint, each read holding every trophy's
earn rate and the players-tracked count. For each case it writes `<id>.out` to the output
directory (`$MAKEFUN_OUTPUT_DIR`, else `/output`): `{"trophies":[{"id": n, "q":
[q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, quantiles non-decreasing, each
0–100. The program is a pure function of the case file — no clock, no randomness — so
two runs on the same case write the same bytes, well inside the 8-second cap.

## The model in brief

- **Share fit (weight 0.40).** Per trophy, a saturating exponential
  `share(t) = S − A·exp(−a·(t − t0))` fitted by least squares over a fixed grid of growth
  rates `a` (closed-form in `(S, A)` given `a`). The sign of `A` is free: a trophy
  progression is winning rises toward its ceiling; one dilution is winning falls toward
  its floor. Fitting the share, not the earner count, is deliberate — on rehearsal data
  the count is a product of two saturating curves and a single exponential kept
  projecting growth that had already happened.
- **Damped trend (weight 0.20).** A least-squares line of the logit rate against
  `ln(1 + hours since release)`, read only halfway from the last read to the target,
  because earn-rate curves bend.
- **Persistence (weight 0.40).** The last read's rate. It earns its full share of the
  vote: fitted curves left alone kept overshooting moves that were already mostly spent.
- The blended median may move at most **1.4 logit units** from the last rate.
- **Spread.** Per trophy, on the logit scale: a base that widens with the horizon,
  scaled by the trophy's own trend-fit residual, plus a small step allowance when the
  recorded rates take many distinct values; floored at 0.45, capped at 1.05.
- A trophy with no earners yet gets a small tier prior for its median (platinum 0.04%,
  gold 0.25%, silver 0.6%, bronze 1.2%) instead of exactly zero.
