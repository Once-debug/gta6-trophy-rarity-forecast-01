// gta6-trophy-rarity-forecast-01 — entry "Second Week Curve" (Node 22, standard library only).
//
// What it does: reads one case file (makefun/gta6-trophies-case/v1) per run from the input
// directory and writes, for every trophy of the case, five quantiles (5/25/50/75/95%) of the
// trophy's earn rate on Exophase at the case's target, as <id>.out in the output directory.
//
// The model. A printed earn rate is a share: of the players Exophase tracks, the fraction
// who have earned the trophy. Over a launch week that share is pulled two ways — new,
// less-engaged players keep arriving and dilute it, while the players already there keep
// progressing and push it up — and it settles toward a ceiling (or a floor, for a trophy
// dilution is winning). So the central opinion fits the share itself with a saturating
// exponential, share(t) = S - A * exp(-a * (t - t0)), the growth rate `a` picked from a
// fixed grid by least squares (given `a` the fit is linear in (S, A): closed-form, and
// deterministic). The sign of A is free, so the fit can rise toward a ceiling or fall
// toward a floor. (An earlier draft of this program fitted the earner count instead; on
// rehearsal data that count is a product of two saturating curves — players arriving,
// players progressing — and a single exponential kept projecting growth that had already
// happened. The share is the better-behaved object, so the count fit is gone.)
//
// The share fit is blended on the logit scale with two cheaper opinions: a damped trend
// (a least-squares line of the logit rate against ln(1 + hours since release), read only
// 30% of the way from the last read to the target, because earn-rate curves bend) and
// plain persistence (the last read's rate). Weights: share 0.20, trend 0.05,
// persistence 0.75 — persistence carries the vote because on rehearsal worlds the
// fitted curves, left alone, kept overshooting moves that were already mostly spent.
// The blend's move away from the last rate is capped at 0.8 logit units so no single
// fit can run away with a trophy.
//
// The spread is per trophy: a base that widens with the horizon, scaled by the trophy's
// own trend-fit residual (a trophy whose logit rate has been jumpy gets a wider band),
// plus a small step allowance when the recorded rates take a few distinct values
// (Exophase refreshes in steps, and a step between the checkpoint and the target is the
// main risk a narrow band misses). It is floored at 0.45 and capped at 2.05 logit
// units. The widths were set on rehearsal worlds built like this contest's data
// (saturating player growth, per-trophy penetration curves, step refreshes, a weekend
// surge), then re-tuned in a second round over five world families with held-out
// worlds the tuning never saw: the wider bands and the heavier persistence weight
// held up on the held-out worlds, the narrower first-round bands did not — a band
// that one unseen refresh can jump clean over is the expensive mistake here.
//
// Assumptions, stated plainly:
//  - Exophase refreshes its counts in steps, so the recorded reads are step functions of
//    the real processes; the fits smooth over the steps instead of chasing them.
//  - A share's move over the remaining days is mostly the continuation of the move the
//    reads already show, bent toward level: that is what the saturating fit and the
//    damped trend both encode, in different currencies.
//  - A trophy nobody has earned yet gets a small tier prior for its median instead of 0,
//    because a week of play almost never leaves a trophy at exactly zero.
//  - Nothing here is random: no clock, no seed, no network. Two runs on the same case
//    write the same bytes.
//
// Run locally with:
//   mkdir -p out && MAKEFUN_INPUT_DIR=<kit input dir> MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level (more digits than the baseline's). */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

interface TrophyMeta {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  game: { release_date: string };
  trophies: TrophyMeta[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clampPct(pct) / (100 - clampPct(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals, as the house baseline rounds. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Median earn rate (percent) assumed for a trophy with no earners yet, by tier. */
const TIER_PRIOR: Record<string, number> = { platinum: 0.15, gold: 0.1, silver: 0.6, bronze: 1.2 };

/**
 * Fit v(t) = V - A * exp(-a * (t - t0)) to (ts, vs) by least squares over a fixed grid
 * of rates `a` (per hour). Given `a`, the model is linear in (V, A), so each grid point
 * is one 2x2 solve; the best-SSE point wins. Returns a predict function, or null when
 * the series is too short or every solve is degenerate.
 */
function fitSaturating(ts: number[], vs: number[]): ((t: number) => number) | null {
  const n = ts.length;
  if (n < 3) return null;
  const t0 = ts[0];
  let best: { sse: number; v: number; a: number; amp: number } | null = null;
  for (let gi = 0; gi <= 48; gi++) {
    const a = 0.0008 * Math.pow(1.16, gi); // 0.0008 .. ~1.06 per hour
    // Normal equations for v ≈ V + A * u, u = -exp(-a (t - t0)).
    let su = 0, sv = 0, suu = 0, suv = 0;
    for (let k = 0; k < n; k++) {
      const u = -Math.exp(-a * (ts[k] - t0));
      su += u; sv += vs[k]; suu += u * u; suv += u * vs[k];
    }
    const det = n * suu - su * su;
    if (Math.abs(det) < 1e-12) continue;
    const v = (sv * suu - su * suv) / det;
    const amp = (n * suv - su * sv) / det;
    let sse = 0;
    for (let k = 0; k < n; k++) {
      const pred = v + amp * -Math.exp(-a * (ts[k] - t0));
      sse += (vs[k] - pred) * (vs[k] - pred);
    }
    if (best === null || sse < best.sse) best = { sse, v, a, amp };
  }
  if (best === null) return null;
  const { v, a, amp } = best;
  return (t: number) => v + amp * -Math.exp(-a * (t - t0));
}

/**
 * Damped trend opinion for one trophy: least-squares line of logit(rate) on
 * x = ln(1 + hours since release), evaluated 30% of the way from the last read's x to
 * the target's x. Returns { level, residSd } in logit units, or null when fewer than
 * three usable reads (rates strictly inside (0, 100)) exist.
 */
function dampedTrend(xs: number[], ys: number[], xEval: number): { level: number; residSd: number } | null {
  const n = xs.length;
  if (n < 3) return null;
  let sx = 0, sy = 0;
  for (let k = 0; k < n; k++) { sx += xs[k]; sy += ys[k]; }
  const mx = sx / n, my = sy / n;
  let sxx = 0, sxy = 0;
  for (let k = 0; k < n; k++) { sxx += (xs[k] - mx) * (xs[k] - mx); sxy += (xs[k] - mx) * (ys[k] - my); }
  if (sxx < 1e-12) return null;
  const b1 = sxy / sxx;
  const b0 = my - b1 * mx;
  let sse = 0;
  for (let k = 0; k < n; k++) { const r = ys[k] - (b0 + b1 * xs[k]); sse += r * r; }
  const residSd = Math.max(0.05, Math.sqrt(sse / Math.max(1, n - 2)));
  return { level: b0 + b1 * xEval, residSd };
}

/** Forecast one case and return its output rows. */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const days = c.horizon_minutes / 1440;
  const hist = c.history;
  const last = hist[hist.length - 1];
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const tLastMs = Date.parse(last.at);
  const tTargetMs = tLastMs + c.horizon_minutes * 60_000;
  const tTargetH = tTargetMs / 3_600_000;
  const xOf = (ms: number): number => Math.log(1 + Math.max(0, (ms - releaseMs) / 3_600_000));
  const xLast = xOf(tLastMs);
  const xEval = xLast + 0.3 * (xOf(tTargetMs) - xLast);

  return c.trophies.map((t, i) => {
    // This trophy's observed series, oldest first: times, rates, trend points.
    const ts: number[] = [];
    const shares: number[] = [];
    const trendX: number[] = [];
    const trendY: number[] = [];
    let lastRate: number | null = null;
    let earners = 0;
    for (const r of hist) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      ts.push(Date.parse(r.at) / 3_600_000);
      shares.push(rate);
      lastRate = rate;
      earners = (rate * r.players_tracked) / 100;
      if (rate > 0 && rate < 100) {
        trendX.push(xOf(Date.parse(r.at)));
        trendY.push(logit(rate));
      }
    }
    const trend = dampedTrend(trendX, trendY, xEval);

    let median: number;
    let sigmaBase: number;
    if (lastRate === null || earners <= 0) {
      median = TIER_PRIOR[t.tier] ?? 0.5;
      sigmaBase = 0.45; // an unobserved trophy could land almost anywhere low
    } else {
      // Share fit: saturating exponential on the printed rates themselves.
      const sFit = fitSaturating(ts, shares);
      let shareTarget = sFit ? sFit(tTargetH) : lastRate;
      shareTarget = Math.min(99.5, Math.max(0.01, shareTarget));
      // Blend the three opinions on the logit scale, cap the move from the last rate.
      const anchor = logit(lastRate);
      let m = 0.2 * logit(shareTarget) + 0.75 * anchor;
      m += trend ? 0.05 * trend.level : 0.05 * anchor;
      m = Math.min(Math.max(m, anchor - 0.8), anchor + 0.8);
      median = invLogit(m);
      sigmaBase = 0.02 + 0.3 * (trend ? trend.residSd : 0.3);
    }

    // Spread: horizon growth, the trophy's own jumpiness, and a step allowance when the
    // recorded rates take many distinct values (refresh steps are the tail risk).
    const distinct = new Set(shares.map((s) => Math.round(s * 100) / 100)).size;
    const jumpy = distinct >= 3 && distinct < 6 ? 0.06 : 0;
    const sigma = Math.min(2.05, Math.max(0.45, 1.45 * (0.22 + sigmaBase * (0.6 + 1.15 * Math.sqrt(days))) + jumpy));
    const m = logit(median);
    return { id: t.id, q: Z.map((z) => round4(invLogit(m + z * sigma))) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
