# Pre-registration

This is the pre-registration for the out-of-sample run, **verbatim**. It is the module docstring of
[`code/orb_open_v8.py`](code/orb_open_v8.py), the exact file that produced the result, so the
predictions shipped with the code rather than being written afterwards. The run also printed the
headline prediction, decision rule and void conditions as the first lines of its own log, before
any trade was simulated:

```text
ORB v8 -- PRE-REGISTERED OUT-OF-SAMPLE RUN   MODE=OOS  FRICTION=measured  u500  N=5
  RANK_BY=rvol  REL_R_MIN=0.008  MAX_GAP_ATR=3.0
  PREDICTION net R +0.09 [-0.02,+0.20] | n 2600 | CAGR +8% | MDD 62%
  DECISION   <=0 CLOSES | 0..+0.05 not tradeable | >+0.05 replicated
  VOID IF    stranded>0 | fill<70% | lev-bound<97% | trades/day<1.0
```

*"The window it has never seen" (Part 1) refers to the frozen configuration. The window had already
been used earlier in the project for my first replication; see the README, §9. Part 1 also calls
`MAX_GAP_ATR = 3.0` "off"; it still skipped 105 candidates in the run (README, §5).*

Run names (Pelican, Jumping Orange Frog, etc.) are QuantConnect's auto-generated backtest IDs; see
[`results/`](results/) for what each one is. How the predictions turned out is scored in the
[README](README.md#5-result).

---

```text
OPENING-RANGE BREAKOUT  --  v8, 2026-09-17
THE OUT-OF-SAMPLE RUN.  PRE-REGISTERED.  ONE SHOT.  NO KNOBS.

===========================================================================
PART 0 -- AN INSTRUMENT CORRECTION THAT CHANGES HOW EVERY PRIOR LOG READS
===========================================================================
Every log this project has produced prints a line like

    Win rate 49.1%   mean gross +0.0228R   NET +0.0025R

and every reader (me included) has taken "gross" to mean the edge before
friction, and therefore taken friction to cost the difference: ~0.02R.
That is wrong by an order of magnitude.

  gross_r = (exit_price - sd.entry_price) * direction / r_dollars

sd.entry_price is order_event.fill_price and exit_price is a fill price too.
Both are POST-slippage. So the field named "gross_r" is already net of the
entire fill model; net_r subtracts only commission and borrow. The label has
been hiding the single most important number in the project.

The true decomposition, recovered from the Pelican run's own friction audit
(take 13.75 c/side, close 2.59 c/side, mean price $172.93, R/price ~0.52%):

    true gross edge   +0.206 R      <- decision price to decision price
    slippage          -0.183 R      <- of which adverse selection -0.125 R
    commission+borrow -0.020 R
    -------------------------------
    NET               +0.0025 R

So the opening-range break carries a real gross edge of about +0.21R, and
the fill model eats 89% of it. Adverse selection ALONE eats 61% of gross.

And here is the part that matters. Because R is DEFINED as
0.20 x ATR, and the adverse term is ADVERSE_ATR_FRAC x ATR, the adverse
charge in R units is exactly

    ADVERSE_ATR_FRAC / R_ACCOUNTING_ATR_PCT  =  0.025 / 0.20  =  0.125 R

for EVERY trade, identically, scale-free, independent of the name, the
price, the volatility and the window. One number, chosen by me, never
measured, applied 3,370 times, and worth more than the entire edge.

Solving net_R = 0 for that parameter: the IS champion (+0.1514R) dies at
ADVERSE_ATR_FRAC = 0.0553. The "conservative" preset is 0.050. So this
strategy is alive under one label I invented and dead under another label
I invented, and the two labels differ by 2.5 thousandths of an ATR.

=> SWEEP 4 below re-prices net R across the whole ADVERSE_ATR_FRAC range
   analytically, at zero extra runtime, with an empirical self-check. From
   now on no run reports a single net R; it reports the curve and the
   breakeven. This run's real deliverable is that breakeven.

===========================================================================
PART 1 -- WHAT IS FROZEN, AND WHY THESE VALUES
===========================================================================
This is "Jumping Orange Frog" (IS: net +0.1514R, CAGR 23.5%, Sharpe 0.600,
MDD 36.9%, beta 0.177, PSR 12.7%, 1,205 trades) run forward, unchanged,
on the window it has never seen.

    MODE                  = "OOS"    2020-01-01 .. 2026-06-30
    FRICTION_PRESET       = "measured"
    RANK_BY               = "rvol"    (the paper's spec; NOT the ranker)
    REL_R_MIN             = 0.008     ATR/price >= 4%. THE one variable.
    N_HOLDINGS            = 5
    UNIVERSE_SIZE         = 500
    ENTRY_MODE            = "stop"    take the break
    stop loss             = NONE      3 replications say every stop loses
    exit                  = MOC only  every intraday exit is -0.06..-0.09R
    MAX_GAP_ATR           = 3.0       i.e. off, as in the champion
    ALLOW_SHORTS          = True

KNOWN DEVIATION, stated up front: v8 keeps v7's ATR(60) readiness
requirement (140-day warm-up), which the champion did not have. It cost
~3.5% of trades in-sample (3,370 vs 3,492 in the matched cell). It is kept
because (a) it powers the diagnostic tables and (b) in a window containing
the 2020-21 IPO/SPAC wave, requiring 60 days of history before trading a
name is the conservative choice. Any OOS/IS comparison carries this caveat.

WHAT IS DEAD AND NOT COMING BACK (do not re-test these here):
    RVOL as a SELECTOR    failed to sort 8 times, every mechanism/universe
    ATR5/ATR60 ranking    v7 selected LOW spike and low spike came back
                          NEGATIVE while the top bucket was best by 6x.
                          The v6 monotone sort inverted under selection.
    MAX_GAP_ATR           bucket table is noise in both directions
    profit targets        flat and non-monotone (+4R +0.006, +6R -0.001)
    intraday exits        uniformly -0.06 to -0.09R vs MOC
    N=10                  ranks 6-10 are -0.0347R each
    market entry @09:35   negative gross: the BREAK is the signal
    long-only             tempting (+0.0517 vs -0.0496) but t=1.07. No.

===========================================================================
PART 2 -- PREDICTIONS.  WRITTEN BEFORE THE RUN.  NOT NEGOTIABLE AFTER.
===========================================================================
sigma(R) ~ 2.74 (from the IS 5th percentile of -4.51R under normality; the
true sigma is larger because the left tail is fat, so every t below is an
UPPER bound on significance).

PRIMARY
  mean net R          point +0.09      80% interval [-0.02, +0.20]
      reasoning: IS +0.1514 at t=1.92 shrinks by ~1-1/t^2 = 0.73 for
      sampling error, then ~0.8 again for having picked the 0.008
      threshold after seeing six buckets: 0.1514 x 0.73 x 0.8 = 0.088.
      Regime adds little: adverse selection is scale-free in R, so higher
      2020-22 volatility does NOT cheapen friction in R terms -- only the
      half-spread component falls, worth ~+0.01R, plus ~+0.01R because
      spreads genuinely narrowed after 2019.

  trades              point 2,600      80% interval [1,800, 3,800]
                      (IS ran 1.2/day; higher vol should widen the
                       qualifying pool to ~1.6/day over 1,635 sessions)

  *** PRE-REGISTERED ADMISSION: at n=2,600, SE = 0.054, so my own central
  *** prediction of +0.09R lands at t=1.67. If I am exactly right, this
  *** run STILL does not reach significance. The decision rule below is
  *** therefore a pre-committed threshold, not a significance test. I am
  *** writing this down now so I cannot later pretend t=1.7 was a pass.

SECONDARY
  CAGR                point  +8%      interval [0%, +18%]   (SPY ~+14%)
  Sharpe              point  0.33     interval [0.05, 0.60]
  max drawdown        point  62%      interval [45%, 78%]
  worst single day    worse than -18%, and it lands in Feb-Mar 2020
                      or Jan-Feb 2021
  breakeven ADVERSE_ATR_FRAC (Sweep 4)
                      point 0.043     interval [0.021, 0.065]
  INTEGRITY CHECK     clean, 0 stranded; ~25% chance of a margin event
                      in March 2020 at 3.6x target gross

THE THREE THAT ACTUALLY TEACH SOMETHING
  (a) 2020 calendar net R > full-period net R. Highest vol, widest pool.
  (b) In 2022, shorts net R > longs net R -- for the first time. If shorts
      STILL lose money in a -19% year, the long/short asymmetry is
      structural rather than a bull-tape artifact, and that is a larger
      finding than the headline number.
  (c) 2023-2026 breadth falls below 1.0 trades/day as dispersion collapses
      into mega-cap. If breadth HOLDS, Option A (universe expansion) is
      unnecessary and the champion scales as-is.

===========================================================================
PART 3 -- DECISION RULE.  COMMITTED IN ADVANCE.
===========================================================================
  net R <= 0              The IS result was selection noise. ORB-on-R/price
                          CLOSES. No further backtests, no paper trade.

  0 < net R <= +0.05      Alive, but breakeven ADVERSE_ATR_FRAC < 0.035 --
                          inside the uncertainty of a parameter never
                          measured. Verdict: NOT tradeable. The only
                          remaining useful work is measuring real adverse
                          selection from live quotes. That is a data
                          project, not a backtest.

  net R > +0.05           Replicated. Proceed to (i) Option A breadth work
                          and (ii) a live fill-measurement program at
                          minimum size on ONE name.

VOID CONDITIONS -- if any fires, the run is not evidence and no table below
it may be quoted or interpreted:
    INTEGRITY CHECK not clean, or stranded positions > 0
    fill rate < 70%
    leverage-bound < 97%
    trades/day < 1.0   (the log's own breadth floor; note that the IS
                        champion ran 1.2/day and only barely passed it)

OVERRIDE, independent of net R:
    max drawdown > 70%  ->  untradeable at 3.6x target gross whatever the
    edge. Any positive result must then be restated at the sizing that
    caps MDD at 25%, which divides CAGR by the same factor.

===========================================================================
PART 4 -- WHAT I COMMIT NOT TO DO
===========================================================================
 1. Not re-run OOS with any knob changed. One shot. If it fails, it fails.
 2. Not mine the OOS bucket tables for a new config. They are for
    understanding only; any hypothesis they suggest must be tested on a
    third window or forward, never on this one.
 3. Not quote Sweeps 1-4 as results. They select rules to TEST.
 4. Not reinterpret a negative result as "the friction model was too
    harsh." If I want to make that argument I must first measure adverse
    selection independently. Sweep 4 exists precisely so that the
    sensitivity is reported whether the result is good or bad.
 5. Not compare an OOS bucket table against an IS bucket table without
    restating the ATR(60) population caveat above.

RUN ORDER
---------
  1. THIS FILE, unchanged.  MODE="OOS", measured.  ~60-70 min at u500.
     Nothing else runs until its log has been read.
```
