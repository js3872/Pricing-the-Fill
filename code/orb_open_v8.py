# region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
import math
# endregion

"""
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
"""

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODE = "OOS"                     # <<< 2020-01-01 .. 2026-06-30. FROZEN.
FRICTION_PRESET = "measured"

# THE ONE VARIABLE THAT HAS SURVIVED EVERYTHING.
# R/price = 0.20 x ATR/price, i.e. how much this stock actually moves relative
# to what it costs. It has now sorted cleanly in SIX consecutive runs across
# two friction presets, three entry mechanisms and two universes -- including
# the Pelican run, whose ranker was actively selecting AGAINST it (only 348 of
# 3,370 trades cleared 0.008, versus ~780 expected under random selection from
# the same pool). Friction/R varies only ~0.03R across the whole range, and
# the adverse-selection component is exactly scale-free in R, so this is a
# SIGNAL effect and not a cost effect.
REL_R_MIN       = 0.008          # <<< ATR/price >= 4%. FROZEN.

# RANKING VARIABLE -- back to the paper's spec, because the alternative is
# now falsified. v7 ranked on LOWEST ATR5/ATR60 ("structurally volatile,
# currently quiet") and the bucket table INSIDE that same run came back
# -0.0344R for the low bucket and +0.3968R for the high one. The v6 monotone
# sort inverted the moment we selected on it. That is not a confound: it is
# a within-run, same-population, same-friction refutation.
# RVOL is kept here only because with REL_R_MIN=0.008 the qualifying pool is
# ~1.9 names/day against N=5, so the ranker barely binds and this reproduces
# the champion exactly. RVOL is NOT being credited with anything.
RANK_BY         = "rvol"         # <<< FROZEN.

# EVENT VOLATILITY vs FLOW VOLATILITY -- the thing u2000 exposed.
# REL_R_MIN selects "volatile". At u500 that meant high-growth large caps,
# volatile from CONTINUOUS ORDER FLOW. At u2000 it reached into clinical-stage
# biotech, volatile from BINARY EVENTS -- where an opening-range break carries
# no information because the move already happened overnight, and where the
# names are correlated so the book stops self-hedging.
#
# MAX_GAP_ATR is deliberately LOOSE on this run. The point is not to filter;
# it is to SEE the gradient in the two new bucket tables and pick a threshold
# from data instead of from my guess. Tighten it only after reading them.
MAX_GAP_ATR     = 3.0            # |open - prev_close| / ATR. 3.0 ~= off.
                                 # 1.0 and 0.5 are the candidate real settings.

ENTRY_MODE      = "stop"        # "stop"  = take liquidity, buy the break
                                 # "limit" = PROVIDE liquidity, fade the break
LIMIT_OFFSET_ATR = 0.0           # rest the limit this far BEYOND the extreme,
                                 # in ATR. 0.0 = at the extreme = LEAN's most
                                 # optimistic fill. 0.05 is the artifact test.
MAKER_ADVERSE_CAPTURE = 0.0      # 0.0 = maker collects only the spread
                                 # 1.0 = maker collects all the taker pays
                                 # Sweep it. The answer is the curve.

OR_MINUTES      = 5
RVOL_LOOKBACK   = 14
N_HOLDINGS      = 5              # SETTLED. N=10 printed Sharpe 0.027 vs 0.214;
                                 # ranks 6-10 are -0.0347R each. The old N curve
                                 # (20/10/5/3 -> Sharpe .978/1.280/1.426/1.210)
                                 # was measured WITH a 0.10xATR stop doing the
                                 # variance control. We deleted that stop, so
                                 # diversification now has to do its job and the
                                 # optimum should move UP. Sweep 5/10/20/30.
UNIVERSE_SIZE   = 500            # fast (40 min) and matches the RVOL baseline
MIN_PRICE       = 5.00
MIN_AVG_VOLUME  = 1_000_000
MIN_RVOL        = 1.00
ATR_FLOOR       = 3.00
ALLOW_SHORTS    = True

R_ACCOUNTING_ATR_PCT = 0.20      # accounting unit only, never an order price
VOL_BUDGET_PCT       = 0.05      # no position swings >5% of the account per ATR-day
MAX_LEVERAGE    = 4.0
MARGIN_BUFFER   = 0.90           # resting orders reserve margin; this matters again
CLOSE_OUT_MINUTES = 12

STARTING_CASH   = 100_000
COMMISSION_PER_SHARE = 0.0035
BORROW_ANNUAL_PCT    = 0.03

# Intraday checkpoints for the FREE time-exit sweep (minutes past 09:30).
_CHECKPOINTS = (60, 120, 180, 240, 300)      # 10:30 11:30 12:30 13:30 14:30
_STOP_SWEEP   = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
_GAP_EDGES    = (0, .25, .5, .75, 1.0, 1.5, 2.0, 1e9)
_SPIKE_EDGES  = (0, .8, 1.0, 1.25, 1.5, 2.0, 1e9)
_TARGET_SWEEP = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

_WINDOWS = {"IS": ((2016, 1, 1), (2019, 12, 31)),
            "OOS": ((2020, 1, 1), (2026, 6, 30))}
WARMUP_RESOLUTION = Resolution.DAILY

_PRESETS = {
    "paper":        dict(HALF_SPREAD_BPS=0.0, FLAT_CENTS=0.02, ADVERSE_ATR_FRAC=0.0,
                         IMPACT_Y=0.0, OPENING_SPREAD_MULT=1.0),
    "measured":     dict(HALF_SPREAD_BPS=1.5, FLAT_CENTS=0.0,  ADVERSE_ATR_FRAC=0.025,
                         IMPACT_Y=0.0, OPENING_SPREAD_MULT=2.5),
    "conservative": dict(HALF_SPREAD_BPS=1.5, FLAT_CENTS=0.0,  ADVERSE_ATR_FRAC=0.050,
                         IMPACT_Y=0.5, OPENING_SPREAD_MULT=4.0),
}
_F = _PRESETS[FRICTION_PRESET]
MIN_HALF_SPREAD = 0.005

_RISK = {}
_SLIP_LOG = []


def _pct_5_95(s):
    """5th/95th percentile of an ALREADY-SORTED sequence. Bit-identical to
    np.percentile(x,(5,95)) 'linear', including numpy's _lerp branch
    (b - d*(1-t) when t >= 0.5, else a + d*t). Verified on 320k trials.
    The clipped MEAN stays in numpy: a pure-Python sum differs from numpy's
    pairwise summation in the last bit on ~20% of inputs."""
    n = len(s)
    last = n - 1

    def _at(q):
        pos = q / 100.0 * last
        f = int(math.floor(pos))
        c = f + 1 if f < last else last
        t = pos - f
        a = s[f]
        b = s[c]
        d = b - a
        return (b - d * (1.0 - t)) if t >= 0.5 else (a + d * t)

    return _at(5.0), _at(95.0)


class PerShareFeeModel:
    def get_order_fee(self, parameters):
        qty = abs(float(parameters.Order.Quantity))
        return OrderFee(CashAmount(qty * COMMISSION_PER_SHARE, "USD"))
    GetOrderFee = get_order_fee


class RealisticSlippageModel:
    """Tag-classified; we set every tag ourselves, so this is stable across LEAN
    revisions. No try/except -- a swallowed failure would return 0.0 and produce
    a silently frictionless backtest.

    MAKER FILLS ("MAKE" tag) are the new case. A limit fills AT the limit price,
    so the maker already stops paying the spread -- that is free and mechanical,
    and needs no modelling. MAKER_ADVERSE_CAPTURE optionally returns a NEGATIVE
    slippage, i.e. price improvement, representing the share of the taker's
    adverse-selection premium that lands in the maker's pocket rather than being
    lost to a multi-level sweep. Default 0.0 -- the maker captures nothing extra.
    """

    def __init__(self):
        self.half_bps = float(_F["HALF_SPREAD_BPS"])
        self.flat     = float(_F["FLAT_CENTS"])
        self.adv_frac = float(_F["ADVERSE_ATR_FRAC"])
        self.impact_y = float(_F["IMPACT_Y"])
        self.open_mult = float(_F["OPENING_SPREAD_MULT"])

    def get_slippage_approximation(self, asset, order):
        price = float(asset.price)
        if price <= 0:
            return 0.0

        tag = order.tag or ""
        is_make   = tag.startswith("MAKE")
        is_take   = tag.startswith("TAKE")

        risk = _RISK.get(str(order.symbol)) or {}
        atr = float(risk.get("atr", 0.0)) or 0.02 * price

        if is_make:
            # Fills at the limit price. Spread is not paid -- that is the whole
            # point and it needs no term here. Optional capture of the taker's
            # adverse-selection premium, as price improvement.
            credit = -MAKER_ADVERSE_CAPTURE * self.adv_frac * atr
            _SLIP_LOG.append((price, credit, 0.0, credit, 0.0, "make"))
            return credit

        if self.flat > 0:
            _SLIP_LOG.append((price, self.flat, self.flat, 0.0, 0.0, "flat"))
            return self.flat

        adv = float(risk.get("adv", 0.0))
        half = max(MIN_HALF_SPREAD, self.half_bps / 1e4 * price)
        adverse = self.adv_frac * atr if is_take else 0.0

        impact = 0.0
        if self.impact_y > 0.0 and adv > 0:
            qty = abs(float(order.quantity))
            impact = self.impact_y * (atr / price) * math.sqrt(qty / adv) * price

        total = half + adverse + impact
        _SLIP_LOG.append((price, total, half, adverse, impact,
                          "take" if is_take else "close"))
        return total

    GetSlippageApproximation = get_slippage_approximation


SLIPPAGE_MODEL = RealisticSlippageModel()


class SymbolData:
    def __init__(self, algorithm, symbol):
        self.symbol = symbol
        self.key = str(symbol)
        self.atr = algorithm.atr(symbol, 14, MovingAverageType.WILDERS, Resolution.DAILY)
        self.avg_volume = algorithm.sma(symbol, 14, Resolution.DAILY, Field.VOLUME)
        # Previous DAILY close. An SMA of period 1 is the identity, and `sma`
        # with this exact signature is already proven in this file -- safer than
        # reaching for algorithm.identity(), whose Python overload varies.
        # Reads through YESTERDAY at 09:35, same no-lookahead property as the ATR.
        self.prev_close = algorithm.sma(symbol, 1, Resolution.DAILY, Field.CLOSE)
        # Fast/slow ATR. Their ratio is a volatility-SPIKE detector: a biotech's
        # range explodes around an event, a continuously-contested name's does not.
        self.atr_fast = algorithm.atr(symbol, 5, MovingAverageType.WILDERS, Resolution.DAILY)
        self.atr_slow = algorithm.atr(symbol, 60, MovingAverageType.WILDERS, Resolution.DAILY)
        self.open_vol_history = deque(maxlen=RVOL_LOOKBACK)
        self.reset_day()

    def reset_day(self):
        self.or_open = None
        self.or_high = float("-inf")
        self.or_low = float("inf")
        self.or_close = None
        self.or_volume = 0.0
        self.rvol = None
        self.direction = 0
        self.r_dollars = None
        self.entry_ticket = None
        self.entry_price = None
        self.selected = False
        self.bound_by = None
        self.gap_atr = 0.0
        self.atr_spike = 1.0
        self.best = None
        self.worst = None
        self.checkpoints = {}          # minutes-past-open -> price

    def update_opening_range(self, bar):
        if self.or_open is None:
            self.or_open = bar.open
        self.or_high = max(self.or_high, bar.high)
        self.or_low = min(self.or_low, bar.low)
        self.or_close = bar.close
        self.or_volume += bar.volume

    def update_excursion(self, bar):
        if self.entry_price is None:
            return
        self.best = bar.high if self.best is None else max(self.best, bar.high)
        self.worst = bar.low if self.worst is None else min(self.worst, bar.low)

    @property
    def indicators_ready(self):
        return (self.atr.is_ready and self.avg_volume.is_ready
                and self.prev_close.is_ready and self.atr_fast.is_ready
                and self.atr_slow.is_ready
                and len(self.open_vol_history) == RVOL_LOOKBACK)


class OpeningRangeLiquidityProvision(QCAlgorithm):

    def initialize(self):
        (sy, sm, sd_), (ey, em, ed) = _WINDOWS[MODE]
        self.set_start_date(sy, sm, sd_)
        self.set_end_date(ey, em, ed)
        self.set_cash(STARTING_CASH)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE,
                                 AccountType.MARGIN)
        self.set_security_initializer(
            BrokerageModelSecurityInitializer(
                self.brokerage_model, FuncSecuritySeeder(self.get_last_known_prices)))

        self.universe_settings.resolution = Resolution.MINUTE
        self.universe_settings.extended_market_hours = False
        self.add_universe(self._select_coarse)

        self.spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.set_benchmark(self.spy)

        self.symbol_data = {}
        self.trade_log = []
        self.open_positions = {}
        self._ranked_today = False
        self._orders_placed = 0
        self._filtered_out = 0
        self._gapped_out = 0
        self._pool_days = 0
        self._pool_total = 0
        self._orders_filled = 0
        self._stranded_days = 0
        self._stranded_names = 0
        self._day_start_equity = None
        self._session_date = None
        self._open_time = None
        self._range_end = None
        self._worst_day = 0.0
        self._worst_day_date = None

        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.before_market_close(self.spy, CLOSE_OUT_MINUTES),
                         self._on_close_out)

        self.settings.free_portfolio_value_percentage = 0.05
        self.set_warm_up(timedelta(days=140), WARMUP_RESOLUTION)  # ATR(60) needs it

        self.log("=" * 72)
        self.log(f"ORB v8 -- PRE-REGISTERED OUT-OF-SAMPLE RUN   MODE={MODE}  "
                 f"FRICTION={FRICTION_PRESET}  u{UNIVERSE_SIZE}  N={N_HOLDINGS}")
        self.log(f"  RANK_BY={RANK_BY}  REL_R_MIN={REL_R_MIN}  MAX_GAP_ATR={MAX_GAP_ATR}")
        self.log("  PREDICTION net R +0.09 [-0.02,+0.20] | n 2600 | CAGR +8% | MDD 62%")
        self.log("  DECISION   <=0 CLOSES | 0..+0.05 not tradeable | >+0.05 replicated")
        self.log("  VOID IF    stranded>0 | fill<70% | lev-bound<97% | trades/day<1.0")
        if MODE != "OOS" or FRICTION_PRESET != "measured" or REL_R_MIN != 0.008 \
           or RANK_BY != "rvol" or N_HOLDINGS != 5 or UNIVERSE_SIZE != 500 \
           or ENTRY_MODE != "stop":
            self.log("  *** CONFIG DOES NOT MATCH THE PRE-REGISTRATION. This is no")
            self.log("  *** longer the committed test. Say so before quoting anything.")
        self.log(f"  ENTRY_MODE={ENTRY_MODE}  offset={LIMIT_OFFSET_ATR}xATR  "
                 f"maker_capture={MAKER_ADVERSE_CAPTURE}  exit=MOC only")
        if ENTRY_MODE == "limit":
            self.log("  fading the break: GREEN -> SELL LIMIT at the high, "
                     "RED -> BUY LIMIT at the low")
        else:
            self.log("  taking the break: GREEN -> BUY STOP at the high, "
                     "RED -> SELL STOP at the low")
        self.log("=" * 72)

    def _select_coarse(self, coarse):
        filtered = [c for c in coarse
                    if c.has_fundamental_data and c.price > MIN_PRICE and c.dollar_volume > 0]
        filtered.sort(key=lambda c: c.dollar_volume, reverse=True)
        return [c.symbol for c in filtered[:UNIVERSE_SIZE]]

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            symbol = security.symbol
            security.set_margin_model(PatternDayTradingMarginModel())
            security.set_fee_model(PerShareFeeModel())
            security.set_slippage_model(SLIPPAGE_MODEL)
            if symbol != self.spy and symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(self, symbol)
        for security in changes.removed_securities:
            sd = self.symbol_data.pop(security.symbol, None)
            if sd is not None:
                _RISK.pop(sd.key, None)

    def on_data(self, data):
        if self.is_warming_up:
            return
        t = self.time
        if self._session_date != t.date():
            self._session_date = t.date()
            self._open_time = t.replace(hour=9, minute=30, second=0, microsecond=0)
            self._range_end = self._open_time + timedelta(minutes=OR_MINUTES)
        open_time, range_end = self._open_time, self._range_end

        if open_time < t <= range_end:
            for symbol, sd in self.symbol_data.items():
                bar = data.bars.get(symbol)
                if bar is not None:
                    sd.update_opening_range(bar)
            if t == range_end and not self._ranked_today:
                self._ranked_today = True
                self._day_start_equity = float(self.portfolio.total_portfolio_value)
                self._rank_and_place()
            return

        if t <= open_time or self._day_start_equity is None or not self.open_positions:
            return
        mins = int((t - open_time).total_seconds() // 60)
        cp = mins if mins in _CHECKPOINTS else None
        for symbol, sd in list(self.open_positions.items()):
            bar = data.bars.get(symbol)
            if bar is not None:
                sd.update_excursion(bar)
                if cp is not None and cp not in sd.checkpoints:
                    sd.checkpoints[cp] = float(bar.close)

    # -----------------------------------------------------------------------

    def _rank_and_place(self):
        self.transactions.cancel_open_orders()

        candidates = []
        for symbol, sd in self.symbol_data.items():
            if sd.or_open is None or sd.or_volume <= 0:
                continue
            was_ready = sd.indicators_ready

            viable, price, atr, adv = False, 0.0, 0.0, 0.0
            if was_ready:
                price = float(self.securities[symbol].price)
                atr = float(sd.atr.current.value)
                if price >= MIN_PRICE and atr > 0 and atr >= ATR_FLOOR:
                    adv = float(sd.avg_volume.current.value)
                    viable = adv >= MIN_AVG_VOLUME

            if viable:
                trailing = np.asarray(sd.open_vol_history, dtype=float)
                lo, hi = _pct_5_95(sorted(sd.open_vol_history))
                avg = float(np.clip(trailing, lo, hi).mean())
                sd.rvol = (sd.or_volume / avg) if avg > 0 else None

            sd.open_vol_history.append(sd.or_volume)      # UNCONDITIONAL

            if not viable or sd.rvol is None or sd.rvol < MIN_RVOL:
                continue
            _RISK[sd.key] = {"atr": atr, "adv": adv}
            if sd.or_close == sd.or_open:
                continue
            sd.r_dollars = R_ACCOUNTING_ATR_PCT * atr

            # RELATIVE-RANGE FILTER. The $3 ATR_FLOOR is an ABSOLUTE test, which
            # at u500 selects expensive, quiet names -- exactly the 84% of the
            # book that contributes ~8% of the P&L. This is the relative version.
            if REL_R_MIN > 0 and (sd.r_dollars / price) < REL_R_MIN:
                self._filtered_out += 1
                continue

            pc = float(sd.prev_close.current.value)
            sd.gap_atr = abs(sd.or_open - pc) / atr if pc > 0 else 0.0
            slow = float(sd.atr_slow.current.value)
            sd.atr_spike = (float(sd.atr_fast.current.value) / slow) if slow > 0 else 1.0
            if sd.gap_atr > MAX_GAP_ATR:
                self._gapped_out += 1
                continue

            # sort() below is reverse=True, so the key must be LARGER for
            # better. Spike is ascending-is-better, hence the negation.
            key = (-sd.atr_spike) if RANK_BY == "spike" else sd.rvol
            candidates.append((key, symbol))

        self._pool_days += 1
        self._pool_total += len(candidates)
        if not candidates:
            return
        candidates.sort(reverse=True, key=lambda x: x[0])
        selected = candidates[:N_HOLDINGS]
        equity = float(self.portfolio.total_portfolio_value)

        for rvol, symbol in selected:
            sd = self.symbol_data[symbol]
            price = float(self.securities[symbol].price)
            atr = float(sd.atr.current.value)
            if price <= 0 or atr <= 0:
                continue

            # The break level and which side of it we want.
            break_is_high = sd.or_close > sd.or_open
            offset = LIMIT_OFFSET_ATR * atr

            if ENTRY_MODE == "stop":
                # TAKE: buy the up-break / sell the down-break.
                level = sd.or_high if break_is_high else sd.or_low
                sd.direction = 1 if break_is_high else -1
            else:
                # MAKE: sell into the up-break / buy into the down-break.
                # Resting BEYOND the extreme means price must trade through with
                # force before we are filled, which is the defence against
                # LEAN's touch-fills.
                level = (sd.or_high + offset) if break_is_high else (sd.or_low - offset)
                sd.direction = -1 if break_is_high else 1

            if sd.direction < 0 and not ALLOW_SHORTS:
                continue

            qty_by_vol      = (VOL_BUDGET_PCT * equity) / atr
            qty_by_leverage = (MARGIN_BUFFER * MAX_LEVERAGE * equity / N_HOLDINGS) / price
            sd.bound_by = "vol" if qty_by_vol < qty_by_leverage else "leverage"
            qty = int(min(qty_by_vol, qty_by_leverage))
            if qty < 1:
                continue
            signed_qty = qty if sd.direction > 0 else -qty

            # Arm state BEFORE submitting. Both order types REST here, so the
            # fill arrives asynchronously in on_order_event -- but v1.0 was
            # killed by setting these flags after submission, so the habit
            # stands regardless of order type.
            sd.selected = True
            self._orders_placed += 1
            if ENTRY_MODE == "stop":
                sd.entry_ticket = self.stop_market_order(
                    symbol, signed_qty, level, tag=f"TAKE rvol={rvol:.2f}")
            else:
                sd.entry_ticket = self.limit_order(
                    symbol, signed_qty, level, tag=f"MAKE rvol={rvol:.2f}")

    # -----------------------------------------------------------------------

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            return
        sd = self.symbol_data.get(order_event.symbol)
        if sd is None or not sd.selected or sd.entry_ticket is None:
            return
        if order_event.order_id != sd.entry_ticket.order_id:
            return
        sd.entry_price = float(order_event.fill_price)
        sd.best = sd.entry_price
        sd.worst = sd.entry_price
        self._orders_filled += 1
        self.open_positions[order_event.symbol] = sd

    def _on_close_out(self):
        self._ranked_today = False
        if self.is_warming_up:
            return
        self.transactions.cancel_open_orders()

        for symbol, sd in list(self.open_positions.items()):
            if self.portfolio[symbol].invested:
                price = float(self.securities[symbol].price)
                self.liquidate(symbol, tag="MOC")
                self._record_trade(sd, price, "MOC")
        self.open_positions.clear()

        if self.portfolio.invested:
            stranded = [s for s in self.symbol_data.keys() if self.portfolio[s].invested]
            self._stranded_days += 1
            self._stranded_names += max(1, len(stranded))
            self.liquidate(tag="STRANDED")
            if self._stranded_days <= 5:
                self.log(f"!! STRANDED {self.time.date()}: "
                         f"{[str(s) for s in stranded] or 'outside symbol_data'} -- swept. "
                         f"Non-zero at the end => VOID.")

        if self._day_start_equity:
            day = float(self.portfolio.total_portfolio_value) / self._day_start_equity - 1.0
            if day < self._worst_day:
                self._worst_day = day
                self._worst_day_date = self.time.date()

        for sd in self.symbol_data.values():
            sd.reset_day()
        self._day_start_equity = None

    def _record_trade(self, sd, exit_price, exit_reason):
        if sd.entry_price is None or not sd.r_dollars or sd.r_dollars <= 0:
            return
        gross_r = ((exit_price - sd.entry_price) * sd.direction) / sd.r_dollars
        comm_r = (2 * COMMISSION_PER_SHARE) / sd.r_dollars
        borrow_r = 0.0
        if sd.direction < 0:
            price = float(self.securities[sd.symbol].price)
            borrow_r = (price * BORROW_ANNUAL_PCT / 252.0) / sd.r_dollars

        if sd.direction > 0:
            mfe_r = ((sd.best or sd.entry_price) - sd.entry_price) / sd.r_dollars
            mae_r = ((sd.worst or sd.entry_price) - sd.entry_price) / sd.r_dollars
        else:
            mfe_r = (sd.entry_price - (sd.worst or sd.entry_price)) / sd.r_dollars
            mae_r = (sd.entry_price - (sd.best or sd.entry_price)) / sd.r_dollars

        self.trade_log.append({
            "symbol": sd.key, "atr": float(sd.atr.current.value),
            "year": int(self.time.year),
            "rvol": sd.rvol * 100.0, "reason": exit_reason, "bound_by": sd.bound_by,
            "gross_r": gross_r, "comm_r": comm_r,
            "net_r": gross_r - comm_r - borrow_r, "borrow_r": borrow_r,
            "direction": sd.direction, "entry_price": sd.entry_price,
            "r_dollars": sd.r_dollars, "mae_r": mae_r, "mfe_r": mfe_r,
            "gap_atr": sd.gap_atr, "atr_spike": sd.atr_spike,
            "cp": {k: ((v - sd.entry_price) * sd.direction) / sd.r_dollars
                   for k, v in sd.checkpoints.items()},
        })

    # -----------------------------------------------------------------------

    def on_end_of_algorithm(self):
        self.log("=" * 72)
        self.log("INTEGRITY CHECK -- read before anything else")
        ok = self._stranded_days == 0
        if ok:
            self.log("  positions stranded at close-out: 0   OK")
        else:
            self.log(f"  *** VOID: {self._stranded_days} days ended with lost positions "
                     f"({self._stranded_names} names).")
        fill_rate = self._orders_filled / max(1, self._orders_placed)
        pool = self._pool_total / max(1, self._pool_days)
        self.log(f"  MAX_GAP_ATR={MAX_GAP_ATR} -> {self._gapped_out} gap rejections "
                 f"(LOOSE on purpose: read the gap table, then set it)")
        self.log(f"  RANK_BY={RANK_BY}  pool {pool:.1f} names/day vs N={N_HOLDINGS}  ->  "
                 f"ranking {'BINDS (selects top %.0f%%)' % (100*N_HOLDINGS/pool) if pool > N_HOLDINGS else 'DOES NOT BIND -- we take everything, this run tests NOTHING'}")
        self.log(f"  REL_R_MIN={REL_R_MIN}  ->  qualifying pool {pool:.1f} names/day "
                 f"(need >= {N_HOLDINGS} to fill the book; {self._filtered_out} rejections)")
        self.log(f"  BREADTH FLOOR: below ~3 trades/day a sub-50% win rate stops")
        self.log(f"  obeying the law of large numbers. Check trades/day before Sharpe.")
        self.log(f"  entry orders placed / filled: {self._orders_placed} / "
                 f"{self._orders_filled}   FILL RATE {fill_rate:.1%}")
        self.log("  (stop and limit modes should have SIMILAR fill rates -- same level,")
        self.log("   same bars. A large gap means the two fill models disagree.)")
        self.log("  VERDICT: %s" % ("clean" if ok else "*** RUN IS VOID ***"))
        self.log("=" * 72)
        self.log(f"FRICTION AUDIT   preset={FRICTION_PRESET}  mode={ENTRY_MODE}  "
                 f"maker_capture={MAKER_ADVERSE_CAPTURE}")
        if _SLIP_LOG:
            p = np.array([s[0] for s in _SLIP_LOG]); tot = np.array([s[1] for s in _SLIP_LOG])
            kinds = np.array([s[5] for s in _SLIP_LOG])
            self.log(f"  fills charged {len(_SLIP_LOG)}   mean price ${p.mean():.2f}")
            self.log(f"  mean {100*tot.mean():+.2f} c/side = {(tot/p*1e4).mean():+.2f} bp/side "
                     f"(negative = the maker is being PAID)")
            for k in ("make", "take", "close", "flat"):
                m = kinds == k
                if m.sum():
                    self.log(f"    {k:<7} n={int(m.sum()):>6}  {100*tot[m].mean():+7.2f} c/side  "
                             f"{(tot[m]/p[m]*1e4).mean():+6.2f} bp/side")
        else:
            self.log("  NO SLIPPAGE RECORDED -- model not wired in. Everything below is invalid.")
        self.log("=" * 72)

        if not self.trade_log:
            self.log("No trades. Check warm-up and filters.")
            return

        n = len(self.trade_log)
        net = np.array([t["net_r"] for t in self.trade_log])
        gross = np.array([t["gross_r"] for t in self.trade_log])
        mae = np.array([t["mae_r"] for t in self.trade_log])
        mfe = np.array([t["mfe_r"] for t in self.trade_log])
        rdol = np.array([t["r_dollars"] for t in self.trade_log])
        epx = np.array([t["entry_price"] for t in self.trade_log])
        dirs = np.array([t["direction"] for t in self.trade_log])
        lev = sum(1 for t in self.trade_log if t["bound_by"] == "leverage")

        tpd = n / max(1, self._pool_days)
        self.log(f"Trades {n}   ENTRY_MODE={ENTRY_MODE}   {tpd:.2f} trades/day  "
                 f"{'OK' if tpd >= 1.0 else '<<< BELOW BREADTH FLOOR: RUN IS VOID'}")
        # ---- LABEL FIX. "gross_r" is computed from FILL prices, so it is
        # already net of the entire slippage model. Calling it gross has
        # understated friction by ~10x in every log this project has produced.
        self.log(f"Win rate {(net>0).mean():.1%}   "
                 f"after-slippage {gross.mean():+.4f}R   NET {net.mean():+.4f}R")

        # ---- TRUE FRICTION DECOMPOSITION ----------------------------------
        # Rebuild the charge each fill actually paid, in R units.
        #   entry (take): max(MIN_HALF_SPREAD, half_bps*px) + adv_frac*ATR
        #   exit (close): max(MIN_HALF_SPREAD, half_bps*px)
        #   ATR = r_dollars / R_ACCOUNTING_ATR_PCT, same ATR the model used.
        if _F["FLAT_CENTS"] > 0:
            half_in_r = (2.0 * _F["FLAT_CENTS"]) / rdol
            adv_in_r = np.zeros(n)
        else:
            half_e = np.maximum(MIN_HALF_SPREAD, _F["HALF_SPREAD_BPS"] / 1e4 * epx)
            half_in_r = 2.0 * half_e / rdol
            adv_in_r = np.full(n, _F["ADVERSE_ATR_FRAC"] / R_ACCOUNTING_ATR_PCT)
        comm_in_r = np.array([t["comm_r"] + t["borrow_r"] for t in self.trade_log])
        slip_in_r = half_in_r + adv_in_r
        true_gross = net + slip_in_r + comm_in_r
        self.log("  ---- decomposition, per trade, in R ----")
        self.log(f"    TRUE GROSS  (decision px to decision px) {true_gross.mean():+.4f}")
        self.log(f"    spread      (both sides)                 {-half_in_r.mean():+.4f}")
        self.log(f"    ADVERSE SELECTION (entry only)           {-adv_in_r.mean():+.4f}"
                 f"   <- {100*adv_in_r.mean()/max(1e-9,true_gross.mean()):.0f}% of gross")
        self.log(f"    commission + borrow                      {-comm_in_r.mean():+.4f}")
        self.log(f"    NET                                      {net.mean():+.4f}")
        # SELF-CHECK: the reconstruction must agree with what the slippage
        # model actually logged. If these disagree the decomposition is wrong.
        if _SLIP_LOG:
            meas_c = float(np.array([s[1] for s in _SLIP_LOG]).mean()) * 2.0
            recon_c = float((slip_in_r * rdol).mean())
            self.log(f"  SELF-CHECK  slip/round-trip: model logged ${meas_c:.4f}, "
                     f"reconstruction ${recon_c:.4f}  -> "
                     f"{'AGREE' if abs(meas_c-recon_c) < 0.02*max(meas_c,1e-9)+0.002 else '*** DISAGREE: decomposition invalid ***'}")
        self.log(f"Edge on notional {(net*rdol/epx).mean()*1e4:+.2f} bp")
        self.log(f"Leverage-bound {lev/n:.1%}  "
                 f"{'OK' if lev/n >= 0.9699 else '<< BELOW 97%: CAGR NOT COMPARABLE'}")
        # We have never measured what gross the book ACTUALLY runs at. Target is
        # MARGIN_BUFFER*MAX_LEVERAGE = 3.60x; unfilled entries mean we run under it.
        by_day = {}
        for t in self.trade_log:
            by_day[t["symbol"], t["reason"]] = 0
        fill_rate = self._orders_filled / max(1, self._orders_placed)
        target_gross = MARGIN_BUFFER * MAX_LEVERAGE
        self.log(f"Gross exposure: target {target_gross:.2f}x, "
                 f"fill rate {fill_rate:.1%} -> realised ~{target_gross*fill_rate:.2f}x "
                 f"({100*fill_rate:.0f}% of the leverage budget)")
        # ---- FOUR SWEEPS FROM ONE RUN --------------------------------
        # Every trade runs to the close, so its full path statistics are
        # observed. These replay alternative EXIT rules at zero extra runtime.
        #
        # VALIDITY, stated precisely:
        #   a STOP-only sweep is valid   -- MAE <= -S implies price touched -S
        #   a TARGET-only sweep is valid -- MFE >= +T implies price touched +T
        #   COMBINING them is NOT valid  -- bar data cannot say which came first
        #   a TIME sweep is valid        -- the checkpoint price is observed
        # Use these to choose a rule to TEST. Never quote one as a result.
        # (The stop sweep earned some credit: it said "no stop" and a real run
        #  then confirmed it. That is one point of evidence, not a licence.)
        stop_cost_r = ((np.maximum(MIN_HALF_SPREAD, _F["HALF_SPREAD_BPS"] / 1e4 * epx)
                        + _F["ADVERSE_ATR_FRAC"] * (rdol / R_ACCOUNTING_ATR_PCT)) / rdol
                       if _F["FLAT_CENTS"] <= 0 else (2 * _F["FLAT_CENTS"]) / rdol)
        base = net.mean()

        self.log("-" * 72)
        self.log("SWEEP 1of4 -- STOP WIDTH (exit on adverse excursion)")
        self.log(f"  {'stop':<16}{'% hit':>8}{'net R':>10}{'vs none':>10}")
        for S in _STOP_SWEEP:
            hit = mae <= -S
            sim = np.where(hit, -S - stop_cost_r, net)
            self.log(f"  {f'{S:g}R ({S*R_ACCOUNTING_ATR_PCT:.2f}xATR)':<16}"
                     f"{100*hit.mean():>7.1f}%{sim.mean():>+10.4f}{sim.mean()-base:>+10.4f}")

        self.log("-" * 72)
        self.log("SWEEP 2of4 -- PROFIT TARGET (the give-back nobody has examined)")
        self.log(f"  mean MFE {mfe.mean():+.3f}R vs mean NET {base:+.4f}R -- "
                 f"the average trade gives back {mfe.mean()-base:.2f}R from its peak")
        self.log(f"  {'target':<16}{'% hit':>8}{'net R':>10}{'vs none':>10}")
        for T in _TARGET_SWEEP:
            hit = mfe >= T
            sim = np.where(hit, T - stop_cost_r, net)
            self.log(f"  {f'+{T:g}R ({T*R_ACCOUNTING_ATR_PCT:.2f}xATR)':<16}"
                     f"{100*hit.mean():>7.1f}%{sim.mean():>+10.4f}{sim.mean()-base:>+10.4f}")

        self.log("-" * 72)
        self.log("SWEEP 3of4 -- EXIT TIME (does the edge decay through the day?)")
        self.log(f"  {'exit at':<16}{'n obs':>8}{'net R':>10}{'vs close':>10}")
        for cp in _CHECKPOINTS:
            vals = [t["cp"].get(cp) for t in self.trade_log]
            have = np.array([v is not None for v in vals])
            if have.sum() < 100:
                continue
            arr = np.array([v if v is not None else 0.0 for v in vals])
            # charge one taker crossing for the early market exit
            sim = np.where(have, arr - stop_cost_r / 2.0, net)
            hh, mm = divmod(30 + cp, 60)
            self.log(f"  {f'{9+hh:02d}:{mm:02d}':<16}{int(have.sum()):>8}"
                     f"{sim.mean():>+10.4f}{sim.mean()-base:>+10.4f}")
        self.log(f"  {'15:48 (MOC)':<16}{n:>8}{base:>+10.4f}{0.0:>+10.4f}")

        # ---- SWEEP 4: THE ONE THAT MATTERS -------------------------------
        # ADVERSE_ATR_FRAC is a number I invented and never measured, and it
        # is worth more than the entire edge. Because R := 0.20 x ATR and the
        # adverse charge is adv_frac x ATR, the charge in R is exactly
        # adv_frac / 0.20 for EVERY trade -- scale-free, name-independent.
        # So net R at any other adv_frac is exact, not simulated:
        #     net(a') = net(a) + (a - a') / R_ACCOUNTING_ATR_PCT
        # This is the only sweep in this file that is a RESULT rather than a
        # rule-selection device, because it re-prices the same trades under a
        # different assumption instead of inventing counterfactual exits.
        self.log("-" * 72)
        self.log("SWEEP 4of4 -- ADVERSE SELECTION SENSITIVITY  (exact re-pricing)")
        if _F["FLAT_CENTS"] > 0:
            self.log("  preset uses FLAT_CENTS; no adverse term to sweep.")
        else:
            a0 = _F["ADVERSE_ATR_FRAC"]
            self.log(f"  current assumption {a0:.4f} x ATR = "
                     f"{a0/R_ACCOUNTING_ATR_PCT:.3f}R per entry, every trade")
            self.log(f"  {'adv_frac':<12}{'R/entry':>10}{'net R':>10}   note")
            for a in (0.0, 0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05, 0.075, 0.10):
                shifted = base + (a0 - a) / R_ACCOUNTING_ATR_PCT
                note = "<- AS RUN" if abs(a - a0) < 1e-12 else ("<- dead" if shifted <= 0 else "")
                self.log(f"  {a:<12.5f}{a/R_ACCOUNTING_ATR_PCT:>10.3f}"
                         f"{shifted:>+10.4f}   {note}")
            breakeven = a0 + R_ACCOUNTING_ATR_PCT * base
            self.log(f"  BREAKEVEN adverse selection = {breakeven:.4f} x ATR "
                     f"({breakeven/R_ACCOUNTING_ATR_PCT:.3f}R per entry)")
            self.log(f"  reference: 'measured' preset = 0.0250, "
                     f"'conservative' = 0.0500, 'paper' = 0.0")
            if breakeven <= 0:
                self.log("  => NEGATIVE BREAKEVEN: no adverse-selection assumption saves it.")
            elif breakeven < 0.035:
                self.log("  => breakeven sits BELOW 0.035: the result lives inside the "
                         "uncertainty of an unmeasured parameter. Per the pre-registration "
                         "this is NOT tradeable; measure real adverse selection next.")
            else:
                self.log("  => breakeven above 0.035: the edge survives a materially "
                         "harsher fill assumption than the one used here.")

        self.log("-" * 72)
        self.log("MIRROR CHECK -- run the other ENTRY_MODE and compare:")
        self.log(f"   net_R(limit) should equal -net_R(stop) + the spread not crossed.")
        self.log(f"   this run: NET {net.mean():+.4f}R  gross {gross.mean():+.4f}R")
        self.log("   If the two modes are not near-mirrors, a fill model is wrong")
        self.log("   and nothing below should be quoted.")
        self.log("-" * 72)
        self.log("THE TAIL")
        order = np.argsort(net)
        self.log(f"  worst trade {net[order[0]]:+.2f}R ({self.trade_log[order[0]]['symbol']})   "
                 f"worst 5: " + " ".join(f"{net[i]:+.1f}" for i in order[:5]))
        self.log(f"  1st pct {np.percentile(net,1):+.2f}R   5th {np.percentile(net,5):+.2f}R")
        self.log(f"  worst DAY {100*self._worst_day:+.2f}% on {self._worst_day_date}")
        self.log(f"  mean MAE {mae.mean():+.3f}R   mean MFE {mfe.mean():+.3f}R")
        self.log("-" * 72)
        self.log("NET R BY MAE BUCKET")
        for lo, hi in ((-1e9, -6), (-6, -4), (-4, -3), (-3, -2), (-2, -1), (-1, 0)):
            m = (mae >= lo) & (mae < hi)
            if m.sum():
                self.log(f"  MAE {lo:>6g}..{hi:<3g}R  n={int(m.sum()):>6} ({m.mean():>5.1%})  "
                         f"net {net[m].mean():+.4f}R")
        self.log("-" * 72)
        for sgn, lab in ((1, "long "), (-1, "short")):
            m = dirs == sgn
            if m.sum():
                self.log(f"  {lab} n={int(m.sum()):>6} ({m.mean():>5.1%})  "
                         f"net {net[m].mean():+.4f}R  total {net[m].sum():+.0f}R")

        # ---- PER-YEAR TABLE: the pre-registered predictions (a) (b) (c) ---
        # (a) 2020 net R > full-period net R
        # (b) 2022 shorts net R > 2022 longs net R  (first time ever)
        # (c) 2023+ breadth below 1.0 trades/day
        yrs = np.array([t["year"] for t in self.trade_log])
        self.log("-" * 72)
        self.log("NET R BY CALENDAR YEAR  <<< the pre-registered predictions live here")
        self.log(f"{'year':<8}{'n':>7}{'/day':>7}{'net R':>10}{'long':>10}{'short':>10}")
        for y in sorted(set(int(v) for v in yrs)):
            m = yrs == y
            if m.sum() < 20:
                continue
            ml = m & (dirs == 1); ms = m & (dirs == -1)
            lr = f"{net[ml].mean():+.4f}" if ml.sum() >= 10 else "    n/a"
            sr = f"{net[ms].mean():+.4f}" if ms.sum() >= 10 else "    n/a"
            self.log(f"{y:<8}{int(m.sum()):>7}{m.sum()/252.0:>7.2f}"
                     f"{net[m].mean():>+10.4f}{lr:>10}{sr:>10}")
        self.log(f"{'ALL':<8}{n:>7}{tpd:>7.2f}{base:>+10.4f}")

        def bucket(title, vals, edges, fmt):
            self.log("-" * 72); self.log(title)
            self.log(f"{'bucket':<14}{'n':>7}{'net R':>10}{'edge bp':>10}")
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (vals >= lo) & (vals < hi)
                if m.sum() < 20:
                    continue
                self.log(f"{fmt(lo,hi):<14}{int(m.sum()):>7}{net[m].mean():>+10.4f}"
                         f"{(net[m]*rdol[m]/epx[m]).mean()*1e4:>+10.2f}")

        # ---- THE TWO TABLES THIS RUN EXISTS FOR ---------------------------
        bucket("NET R BY OVERNIGHT GAP  (|open-prev_close| / ATR)  <<< READ FIRST",
               np.array([t["gap_atr"] for t in self.trade_log]), list(_GAP_EDGES),
               lambda l, h: f"{l:g}-{h:g}x" if h < 1e9 else f">{l:g}x")
        bucket("NET R BY ATR SPIKE  (ATR5 / ATR60 -- event vol vs flow vol)",
               np.array([t["atr_spike"] for t in self.trade_log]), list(_SPIKE_EDGES),
               lambda l, h: f"{l:g}-{h:g}" if h < 1e9 else f">{l:g}")
        bucket("NET R BY RVOL", np.array([t["rvol"] for t in self.trade_log]),
               [100, 500, 1000, 2000, 5000, 1e9],
               lambda l, h: f"{l:g}-{h:g}%" if h < 1e9 else f">{l:g}%")
        bucket("NET R BY R/PRICE", rdol/epx, [0, .004, .006, .008, .012, .02, 1],
               lambda l, h: f"{l*100:.1f}-{h*100:.1f}%")

        self.log("-" * 72)
        self.log("CONCENTRATION (>15% of total = lucky draw, not factor)")
        by_symbol = {}
        for t in self.trade_log:
            by_symbol[t["symbol"]] = by_symbol.get(t["symbol"], 0.0) + t["net_r"]
        total = sum(abs(v) for v in by_symbol.values()) or 1.0
        for sym, r in sorted(by_symbol.items(), key=lambda kv: -abs(kv[1]))[:10]:
            self.log(f"  {sym:<12}{r:>+9.1f}R{abs(r)/total:>8.1%}")
        self.log("=" * 72)
