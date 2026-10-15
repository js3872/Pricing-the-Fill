# region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
import math
# endregion

"""
5-minute Opening Range Breakout on Stocks in Play
Zarattini, Barbon & Aziz (2024), SFI Research Paper 24-98.

REVISION 6 -- 2026-09-14. Two changes only, both to the COST AND SIZING MODEL.
The signal, the universe, the filters and the entry/exit logic are byte-identical
to rev 5. Nothing about what we trade has changed; only what we assume it costs
and how much of it we buy.

  (1) SIZING FIX. Rev 5 sized by `equity / len(selected)`, so on a day with one
      qualifying name the entire 3.6x gross landed in that one stock. Forensics
      on `Swimming Fluorescent Yellow Dinosaur` found 285 trades carrying >1.0x
      of the whole account in a single name, 16 over 2x, max 3.50x. Now divides
      by N_HOLDINGS. On the IS replay this improved CAGR, Sharpe AND drawdown
      simultaneously (71.6% / 1.55 / -26.6% vs 66.2% / 1.39 / -33.7%), which is
      the signature of removing uncompensated variance rather than adding edge.

  (2) REALISTIC FILL MODEL. Rev 5 charged a flat 2c/side. Forensics showed that
      85.7% of entries and 83.6% of protective stops filled at EXACTLY the
      trigger price plus that 2c -- LEAN models minute-bar gaps but cannot see
      sub-minute adverse selection, which is the entire life of a stop resting
      at the opening-range extreme of the day's highest-RVOL name. Replaced with
      a three-component model (half-spread + stop adverse-selection + square-root
      impact). See RealisticSlippageModel.

---------------------------------------------------------------------------
PROTOCOL NOTE -- why re-running the IS window is legitimate here

This is a COST MODEL change, not a strategy change. No parameter is being
selected, no threshold tuned, no window mined. We are asking "what did the thing
we already tested actually cost to trade," which is a measurement question, not a
search. Re-running 2016-2019 burns nothing. The OOS window (2020 - Jun 2026)
remains SPENT and must not be re-run to "check" the friction result.

PRE-REGISTERED PREDICTION, corrected 2026-09-15 before any run:

  net R per trade  |  paper +0.1419 (given) -> measured  +0.00 to +0.02
                                            -> conservative  ~ -0.21
  CAGR             |  measured  0% to +12%      conservative  ~ -55%
  Sharpe           |  measured  0.10 to 0.45    conservative  ~ -2
  stop-out rate    |  RISES from 65.8% to ~67-69% under measured
  avg loss on stops|  -1.02R -> ~ -1.18R under measured   <-- cleanest tell
                                                              that the model bit

  ERRATUM. The first version of this block predicted net R -0.05 to -0.10 under
  "measured". That was WRONG and the error is instructive. It charged entry
  slippage to every trade. But the protective stop is placed at the ENTRY FILL
  +/- r_dollars, so on a stopped-out trade a worse entry drags the stop down with
  it and the entry slippage cancels out of the R-multiple entirely:

      long, stop-out:  pnl/share = (entry+se-R-sx) - (entry+se) = -(R + sx)

  Only the EXIT slippage bites on the 66% that stop out. Both bite on the 34%
  that reach the close. Blending a symmetric round-trip cost across all trades
  double-charges two thirds of the book. Corrected on the trade ledger
  (R recovered per-trade from 2,298 filled and 1,195 cancelled protective
  stops), "measured" lands on roughly ZERO, not clearly negative.

  SECOND-ORDER EFFECT NOT MODELLED, direction known: a worse entry fill places
  the protective stop CLOSER to the trigger price, so more trades should stop
  out. That makes the prediction above mildly OPTIMISTIC. If the stop-out rate
  does not rise at all, something is wrong with how the model is wired.

  THE NUMBER THAT ACTUALLY MATTERS. All of the above is the IS window -- the
  fitted one. Measured OOS decay was net R +0.1419 -> +0.0696, i.e. -0.0723R.
  Applying that to a "measured" result of ~+0.005R gives an OOS-equivalent of
  about -0.067R. Best case here is "zero in-sample, negative out-of-sample."

  FALSIFIER: if "measured" prints net R above +0.05, read the FRICTION AUDIT
  block FIRST -- most likely the model is not being charged where we think.
  If the audit is clean and the number is still that good, ADVERSE_ATR_FRAC is
  too kind and must be re-derived from something other than LEAN's own visible
  gap fills. Do not celebrate a positive result here. Interrogate it.
---------------------------------------------------------------------------

RUN ORDER. Run the control FIRST or the rest is unreadable:
  1. FRICTION_PRESET="paper",  SIZING_FIX=False  -> MUST reproduce rev 5 exactly:
                                                    3,494 trades, Sharpe 1.629,
                                                    CAGR 66.10%, stop-out 65.8%.
                                                    If it does not, the refactor
                                                    changed something it should
                                                    not have. Stop and find it.
  2. FRICTION_PRESET="paper",  SIZING_FIX=True   -> isolates the sizing fix alone.
  3. FRICTION_PRESET="measured", SIZING_FIX=True -> THE ACTUAL QUESTION.
  4. FRICTION_PRESET="conservative", SIZING_FIX=True -> adds market impact.

Read the FRICTION AUDIT block at the end of the log before reading any
performance number. If mean charged slippage is not close to the preset's
intent, the model is not doing what we think and everything downstream is noise.
"""


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODE = "IS"                  # "IS" = 2016-2019, "OOS" = 2020 - Jun 2026.
                             # THE OOS WINDOW IS SPENT. Do not run it.

RANK_MODE = "top"            # "top" -> ranks 1-5 (the strategy)
                             # "control" -> ranks 21-25 (rank control)

DIRECTION_SIGN = 1           # 1 -> with the opening range. -1 -> direction control.

FILTER_MODE = "flat"         # "none" | "flat" (ATR_FLOOR) | "joint"

SIZING_FIX = True            # True  -> divide the leverage budget by N_HOLDINGS
                             # False -> rev 5 behaviour, divide by len(selected)
                             #          (keep False only for the control run)

FRICTION_PRESET = "measured" # "paper" | "measured" | "conservative"  -- see below

# --- Paper spec. Do not tune. ---
OR_MINUTES      = 5
RVOL_LOOKBACK   = 14
N_HOLDINGS      = 5
STOP_ATR_PCT    = 0.20       # R = 0.20 * daily ATR  (paper is 0.10; Stage 2 sweep
                             # settled on 0.20 on a broad plateau, not a spike)
RISK_PCT        = 0.01
MAX_LEVERAGE    = 4.0
MARGIN_BUFFER   = 0.90       # All N entry stops rest simultaneously and LEAN
                             # reserves margin for every resting order.
CLOSE_OUT_MINUTES = 12       # Must exceed LEAN's PatternDayTradingMarginModel
                             # 4x->2x switch ~10 min before the close.
MIN_PRICE       = 5.00
MIN_AVG_VOLUME  = 1_000_000
MIN_RVOL        = 1.00

# --- Ours, not the paper's. ---
STARTING_CASH   = 100_000
ATR_FLOOR       = 3.00
UNIVERSE_SIZE   = 500
ALLOW_SHORTS    = True

COMMISSION_PER_SHARE = 0.0035
BORROW_ANNUAL_PCT    = 0.03

_WINDOWS = {
    "IS":  ((2016, 1, 1), (2019, 12, 31)),
    "OOS": ((2020, 1, 1), (2026, 6, 30)),   # rev 5 had (2001,1,1)-(2010,8,16)
                                            # here while its own comment claimed
                                            # 2020-2026. Anyone who flipped MODE
                                            # got a plausible-looking number for
                                            # the wrong decade.
}


# ---------------------------------------------------------------------------
# FRICTION PRESETS
#
# HALF_SPREAD_BPS      half the quoted spread, in bp of price, floored at 1 tick.
#                      Liquid $150 names quote 1-3c midday; the first five
#                      minutes run several times wider. 1.5bp = 2.2c at $148.
#
# ADVERSE_ATR_FRAC     adverse selection charged ONLY on stop-triggered orders,
#                      as a fraction of daily ATR. CALIBRATED, not guessed: of
#                      the 3,494 entry fills in the rev-5 IS run, the 500 that
#                      gapped through the trigger inside one minute bar -- the
#                      only ones LEAN could see and charge -- cost 6.16 bp/side
#                      on average. At the run's mean price $163.87 and median
#                      ATR $3.98 that is 0.025 x ATR. So the default charges the
#                      other 85% of fills what LEAN already charged the visible
#                      15%. That is a floor on the true number, not a guess at it.
#
# IMPACT_Y             square-root law: impact = Y * sigma_daily * sqrt(Q/ADV).
#                      Y ~ 0.5 is the standard range. Off by default because it
#                      is the least certain component and should not be allowed
#                      to carry the verdict.
# ---------------------------------------------------------------------------

_PRESETS = {
    # Reproduces rev 5 exactly. Control run only.
    "paper":        dict(HALF_SPREAD_BPS=0.0, FLAT_CENTS=0.02,
                         ADVERSE_ATR_FRAC=0.0,   IMPACT_Y=0.0),
    # Calibrated to LEAN's own visible gap fills. The primary run.
    "measured":     dict(HALF_SPREAD_BPS=1.5, FLAT_CENTS=0.0,
                         ADVERSE_ATR_FRAC=0.025, IMPACT_Y=0.0),
    # Adds square-root market impact on top.
    "conservative": dict(HALF_SPREAD_BPS=1.5, FLAT_CENTS=0.0,
                         ADVERSE_ATR_FRAC=0.050, IMPACT_Y=0.5),
}
_F = _PRESETS[FRICTION_PRESET]

MIN_HALF_SPREAD = 0.005      # half a penny floor

# Figure 4, digitized. Used only by FILTER_MODE="joint".
_RVOL_EDGE = [(100, -0.020), (500, 0.080), (1000, 0.115), (1500, 0.165),
              (2000, 0.195), (2500, 0.265), (3000, 0.335), (1e9, 0.380)]


def expected_gross_r(rvol_pct):
    for threshold, edge in _RVOL_EDGE:
        if rvol_pct < threshold:
            return edge
    return _RVOL_EDGE[-1][1]


# Per-symbol risk cache, populated at ranking time, read by the slippage model.
# A module-level dict rather than an attribute lookup because the slippage model
# is handed only (asset, order) and must not depend on the algorithm object.
_RISK = {}

# Every slippage charge, for the end-of-run audit.
_SLIP_LOG = []


# ---------------------------------------------------------------------------
# COST MODELS
#
# Both are PLAIN classes. QC's Python layer duck-types custom models; it does
# not export a `SlippageModel` base at all, and subclassing `FeeModel` is
# fragile across API versions. Each method is aliased in both snake_case and
# PascalCase so this loads regardless of which convention LEAN dispatches on.
# (Rev 1 crashed on load for exactly this reason. Do not "clean it up".)
# ---------------------------------------------------------------------------

class PerShareFeeModel:
    """IBKR Pro tiered, the paper's assumption."""
    def get_order_fee(self, parameters):
        qty = abs(float(parameters.Order.Quantity))
        return OrderFee(CashAmount(qty * COMMISSION_PER_SHARE, "USD"))

    GetOrderFee = get_order_fee


class RealisticSlippageModel:
    """
    Three additive components, all returned as a positive price magnitude.
    LEAN applies the sign (worse for the filler, always).

        1. HALF-SPREAD       max(half a penny, HALF_SPREAD_BPS/1e4 * price)
                             Paid by every order. You cross to get done.

        2. ADVERSE SELECTION ADVERSE_ATR_FRAC * daily ATR, charged ONLY to
                             stop-triggered orders. This is the component rev 5
                             had no representation of at all. A stop resting at
                             the opening-range extreme of the day's highest-RVOL
                             name sits on the single most-trafficked price of
                             the session, and the breakouts worth having are the
                             ones with the least size behind them. You do not
                             get filled at that price because you were early;
                             you get filled because someone ran through it.

        3. IMPACT            IMPACT_Y * sigma_daily * sqrt(qty/ADV) * price,
                             the square-root law. Off unless "conservative".

    Order classification is by TAG, not by OrderType. We set every tag
    ourselves ("ENTRY ...", "STOP", "MOC", "NOSTOP"), so this is stable across
    every LEAN API revision. OrderType enum casing is not.
    """

    def __init__(self):
        self.half_bps  = float(_F["HALF_SPREAD_BPS"])
        self.flat      = float(_F["FLAT_CENTS"])
        self.adv_frac  = float(_F["ADVERSE_ATR_FRAC"])
        self.impact_y  = float(_F["IMPACT_Y"])

    def get_slippage_approximation(self, asset, order):
        try:
            price = float(asset.price)
        except Exception:
            price = 0.0
        if price <= 0:
            return 0.0

        # --- rev 5 compatibility path -------------------------------------
        if self.flat > 0:
            _SLIP_LOG.append((price, self.flat, self.flat, 0.0, 0.0, "flat"))
            return self.flat

        tag = ""
        try:
            tag = (order.tag or "")
        except Exception:
            try:
                tag = (order.Tag or "")
            except Exception:
                tag = ""
        is_stop_order = tag.startswith("ENTRY") or tag.startswith("STOP")

        risk = _RISK.get(str(order.symbol)) or {}
        atr  = float(risk.get("atr", 0.0))
        adv  = float(risk.get("adv", 0.0))

        # 1. half-spread
        half = max(MIN_HALF_SPREAD, self.half_bps / 1e4 * price)

        # 2. adverse selection, stop-triggered orders only
        adverse = 0.0
        if is_stop_order and self.adv_frac > 0.0:
            # Fall back to 2% of price if ATR is unknown -- roughly a typical
            # daily range for names that clear our filters.
            adverse = self.adv_frac * (atr if atr > 0 else 0.02 * price)

        # 3. square-root impact
        impact = 0.0
        if self.impact_y > 0.0 and adv > 0:
            try:
                qty = abs(float(order.quantity))
            except Exception:
                qty = abs(float(order.Quantity))
            sigma = (atr / price) if (atr > 0 and price > 0) else 0.02
            impact = self.impact_y * sigma * math.sqrt(qty / adv) * price

        total = half + adverse + impact
        _SLIP_LOG.append((price, total, half, adverse, impact,
                          "stop" if is_stop_order else "market"))
        return total

    GetSlippageApproximation = get_slippage_approximation


# Single shared instance so the audit log accumulates in one place.
SLIPPAGE_MODEL = RealisticSlippageModel()


def estimate_round_trip_cents(price, atr):
    """Forward-looking friction estimate for FILTER_MODE='joint'. Entry is always
    a stop; the exit is a stop ~2/3 of the time and a market order ~1/3."""
    if _F["FLAT_CENTS"] > 0:
        return 2 * _F["FLAT_CENTS"]
    half = max(MIN_HALF_SPREAD, _F["HALF_SPREAD_BPS"] / 1e4 * price)
    adverse = _F["ADVERSE_ATR_FRAC"] * atr
    return (half + adverse) + (half + 0.66 * adverse)


# ---------------------------------------------------------------------------
# PER-SYMBOL STATE
# ---------------------------------------------------------------------------

class SymbolData:
    def __init__(self, algorithm, symbol):
        self.symbol = symbol
        # Daily ATR consolidates at the close, so the value read at 09:35 today
        # reflects data through yesterday -- no lookahead.
        self.atr = algorithm.atr(symbol, 14, MovingAverageType.WILDERS, Resolution.DAILY)
        self.avg_volume = algorithm.sma(symbol, 14, Resolution.DAILY, Field.VOLUME)
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
        self.stop_ticket = None
        self.entry_price = None
        self.selected = False
        self.bound_by = None

    def update_opening_range(self, bar):
        if self.or_open is None:
            self.or_open = bar.open
        self.or_high = max(self.or_high, bar.high)
        self.or_low = min(self.or_low, bar.low)
        self.or_close = bar.close
        self.or_volume += bar.volume

    @property
    def indicators_ready(self):
        return (self.atr.is_ready
                and self.avg_volume.is_ready
                and len(self.open_vol_history) == RVOL_LOOKBACK)


# ---------------------------------------------------------------------------
# ALGORITHM
# ---------------------------------------------------------------------------

class OpeningRangeBreakoutRelativeVolume(QCAlgorithm):

    def initialize(self):
        (sy, sm, sd_), (ey, em, ed) = _WINDOWS[MODE]
        self.set_start_date(sy, sm, sd_)
        self.set_end_date(ey, em, ed)
        self.set_cash(STARTING_CASH)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE,
                                 AccountType.MARGIN)

        # Without a seeder, newly-added securities can be ordered before their
        # first bar exists -- which silently drops the highest-RVOL entrants,
        # i.e. precisely the strategy.
        self.set_security_initializer(
            BrokerageModelSecurityInitializer(
                self.brokerage_model,
                FuncSecuritySeeder(self.get_last_known_prices)))

        self.universe_settings.resolution = Resolution.MINUTE
        self.universe_settings.extended_market_hours = False
        self.add_universe(self._select_coarse)

        self.spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.set_benchmark(self.spy)

        self.symbol_data = {}
        self.trade_log = []
        self.open_positions = {}
        self._ranked_today = False
        self._unprotected = 0

        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.before_market_close(self.spy, CLOSE_OUT_MINUTES),
                         self._on_close_out)

        self.settings.free_portfolio_value_percentage = 0.05
        self.set_warm_up(timedelta(days=45))

        self.log("=" * 70)
        self.log(f"REV 6  MODE={MODE}  FRICTION={FRICTION_PRESET}  SIZING_FIX={SIZING_FIX}")
        self.log(f"  half-spread {_F['HALF_SPREAD_BPS']}bp | flat {_F['FLAT_CENTS']}c | "
                 f"adverse {_F['ADVERSE_ATR_FRAC']}xATR | impact Y={_F['IMPACT_Y']}")
        self.log("=" * 70)

    # -----------------------------------------------------------------------
    # UNIVERSE
    # -----------------------------------------------------------------------

    def _select_coarse(self, coarse):
        """Point-in-time by construction: QC's coarse universe includes delisted
        names on the dates they traded. VERIFIED on the rev-5 IS run -- LNKD,
        KITE, VRX, AGN, RHT, CELG, MON, GRUB, XLNX and TIF all appear in the
        trade log. Survivorship is genuinely handled, not assumed."""
        filtered = [c for c in coarse
                    if c.has_fundamental_data
                    and c.price > MIN_PRICE
                    and c.dollar_volume > 0]
        filtered.sort(key=lambda c: c.dollar_volume, reverse=True)
        return [c.symbol for c in filtered[:UNIVERSE_SIZE]]

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            symbol = security.symbol
            # LEAN defaults equities to Reg T 2x. The paper assumes 4x day-trading
            # buying power, available only to a PDT account and only intraday.
            # We flatten daily, so this is the correct model -- and without it
            # MAX_LEVERAGE=4.0 is unreachable and a quarter of the book silently
            # fails to open (rev 5 defect 8).
            security.set_margin_model(PatternDayTradingMarginModel())
            security.set_fee_model(PerShareFeeModel())
            security.set_slippage_model(SLIPPAGE_MODEL)
            if symbol != self.spy and symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(self, symbol)

        for security in changes.removed_securities:
            self.symbol_data.pop(security.symbol, None)
            _RISK.pop(str(security.symbol), None)

    # -----------------------------------------------------------------------
    # DATA -- opening range accumulation and trigger
    # -----------------------------------------------------------------------

    def on_data(self, data):
        if self.is_warming_up:
            return

        open_time = self.time.replace(hour=9, minute=30, second=0, microsecond=0)
        range_end = open_time + timedelta(minutes=OR_MINUTES)

        # Minute bars timestamp at the END of their period, so the five bars
        # covering 09:30-09:35 arrive at 09:31..09:35. Rev 1 used
        # `open_time <= t < range_end` and quietly made this a 4-minute range.
        if not (open_time < self.time <= range_end):
            return

        for symbol, sd in self.symbol_data.items():
            bar = data.bars.get(symbol)
            if bar is not None:
                sd.update_opening_range(bar)

        # Ranking is triggered from inside the data handler, not by a scheduled
        # event -- a scheduled event at +5min races the 09:34-09:35 bar.
        if self.time == range_end and not self._ranked_today:
            self._ranked_today = True
            self._rank_and_enter()

    # -----------------------------------------------------------------------
    # RANK AND ENTER
    # -----------------------------------------------------------------------

    def _rank_and_enter(self):
        self.transactions.cancel_open_orders()

        candidates = []
        for symbol, sd in self.symbol_data.items():
            if sd.or_open is None or sd.or_volume <= 0:
                continue

            # RVOL against PRIOR days only, THEN append today. Reversing these
            # contaminates the denominator with the numerator.
            if sd.indicators_ready:
                trailing = np.array(sd.open_vol_history, dtype=float)
                trailing = np.clip(trailing,
                                   np.percentile(trailing, 5),
                                   np.percentile(trailing, 95))
                avg = float(trailing.mean())
                sd.rvol = (sd.or_volume / avg) if avg > 0 else None

            was_ready = sd.indicators_ready
            sd.open_vol_history.append(sd.or_volume)

            if not was_ready or sd.rvol is None or sd.rvol < MIN_RVOL:
                continue

            security = self.securities[symbol]
            atr = float(sd.atr.current.value)
            if security.price < MIN_PRICE or atr <= 0:
                continue
            adv = float(sd.avg_volume.current.value)
            if adv < MIN_AVG_VOLUME:
                continue

            # Publish risk inputs for the slippage model BEFORE any order goes in.
            _RISK[str(symbol)] = {"atr": atr, "adv": adv}

            r_dollars = STOP_ATR_PCT * atr
            rvol_pct = sd.rvol * 100.0

            if FILTER_MODE == "flat":
                if atr < ATR_FLOOR:
                    continue
            elif FILTER_MODE == "joint":
                est_cost_r = (2 * COMMISSION_PER_SHARE
                              + estimate_round_trip_cents(float(security.price), atr)) / r_dollars
                if expected_gross_r(rvol_pct) <= est_cost_r:
                    continue

            if sd.or_close == sd.or_open:       # doji -> no trade, per spec
                continue

            sd.direction = (1 if sd.or_close > sd.or_open else -1) * DIRECTION_SIGN
            if sd.direction < 0 and not ALLOW_SHORTS:
                continue

            sd.r_dollars = r_dollars
            candidates.append((sd.rvol, symbol))

        if not candidates:
            return

        candidates.sort(reverse=True, key=lambda x: x[0])
        selected = (candidates[:N_HOLDINGS] if RANK_MODE == "top"
                    else candidates[4 * N_HOLDINGS: 5 * N_HOLDINGS])
        if not selected:
            return

        equity = float(self.portfolio.total_portfolio_value)

        # --- THE SIZING FIX -------------------------------------------------
        # Rev 5 used len(selected). On a one-name day that put 3.6x of the whole
        # account into a single stock. Dividing by N_HOLDINGS means a thin day
        # simply runs a smaller book, which is what a risk manager would do and
        # what a real 4x day-trading account would be forced into anyway.
        divisor = N_HOLDINGS if SIZING_FIX else len(selected)
        # --------------------------------------------------------------------

        for rvol, symbol in selected:
            sd = self.symbol_data[symbol]
            price = float(self.securities[symbol].price)
            if price <= 0 or sd.r_dollars <= 0:
                continue

            qty_by_risk = (RISK_PCT * equity) / sd.r_dollars
            qty_by_leverage = (MARGIN_BUFFER * MAX_LEVERAGE * equity / divisor) / price
            sd.bound_by = "risk" if qty_by_risk < qty_by_leverage else "leverage"
            qty = int(min(qty_by_risk, qty_by_leverage))
            if qty < 1:
                continue

            trigger = sd.or_high if sd.direction > 0 else sd.or_low
            signed_qty = qty if sd.direction > 0 else -qty
            sd.entry_ticket = self.stop_market_order(
                symbol, signed_qty, trigger, tag=f"ENTRY rvol={rvol:.2f}")
            sd.selected = True

    # -----------------------------------------------------------------------
    # ORDER EVENTS
    # -----------------------------------------------------------------------

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            return

        sd = self.symbol_data.get(order_event.symbol)
        if sd is None or not sd.selected:
            return

        if sd.entry_ticket is not None and order_event.order_id == sd.entry_ticket.order_id:
            sd.entry_price = float(order_event.fill_price)
            stop_price = (sd.entry_price - sd.r_dollars if sd.direction > 0
                          else sd.entry_price + sd.r_dollars)
            sd.stop_ticket = self.stop_market_order(
                order_event.symbol, -order_event.fill_quantity, stop_price, tag="STOP")

            # A rejected protective stop leaves the position running all day with
            # no -1R floor. Rare, but the outcome distribution is unbounded, and
            # in every run to date such trades were winners -- so the defect
            # flattered results. Detect synchronously and flatten.
            if sd.stop_ticket.status == OrderStatus.INVALID:
                self._unprotected += 1
                px = float(self.securities[order_event.symbol].price)
                self.liquidate(order_event.symbol, tag="NOSTOP")
                self._record_trade(sd, px, "NOSTOP")
                return

            self.open_positions[order_event.symbol] = sd

        elif sd.stop_ticket is not None and order_event.order_id == sd.stop_ticket.order_id:
            self._record_trade(sd, float(order_event.fill_price), "STOP")
            self.open_positions.pop(order_event.symbol, None)

    # -----------------------------------------------------------------------
    # END OF DAY
    # -----------------------------------------------------------------------

    def _on_close_out(self):
        self._ranked_today = False
        if self.is_warming_up:
            return

        # Cancel BEFORE liquidating. Rev 1 liquidated first, leaving live stops
        # that could fill against a flat book and open a phantom short.
        self.transactions.cancel_open_orders()

        for symbol, sd in list(self.open_positions.items()):
            if self.portfolio[symbol].invested:
                price = float(self.securities[symbol].price)
                self.liquidate(symbol, tag="MOC")
                self._record_trade(sd, price, "MOC")

        self.open_positions.clear()
        for sd in self.symbol_data.values():
            sd.reset_day()

    def _record_trade(self, sd, exit_price, exit_reason):
        if sd.entry_price is None or not sd.r_dollars or sd.r_dollars <= 0:
            return

        # LEAN applies the slippage model to FILL PRICES, so gross_r ALREADY
        # contains slippage. Only commission and borrow are separate charges.
        # (Rev 3 subtracted slippage a second time, understating every net
        # figure by ~0.07R and inverting the sign of the long book.)
        gross_r = ((exit_price - sd.entry_price) * sd.direction) / sd.r_dollars
        comm_r = (2 * COMMISSION_PER_SHARE) / sd.r_dollars

        borrow_r = 0.0
        if sd.direction < 0:
            price = float(self.securities[sd.symbol].price)
            borrow_r = (price * BORROW_ANNUAL_PCT / 252.0) / sd.r_dollars

        self.trade_log.append({
            "symbol": str(sd.symbol), "atr": float(sd.atr.current.value),
            "rvol": sd.rvol * 100.0, "reason": exit_reason,
            "bound_by": sd.bound_by, "gross_r": gross_r,
            "comm_r": comm_r,
            "net_r": gross_r - comm_r - borrow_r,
            "borrow_r": borrow_r, "direction": sd.direction,
            "entry_price": sd.entry_price, "r_dollars": sd.r_dollars,
        })

    # -----------------------------------------------------------------------
    # ATTRIBUTION -- automatic, so it cannot be skipped
    # -----------------------------------------------------------------------

    def on_end_of_algorithm(self):
        # ---- FRICTION AUDIT FIRST. Read this before any performance number. --
        self.log("=" * 70)
        self.log(f"FRICTION AUDIT   preset={FRICTION_PRESET}   sizing_fix={SIZING_FIX}")
        if _SLIP_LOG:
            arr_price   = np.array([s[0] for s in _SLIP_LOG])
            arr_total   = np.array([s[1] for s in _SLIP_LOG])
            arr_half    = np.array([s[2] for s in _SLIP_LOG])
            arr_adverse = np.array([s[3] for s in _SLIP_LOG])
            arr_impact  = np.array([s[4] for s in _SLIP_LOG])
            kinds       = np.array([s[5] for s in _SLIP_LOG])
            bp = arr_total / arr_price * 1e4
            self.log(f"  fills charged: {len(_SLIP_LOG)}   mean price ${arr_price.mean():.2f}")
            self.log(f"  mean slippage  {100*arr_total.mean():6.2f} c/side   "
                     f"= {bp.mean():5.2f} bp/side   (median {np.median(bp):5.2f} bp)")
            self.log(f"    half-spread  {100*arr_half.mean():6.2f} c   "
                     f"adverse {100*arr_adverse.mean():6.2f} c   "
                     f"impact {100*arr_impact.mean():6.2f} c")
            for k in ("stop", "market", "flat"):
                m = kinds == k
                if m.sum():
                    self.log(f"    {k:<7} n={int(m.sum()):>6}  "
                             f"{100*arr_total[m].mean():6.2f} c/side  "
                             f"{(arr_total[m]/arr_price[m]*1e4).mean():5.2f} bp/side")
        else:
            self.log("  NO SLIPPAGE CHARGES RECORDED -- the model is not wired in. "
                     "Every number below is invalid.")
        self.log("=" * 70)

        if not self.trade_log:
            self.log("No trades recorded. Check warm-up and universe filters.")
            return

        n = len(self.trade_log)
        gross = np.array([t["gross_r"] for t in self.trade_log])
        net = np.array([t["net_r"] for t in self.trade_log])
        comm = np.array([t["comm_r"] for t in self.trade_log])
        atrs = np.array([t["atr"] for t in self.trade_log])
        rvols = np.array([t["rvol"] for t in self.trade_log])
        rdol = np.array([t["r_dollars"] for t in self.trade_log])
        epx = np.array([t["entry_price"] for t in self.trade_log])
        stops = sum(1 for t in self.trade_log if t["reason"] == "STOP")
        lev = sum(1 for t in self.trade_log if t["bound_by"] == "leverage")

        self.log(f"MODE={MODE}  RANK={RANK_MODE}  DIR={DIRECTION_SIGN}  "
                 f"FILTER={FILTER_MODE}  UNIVERSE={UNIVERSE_SIZE}  N={N_HOLDINGS}")
        self.log("-" * 70)
        self.log(f"Trades {n}   stop-out {stops/n:.1%}   "
                 f"(expect ~65% at stop 0.20; ~80% at 0.10. A big gap means "
                 f"fills differ and everything downstream is noise.)")
        self.log(f"Leverage-bound sizing: {lev/n:.1%} of trades   "
                 f"(below ~97% and CAGR is not comparable across runs)")
        nostop = sum(1 for t in self.trade_log if t["reason"] == "NOSTOP")
        self.log(f"Stops rejected -> flattened immediately: {nostop} ({nostop/n:.2%})")
        self.log(f"Win rate (net) {(net>0).mean():.1%}")
        self.log(f"Mean gross {gross.mean():+.4f}R   commission {comm.mean():.4f}R   "
                 f"NET {net.mean():+.4f}R")
        # Edge in ATR units, so it is comparable across STOP_ATR_PCT settings.
        self.log(f"Edge in ATR units {net.mean()*STOP_ATR_PCT:+.5f}   "
                 f"| per-trade edge on notional {(net.mean()*rdol/epx).mean()*1e4:+.2f} bp")
        self.log(f"Median R ${np.median(rdol):.3f} = {np.median(rdol/epx)*100:.2f}% of price "
                 f"| breakeven extra friction {(net.mean()*rdol/epx).mean()*1e4/2:+.2f} bp/side")

        dirs = np.array([t["direction"] for t in self.trade_log])
        borrow = np.array([t["borrow_r"] for t in self.trade_log])
        for sgn, lab in ((1, "long "), (-1, "short")):
            m = dirs == sgn
            if m.sum() == 0:
                continue
            self.log(f"  {lab} n={m.sum():>6} ({m.mean():>5.1%})  net {net[m].mean():+.4f}R"
                     f"  (borrow {borrow[m].mean():.4f}R)  total {net[m].sum():+.1f}R")

        wins, losses = net[net > 0], net[net <= 0]
        if len(wins) and len(losses):
            self.log(f"Avg win {wins.mean():+.3f}R  avg loss {losses.mean():+.3f}R  "
                     f"ratio {abs(wins.mean()/losses.mean()):.2f}x")

        def bucket_table(title, values, edges, fmt):
            self.log("-" * 70)
            self.log(title)
            self.log(f"{'bucket':<14}{'n':>7}{'gross':>10}{'comm':>9}{'net':>10}")
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (values >= lo) & (values < hi)
                if m.sum() == 0:
                    continue
                label = fmt(lo, hi)
                self.log(f"{label:<14}{m.sum():>7}{gross[m].mean():>+10.4f}"
                         f"{comm[m].mean():>9.4f}{net[m].mean():>+10.4f}")

        bucket_table("NET R BY DOLLAR ATR", atrs,
                     [0.5, 1, 2, 3, 5, 10, 1e9],
                     lambda lo, hi: f"${lo:g}-{hi:g}" if hi < 1e9 else f">${lo:g}")

        bucket_table("NET R BY RVOL BUCKET", rvols,
                     [100, 500, 1000, 1500, 2000, 3000, 1e9],
                     lambda lo, hi: f"{lo:g}-{hi:g}%" if hi < 1e9 else f">{lo:g}%")

        # Price level is the lever with the most external support: friction in bp
        # is cents / price, which is arithmetic and holds in every regime.
        bucket_table("NET R BY ENTRY PRICE (the friction lever)", epx,
                     [0, 50, 100, 150, 250, 1e9],
                     lambda lo, hi: f"${lo:g}-{hi:g}" if hi < 1e9 else f">${lo:g}")

        self.log("-" * 70)
        self.log("CONCENTRATION (>15% of total = lucky draw, not factor)")
        by_symbol = {}
        for t in self.trade_log:
            by_symbol[t["symbol"]] = by_symbol.get(t["symbol"], 0.0) + t["net_r"]
        total = sum(abs(v) for v in by_symbol.values()) or 1.0
        for sym, r in sorted(by_symbol.items(), key=lambda kv: -abs(kv[1]))[:10]:
            self.log(f"  {sym:<10}{r:>+9.1f}R{abs(r)/total:>8.1%}")
        self.log("=" * 70)


# ---------------------------------------------------------------------------
# CHANGELOG
#
# REV 6 -- 2026-09-14, after order-book forensics on the rev-5 IS artifact
#  14. SIZING. `equity / len(selected)` -> `equity / N_HOLDINGS`. 285 trades in
#      the rev-5 IS run carried >1.0x of the entire account in one name (max
#      3.50x). On the replay, fixing it improved CAGR, Sharpe and drawdown at
#      once -- uncompensated variance, not edge.
#  15. FILL MODEL. Flat 2c/side replaced with half-spread + stop adverse-
#      selection + optional square-root impact. Rev 5 filled 85.7% of entries
#      and 83.6% of protective stops at EXACTLY trigger+2c. Breakeven extra
#      friction on the rev-5 IS run was 5.31 bp/side; the model had no component
#      capable of consuming it.
#  16. _WINDOWS["OOS"] was (2001,1,1)-(2010,8,16) while its own comment said
#      2020-2026. Corrected. The OOS window is spent either way.
#  17. Reporting: friction audit block, per-trade edge on notional, breakeven
#      friction, and a net-R-by-entry-price table (price level is the one
#      friction lever with independent external support).
#
# REV 5 -- margin defects surfaced from the raw logs
#  12. Rejected protective stops -- submitted from inside on_order_event before
#      the portfolio registers the position, so LEAN read the closing order as
#      OPENING a short and demanded margin we did not have at 3.42x gross.
#      Each left a position running with no -1R floor; all were winners, so the
#      defect flattered results. Now detected synchronously and flattened.
#  13. Margin calls at 15:50 -- PatternDayTradingMarginModel drops 4x->2x about
#      10 minutes before the close. CLOSE_OUT_MINUTES = 12 flattens ahead of it.
#
# REV 4 -- slippage double-count in attribution. gross_r is computed from LEAN
#      fill prices which already include slippage; cost_r added it a second time,
#      understating every net figure by ~0.07R and inverting the sign of the long
#      book. Lesson: fix the instrument when you find the fault, not just the report.
#
# REV 3 -- MARGIN_BUFFER referenced but never defined (a str.replace no-op'd
#      silently); MAX_LEVERAGE=4.0 unreachable under Reg T 2x so 26.1% of entries
#      were rejected and realised gross was 2.13x; rejection spam consumed QC's
#      100kb log cap and discarded the whole attribution block.
#
# REV 2 -- crashed on load: class(SlippageModel) (no such base in QC Python),
#      class(FeeModel) (exists but fragile). Ran silently wrong: 4-minute opening
#      range (bars stamp at period END); scheduled ranking raced the final bar;
#      MIN_AVG_VOLUME declared and never enforced; volume SMA fed opening-range
#      volume instead of daily volume; EOD liquidate ran before cancelling stops.
# ---------------------------------------------------------------------------
