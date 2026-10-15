# region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
import math
# endregion

"""
OPENING-RANGE BREAKOUT, NO STOP  --  v4, 2026-09-16   (RECOVER THE CAGR)

THE IDEA
--------
Every run in this project has established the same thing from a different angle:
a resting STOP order at the opening-range extreme costs ~10 bp/side because it
sits at the most-watched price of the session and gets run over. Breakeven is
~8.4c/side. The edge is real (+15 bp gross) and it is inseparable from the order
type that makes it unaffordable.

If somebody is collecting that 10 bp, the obvious question is whether we can be
them. So: rest a LIMIT at the same level instead of a stop, and provide the
liquidity the breakout trader is taking.

    GREEN opening candle -> rest a SELL LIMIT at or_high   (short the up-break)
    RED   opening candle -> rest a BUY  LIMIT at or_low    (long  the down-break)

Same price level as the momentum entry. Opposite side of the same trade.

THIS IS NOT THE OLD DIRECTION CONTROL
-------------------------------------
rev6 computed `direction = (+1 if green else -1) * DIRECTION_SIGN` and then
`trigger = or_high if direction > 0 else or_low`. With DIRECTION_SIGN=-1 and a
green candle that becomes a SELL STOP at the LOW -- shorting the downside break
of an up-candle, a different price level entirely. It was never a fade, and it
paid the spread to do whatever it was doing. Path 2 is genuinely untested.

THE MIRROR PROPERTY -- THE PRIMARY CHECK, NOT THE RESULT
--------------------------------------------------------
LEAN triggers a stop when price TOUCHES the level. LEAN fills a limit when price
TOUCHES the level. Same level, same bars, therefore:

    ENTRY_MODE="stop" and ENTRY_MODE="limit" TRADE THE SAME SET, MIRRORED.

    net_R(limit)  ==  -net_R(stop)  +  the spread we no longer cross

RUN BOTH. If that identity does not hold to within the modelled spread, one of
the two fill models is wrong and every number is noise. This is the check the
project has never had: two order types priced against each other on the same
trades.

The "stop" mode is independently valuable -- it fills the one cell of the 2x2
this project has never run: BREAK ENTRY + NO STOP.

               |  tight stop 0.20xATR  |  no stop
    -----------+-----------------------+------------------
    break      |  +0.1373R (rev6 run2) |  ENTRY_MODE="stop"   <- this file
    market     |  (never run)          |  -0.1338R (v2)

WHAT IS GENUINELY UNCERTAIN, AND IS THEREFORE A SWEPT PARAMETER
---------------------------------------------------------------
When the momentum trader pays 12.4c to enter, ~2.2c is spread and ~10c is
adverse selection. The maker certainly stops paying the spread -- that is
mechanical, and LEAN's limit fill already gives it to us for free by filling at
the limit price. Whether the maker also CAPTURES part of the adverse-selection
premium depends on something unknowable from bar data: is the taker's bad fill a
transfer to resting sellers, or the result of sweeping several price levels?

MAKER_ADVERSE_CAPTURE makes that explicit. 0.0 = the maker collects nothing
beyond the spread (pessimistic, the default). 1.0 = the maker collects
everything the taker pays. The honest answer is between. SWEEP IT -- the result
is the curve, not any single point.

THE FILL MODEL IS OPTIMISTIC FOR LIMITS, IN THE MIRROR OF HOW IT WAS
OPTIMISTIC FOR STOPS. READ THIS BEFORE BELIEVING ANYTHING.
--------------------------------------------------------------------
LEAN fills a resting limit whenever the bar touches it. In reality you are in a
queue: if price touches your price and reverses you probably do NOT fill, and
the fills you DO get are disproportionately the ones where price kept going --
against you. So LEAN hands the maker the good fills for free and the real world
does not.

That is exactly the 2-cent stop problem wearing the other hat, and it is the
single most likely way this run flatters us. LIMIT_OFFSET_ATR is the defence:
it rests the order BEYOND the extreme, so price must trade through with real
force before we are filled.

    PRE-REGISTERED: if this looks good at LIMIT_OFFSET_ATR = 0.0 and collapses
    at 0.05, it is a fill artifact and must be retracted. Run the offset sweep
    BEFORE quoting any headline number.

PRE-REGISTERED PREDICTION (derived, not guessed, from data already in hand)
---------------------------------------------------------------------------
From rev6: 66% stop out at -1.05R, 34% reach the close at +2.43R.
From v1's autopsy: trades through -1R close at -1.0929R when let run.
Therefore break-entry-no-stop  =  0.66(-1.0929) + 0.34(2.43)  =  +0.105R.

    ENTRY_MODE="stop",  paper   ->  net R  +0.08 to +0.13
    ENTRY_MODE="limit", paper   ->  net R  -0.08 to -0.05   (mirror, plus 2c)
    ENTRY_MODE="limit", measured, MAKER_ADVERSE_CAPTURE=0.0  ->  -0.08 to -0.05
    ENTRY_MODE="limit", measured, MAKER_ADVERSE_CAPTURE=1.0  ->  +0.02 to +0.06

    So the whole question is the capture parameter, and the run's job is to
    produce the curve across it rather than a verdict.

    FALSIFIER FOR THE WHOLE IDEA: if limit mode is negative even at
    MAKER_ADVERSE_CAPTURE=1.0, liquidity provision at this level does not pay
    under any assumption and Path 2 closes.

RUN ORDER  (u500: ~40 min each, and it is what every old-build number used)
---------------------------------------------------------------------------
  1. ENTRY_MODE="stop",  paper   -> fills the missing 2x2 cell
  2. ENTRY_MODE="limit", paper   -> MIRROR CHECK against run 1
  3. ENTRY_MODE="limit", measured, MAKER_ADVERSE_CAPTURE = 0.0 / 0.5 / 1.0
  4. ENTRY_MODE="limit", measured, LIMIT_OFFSET_ATR = 0.05  -> fill-artifact test
"""

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODE = "IS"
FRICTION_PRESET = "measured"

# THE ONE VARIABLE UNDER TEST IN RUN A.
# R/price = 0.20 x ATR/price, i.e. how much this stock actually moves relative
# to what it costs. It has sorted cleanly in FIVE consecutive runs across two
# friction presets and three entry mechanisms, while RVOL has failed to sort in
# all five. Friction/R varies only ~0.025R across the whole range, so this is a
# SIGNAL effect, not a cost effect.
#   0.000 = off (every run so far)
#   0.008 = ATR/price >= 4%   <- run A
REL_R_MIN       = 0.008

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
UNIVERSE_SIZE   = 500            # u500: comparable to every old-build run, 4x faster
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
        self.set_warm_up(timedelta(days=45), WARMUP_RESOLUTION)

        self.log("=" * 72)
        self.log(f"OR LIQUIDITY PROVISION v3  MODE={MODE}  FRICTION={FRICTION_PRESET}  "
                 f"u{UNIVERSE_SIZE}  N={N_HOLDINGS}")
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

            candidates.append((sd.rvol, symbol))

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
            "rvol": sd.rvol * 100.0, "reason": exit_reason, "bound_by": sd.bound_by,
            "gross_r": gross_r, "comm_r": comm_r,
            "net_r": gross_r - comm_r - borrow_r, "borrow_r": borrow_r,
            "direction": sd.direction, "entry_price": sd.entry_price,
            "r_dollars": sd.r_dollars, "mae_r": mae_r, "mfe_r": mfe_r,
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

        self.log(f"Trades {n}   ENTRY_MODE={ENTRY_MODE}")
        self.log(f"Win rate {(net>0).mean():.1%}   mean gross {gross.mean():+.4f}R   "
                 f"NET {net.mean():+.4f}R")
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
        # ---- THREE FREE SWEEPS FROM ONE RUN --------------------------------
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
        self.log("SWEEP 1of3 -- STOP WIDTH (exit on adverse excursion)")
        self.log(f"  {'stop':<16}{'% hit':>8}{'net R':>10}{'vs none':>10}")
        for S in _STOP_SWEEP:
            hit = mae <= -S
            sim = np.where(hit, -S - stop_cost_r, net)
            self.log(f"  {f'{S:g}R ({S*R_ACCOUNTING_ATR_PCT:.2f}xATR)':<16}"
                     f"{100*hit.mean():>7.1f}%{sim.mean():>+10.4f}{sim.mean()-base:>+10.4f}")

        self.log("-" * 72)
        self.log("SWEEP 2of3 -- PROFIT TARGET (the give-back nobody has examined)")
        self.log(f"  mean MFE {mfe.mean():+.3f}R vs mean NET {base:+.4f}R -- "
                 f"the average trade gives back {mfe.mean()-base:.2f}R from its peak")
        self.log(f"  {'target':<16}{'% hit':>8}{'net R':>10}{'vs none':>10}")
        for T in _TARGET_SWEEP:
            hit = mfe >= T
            sim = np.where(hit, T - stop_cost_r, net)
            self.log(f"  {f'+{T:g}R ({T*R_ACCOUNTING_ATR_PCT:.2f}xATR)':<16}"
                     f"{100*hit.mean():>7.1f}%{sim.mean():>+10.4f}{sim.mean()-base:>+10.4f}")

        self.log("-" * 72)
        self.log("SWEEP 3of3 -- EXIT TIME (does the edge decay through the day?)")
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

        def bucket(title, vals, edges, fmt):
            self.log("-" * 72); self.log(title)
            self.log(f"{'bucket':<14}{'n':>7}{'net R':>10}{'edge bp':>10}")
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (vals >= lo) & (vals < hi)
                if m.sum() < 20:
                    continue
                self.log(f"{fmt(lo,hi):<14}{int(m.sum()):>7}{net[m].mean():>+10.4f}"
                         f"{(net[m]*rdol[m]/epx[m]).mean()*1e4:>+10.2f}")

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
