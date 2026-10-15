# Run index

Every number in the main README comes from one of these QuantConnect backtests. Run names are
QuantConnect's auto-generated IDs. All runs: $100,000 start, 500 most liquid US stocks
(point-in-time, delisted names included), 5 positions, flat by the close.

| Run | Code | Window | What it tests | Fills | Trades | Net R / trade | CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|
| **Calculating Tan Alligator** | [`v8`](../code/orb_open_v8.py) | **OOS** 2020 – Jun 2026 | **The pre-registered test** of the frozen config | measured | 5,294 | **−0.0855** | −63.5% | −0.686 | 99.9% |
| Jumping Orange Frog | [`v5`](../code/archive/orb_open_v5.py) | IS 2016–19 | Best in-sample config; frozen for the test | measured | 1,205 | +0.1514 | +23.5% | 0.600 | 36.9% |
| Upgraded Fluorescent Orange Pelican | [`v7`](../code/archive/orb_open_v7.py) | IS 2016–19 | Source of the in-sample cost decomposition | measured | 3,370 | +0.0025 | −13.4% | −0.214 | 66.7% |
| Alert Red Orange Sardine | rev 5 ¹ | OOS 2020 – Jun 2026 | Paper's spec (with a 1R stop), out-of-sample | 2¢/side | 6,609 | +0.0696 | +33.5% | 0.738 | 44.7% |
| Geeky Fluorescent Pink Shark | [`v5`](../code/archive/orb_open_v5.py) | OOS 2020 – Jun 2026 | Frozen config with free fills | 2¢/side | 6,470 | +0.0977 | +2.3% | 0.424 | 97.0% |
| Creative Apricot Chicken | [`v5`](../code/archive/orb_open_v5.py) | 2016 – Jun 2026 | Frozen config, free fills, full history | 2¢/side | 7,683 | +0.1266 | +22.1% | 0.585 | 96.7% |

¹ Reproduce with [`archive/orb_rvol_rev6.py`](../code/archive/orb_rvol_rev6.py) set to
`MODE="OOS"`, `FRICTION_PRESET="paper"`, `SIZING_FIX=False`. Rev 6 differs from rev 5 only in
those two switches (see its changelog).

**Fills.** *measured* = 1.5 bp half-spread on both sides + adverse selection of 0.025 × ATR on the
breakout entry + $0.0035/share commission + 3%/yr borrow on shorts. *2¢/side* = a flat two
cents per share per fill, roughly the paper's assumption.

**Sharpe and CAGR** are QuantConnect's portfolio statistics. CAGR is not comparable across runs
where position sizing was capped differently; net R per trade is size-independent and is the
primary metric throughout.

