# =============================================================================
# regime/backtest.py
#
# Backtest V1 — validates the Layer 1 regime classifier against a hand-labelled
# list of historical episodes from 1974 onward. For each episode we take the
# midpoint month, reconstruct a point-in-time signals dict from fetch_historical_panel(),
# run classify_regime(), and compare the output to the expected label.
#
# Output: a DataFrame with one row per episode (Episode | Period | Expected |
# Classifier | Hit) plus a hit-rate summary (full hits vs partial vs misses).
#
# Why midpoint (not rolling): V1 keeps the signal low — one representative
# sample per episode. Picking midpoints avoids edge effects where the regime
# is still transitioning. V2 could roll monthly across each episode window
# and report regime-share rather than a single label.
#
# Data availability (binding constraints, affects pre-2003 accuracy):
#   DGS2 / T10Y2Y   from 1976
#   MICH            from 1978
#   USSLIND (LEI)   from 1982
#   T5YIFR / DFII10 from 2003
# The classifier handles these gaps gracefully (skip-and-rescale thresholds).
# =============================================================================

from __future__ import annotations

import pandas as pd

from regime.classifier import classify_regime, RegimeResult


# -----------------------------------------------------------------------------
# HISTORICAL_EPISODES
# Hand-labelled episodes with widely-accepted regime attribution. Each tuple:
#   (label, expected_regime, start_YYYY-MM, end_YYYY-MM, rationale)
#
# Sources for the labelling: Fed research papers, NBER dating, and the regime
# framework's own logic applied retrospectively to CPI / PMI / GDP data.
# These are the episodes an interviewer would expect a macro framework to
# correctly identify — the "face validity" test.
# -----------------------------------------------------------------------------
HISTORICAL_EPISODES: list[tuple[str, str, str, str, str]] = [
    ("1974 Oil Shock",              "Stagflation",    "1974-01", "1975-03",
     "OPEC embargo — CPI peaked near 12% while GDP contracted"),
    ("Volcker Recession",           "Deflation/Bust", "1981-09", "1982-10",
     "Fed Funds at 19% crushed demand; CPI and growth both collapsing"),
    ("Mid-80s Expansion",           "Goldilocks",     "1985-01", "1987-06",
     "Disinflation from Volcker era meets strong growth"),
    ("Late-80s Overheating",        "Overheating",    "1988-06", "1989-06",
     "Core CPI re-accelerating, Fed hiking, labour market tight"),
    ("Early-90s Recession",         "Deflation/Bust", "1990-09", "1991-03",
     "Gulf War oil shock + S&L crisis — growth contracts, disinflation"),
    ("Mid-90s Goldilocks",          "Goldilocks",     "1995-06", "1997-12",
     "Productivity boom, CPI anchored, strong employment — the canonical case"),
    ("Dot-com Bust",                "Deflation/Bust", "2001-03", "2002-10",
     "Tech-led recession; CPI drifting below 2%"),
    ("Mid-00s Overheating",         "Overheating",    "2004-06", "2006-06",
     "Housing boom, commodities surging, Fed hiking 17 consecutive meetings"),
    ("Global Financial Crisis",     "Deflation/Bust", "2008-09", "2009-06",
     "Lehman collapse — CPI briefly negative, growth -4% annualised"),
    ("Post-GFC Goldilocks",         "Goldilocks",     "2013-01", "2015-06",
     "Recovery with PCE stuck below target; QE tailwind"),
    ("COVID Crash",                 "Deflation/Bust", "2020-03", "2020-05",
     "Oil briefly negative, jobless claims at record, deflation scare"),
    ("Post-COVID Overheating",      "Overheating",    "2021-06", "2022-02",
     "Re-opening boom, CPI climbing to 7%+, unemployment falling"),
    ("2022 Stagflation Scare",      "Stagflation",    "2022-06", "2022-12",
     "CPI 9%, Fed 75bp hikes, PMI rolling over, yield curve inverting"),
    ("2024 Disinflation",           "Goldilocks",     "2024-01", "2024-12",
     "CPI cooling toward target, growth resilient, Fed cutting cycle begins"),
]


# Axis decomposition — used to distinguish a "partial hit" (one axis right,
# other axis wrong) from a full miss. Tuple format: (growth_up, inflation_up).
REGIME_AXES: dict[str, tuple[bool, bool]] = {
    "Overheating":    (True,  True),
    "Stagflation":    (False, True),
    "Goldilocks":     (True,  False),
    "Deflation/Bust": (False, False),
}


# -----------------------------------------------------------------------------
# build_signals_at()
# Point-in-time reconstruction of the signals dict consumed by classify_regime().
# Same logic as fetch_macro_inputs() in data/fetcher.py, but operating on
# pre-fetched historical series (panel dict) sliced via .loc[:date].
#
# Any signal whose underlying series doesn't yet exist at `date` is simply
# omitted — the classifier's graceful-degradation path handles that.
# -----------------------------------------------------------------------------
def build_signals_at(date: pd.Timestamp, panel: dict[str, pd.Series]) -> dict[str, float]:
    """Construct the classifier input dict as of a specific point in time."""

    signals: dict[str, float] = {}

    def _slice(sid: str) -> pd.Series | None:
        """Return the panel series sliced up to `date`, or None if unavailable."""
        s = panel.get(sid)
        if s is None or s.empty:
            return None
        s = s.loc[:date]
        return s if not s.empty else None

    # --- PMI proxy (INDPRO 6m annualised growth mapped to 0-100 PMI scale) ---
    # Mirrors fetcher.fetch_macro_inputs(): pmi = 50 + annualised_pct * 2.
    indpro = _slice("INDPRO")
    if indpro is not None and len(indpro) >= 7:
        pct6 = indpro.pct_change(6).dropna()
        if len(pct6) >= 1:
            # 6m pct × (12/6) → annualised, then ×2 to map onto PMI scale
            signals["pmi_proxy"] = round(50 + float(pct6.iloc[-1] * 100) * 4, 1)

    # --- LEI MoM (Conference Board leading index) ---
    lei = _slice("USSLIND")
    if lei is not None and len(lei) >= 2:
        mom = lei.pct_change().dropna()
        if len(mom) >= 1:
            signals["lei_mom"] = round(float(mom.iloc[-1] * 100), 3)

    # --- Initial claims 4-week vs prior 4-week trend ---
    # ICSA is weekly, so 8 observations = last 8 weeks.
    claims = _slice("ICSA")
    if claims is not None and len(claims) >= 8:
        now4  = float(claims.iloc[-4:].mean())
        prev4 = float(claims.iloc[-8:-4].mean())
        signals["claims_trend_change"] = round(now4 - prev4, 0)

    # --- 10y / 2y yield 1-month changes (≈20 trading days back) ---
    y10 = _slice("DGS10")
    y2  = _slice("DGS2")
    if y10 is not None and len(y10) >= 21:
        signals["yield_10y_change"] = round(float(y10.iloc[-1]) - float(y10.iloc[-21]), 3)
    if y2 is not None and len(y2) >= 21:
        signals["yield_2y_change"]  = round(float(y2.iloc[-1])  - float(y2.iloc[-21]),  3)
    # Derived spread change (required alongside yield_10y_change for bear-steepener vote)
    if "yield_10y_change" in signals and "yield_2y_change" in signals:
        signals["spread_10y2y_change"] = round(
            signals["yield_10y_change"] - signals["yield_2y_change"], 3
        )

    # --- CPI YoY current and lag (3 months back) ---
    # Need >=16 obs: 12 for YoY, +4 more for 3m lag.
    cpi = _slice("CPIAUCSL")
    if cpi is not None and len(cpi) >= 16:
        yoy = (cpi.pct_change(12) * 100).dropna()
        if len(yoy) >= 4:
            signals["cpi_yoy"]     = round(float(yoy.iloc[-1]), 3)
            signals["cpi_yoy_lag"] = round(float(yoy.iloc[-4]), 3)

    # --- Core CPI YoY current and lag ---
    core = _slice("CPILFESL")
    if core is not None and len(core) >= 16:
        yoy = (core.pct_change(12) * 100).dropna()
        if len(yoy) >= 4:
            signals["core_cpi_yoy"]     = round(float(yoy.iloc[-1]), 3)
            signals["core_cpi_yoy_lag"] = round(float(yoy.iloc[-4]), 3)

    # --- PPI MoM ---
    ppi = _slice("PPIACO")
    if ppi is not None and len(ppi) >= 2:
        mom = ppi.pct_change().dropna()
        if len(mom) >= 1:
            signals["ppi_mom"] = round(float(mom.iloc[-1] * 100), 3)

    # --- 5Y5Y breakeven current and 3m lag (daily series; iloc[-61] ≈ 3m back) ---
    be = _slice("T5YIFR")
    if be is not None and len(be) >= 61:
        signals["breakeven_5y5y"]     = round(float(be.iloc[-1]),   3)
        signals["breakeven_5y5y_lag"] = round(float(be.iloc[-61]),  3)

    # --- Michigan 1Y inflation expectations ---
    mich = _slice("MICH")
    if mich is not None and len(mich) >= 1:
        signals["michigan_exp"] = round(float(mich.iloc[-1]), 2)

    return signals


# -----------------------------------------------------------------------------
# _episode_midpoint()
# Return the calendar midpoint between an episode's start month and end month.
# Month-end on the end side so short episodes (1-3 months) still get a sensible
# sample date rather than collapsing to the start.
# -----------------------------------------------------------------------------
def _episode_midpoint(start_str: str, end_str: str) -> pd.Timestamp:
    start_ts = pd.Timestamp(start_str)
    end_ts   = pd.Timestamp(end_str) + pd.offsets.MonthEnd(0)
    return start_ts + (end_ts - start_ts) / 2


# -----------------------------------------------------------------------------
# _classify_hit()
# Compare expected vs actual regime. Returns one of:
#   "Hit"        — full match
#   "Partial"    — one of growth/inflation axes matches, other doesn't
#   "Miss"       — both axes wrong (i.e. diagonal opposite quadrant)
#   "No data"    — classifier returned Insufficient data
# -----------------------------------------------------------------------------
def _classify_hit(expected: str, actual: str) -> str:
    if actual == "Insufficient data":
        return "No data"
    if expected == actual:
        return "Hit"
    if expected in REGIME_AXES and actual in REGIME_AXES:
        exp_g, exp_i = REGIME_AXES[expected]
        act_g, act_i = REGIME_AXES[actual]
        if exp_g == act_g or exp_i == act_i:
            return "Partial"
    return "Miss"


# -----------------------------------------------------------------------------
# backtest_episodes()
# Run classifier on the midpoint of each historical episode.
# Returns a DataFrame with one row per episode.
# -----------------------------------------------------------------------------
def backtest_episodes(panel: dict[str, pd.Series]) -> pd.DataFrame:
    """Run the classifier at each episode's midpoint and tabulate hits vs misses."""

    rows: list[dict] = []
    for label, expected, start, end, rationale in HISTORICAL_EPISODES:
        mid = _episode_midpoint(start, end)
        signals = build_signals_at(mid, panel)
        result: RegimeResult = classify_regime(signals)
        hit = _classify_hit(expected, result.regime)

        rows.append({
            "Episode":    label,
            "Period":     f"{start} → {end}",
            "Sample":     mid.strftime("%Y-%m"),
            "Expected":   expected,
            "Classifier": result.regime,
            "Hit":        hit,
            # Keep score breakdown for optional drill-down UI / debugging
            "G score":    f"{result.growth_score}/{result.growth_available}",
            "I score":    f"{result.inflation_score}/{result.inflation_available}",
            "Confidence": result.confidence,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# hit_rate_summary()
# Aggregate the backtest DataFrame into summary counts + percentages.
# Partial hits count as half in the weighted score — credits the classifier for
# getting one axis right while still penalising the mismatch.
# -----------------------------------------------------------------------------
def hit_rate_summary(results: pd.DataFrame) -> dict[str, float]:
    """Return summary stats from a backtest DataFrame."""
    total    = len(results)
    hits     = int((results["Hit"] == "Hit").sum())
    partials = int((results["Hit"] == "Partial").sum())
    misses   = int((results["Hit"] == "Miss").sum())
    no_data  = int((results["Hit"] == "No data").sum())

    # Scorable = total minus no-data (we can't fairly grade insufficient-data cases)
    scorable = total - no_data
    hit_rate = (hits / scorable * 100) if scorable else 0.0
    weighted = ((hits + 0.5 * partials) / scorable * 100) if scorable else 0.0

    return {
        "total":            total,
        "hits":             hits,
        "partials":         partials,
        "misses":           misses,
        "no_data":          no_data,
        "hit_rate_pct":     round(hit_rate, 1),
        "weighted_pct":     round(weighted, 1),
    }


# -----------------------------------------------------------------------------
# CLI smoke test — run `python3 regime/backtest.py` to verify end-to-end.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from data.fetcher import fetch_historical_panel

    print("Fetching historical panel (first run may take ~30s)...")
    panel = fetch_historical_panel()
    for sid, s in panel.items():
        print(f"  {sid:<10} {len(s):>6} obs  {s.index.min().date()} → {s.index.max().date()}")

    print("\nRunning backtest...")
    df = backtest_episodes(panel)
    print(df.to_string(index=False))

    print("\nSummary:")
    for k, v in hit_rate_summary(df).items():
        print(f"  {k:<15} {v}")
