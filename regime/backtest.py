# =============================================================================
# regime/backtest.py
#
# Backtest V2 — validates the Layer 1 regime classifier against a hand-labelled
# list of historical episodes from 1974 onward. For each episode we sample the
# classifier at every month in the window, reconstructing a point-in-time
# signals dict from fetch_historical_panel() and running classify_regime().
# Episode verdict is the *plurality* of the monthly calls (majority vote).
#
# Why monthly plurality (not midpoint): a 2.5-year episode isn't a single
# month — the classifier will legitimately drift as signals turn. A midpoint
# sample is a coin-flip on transition months. Plurality voting gives the
# expected regime room to dominate the window even if one or two months miss.
#
# Output: a DataFrame with one row per episode (Episode | Period | Months |
# Expected | Plurality | Share | Hit) plus a hit-rate summary.
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

    # --- CPI disinflation override ---
    # Needs 24 months of CPI so we can compute 12 months of YoY values.
    # cpi_rolled_over fires when CPI YoY is ≥0.5pp below its trailing 12m high;
    # cpi_3m_decel fires when YoY is below its 3-month-ago value. When BOTH
    # fire, the classifier treats inflation as "not rising" regardless of the
    # other vote counts — captures late-2022-onward disinflation.
    if cpi is not None and len(cpi) >= 24:
        yoy_full = (cpi.pct_change(12) * 100).dropna()
        if len(yoy_full) >= 12:
            cpi_12m_peak = float(yoy_full.tail(12).max())
            cpi_peak_gap = cpi_12m_peak - float(yoy_full.iloc[-1])
            signals["cpi_peak_gap"]   = round(cpi_peak_gap, 3)
            signals["cpi_rolled_over"] = cpi_peak_gap > 0.5
            if len(yoy_full) >= 4:
                signals["cpi_3m_decel"] = float(yoy_full.iloc[-1]) < float(yoy_full.iloc[-4])

    # --- Sahm-rule recession trigger ---
    # 3m MA of UNRATE ≥ 0.5pp above its trailing 12m minimum. Hard override:
    # if fired, classifier returns Deflation/Bust regardless of scoring.
    unrate = _slice("UNRATE")
    if unrate is not None and len(unrate) >= 15:
        unrate_3m = unrate.rolling(3).mean().dropna()
        if len(unrate_3m) >= 12:
            sahm_current = float(unrate_3m.iloc[-1])
            sahm_12m_min = float(unrate_3m.tail(12).min())
            signals["sahm_trigger"] = (sahm_current - sahm_12m_min) >= 0.5

    return signals


# -----------------------------------------------------------------------------
# _episode_months()
# Return the list of month-end timestamps spanning an episode (inclusive).
# Sampling month-end ensures the latest monthly FRED release for that month
# is available (CPI, PPI, INDPRO all publish mid-following-month but .loc[:date]
# will still pick up the most recent dated observation).
# -----------------------------------------------------------------------------
def _episode_months(start_str: str, end_str: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start_str) + pd.offsets.MonthEnd(0)
    end_ts   = pd.Timestamp(end_str)   + pd.offsets.MonthEnd(0)
    return list(pd.date_range(start=start_ts, end=end_ts, freq="ME"))


# -----------------------------------------------------------------------------
# _grade_episode()
# Given the expected regime and a count of classifier calls across the episode,
# return a verdict. Plurality voting: the classifier gets credit for being
# right *most of the time*, not just at a single midpoint.
#
#   Hit       — expected regime is the plurality (most common call)
#   Partial   — expected regime appears in ≥25% of months (visible but not dominant)
#   Miss      — expected regime appears in <25% of months
#   No data   — every month returned Insufficient data
# -----------------------------------------------------------------------------
def _grade_episode(expected: str, counts: dict[str, int], total: int) -> str:
    scorable = total - counts.get("Insufficient data", 0)
    if scorable == 0:
        return "No data"

    # Plurality excludes no-data months so a handful of early missing-signal
    # observations don't poison the verdict.
    scorable_counts = {k: v for k, v in counts.items() if k != "Insufficient data"}
    plurality = max(scorable_counts, key=scorable_counts.get)
    expected_share = scorable_counts.get(expected, 0) / scorable

    if plurality == expected:
        return "Hit"
    if expected_share >= 0.25:
        return "Partial"
    return "Miss"


# -----------------------------------------------------------------------------
# backtest_episodes()
# Run classifier at every month of each historical episode and tabulate a
# plurality verdict. Returns a DataFrame with one row per episode.
# -----------------------------------------------------------------------------
def backtest_episodes(panel: dict[str, pd.Series]) -> pd.DataFrame:
    """Run the classifier monthly across each episode window; grade by plurality."""

    rows: list[dict] = []
    for label, expected, start, end, rationale in HISTORICAL_EPISODES:
        months = _episode_months(start, end)
        counts: dict[str, int] = {}
        for ts in months:
            signals = build_signals_at(ts, panel)
            result: RegimeResult = classify_regime(signals)
            counts[result.regime] = counts.get(result.regime, 0) + 1

        total = len(months)
        hit = _grade_episode(expected, counts, total)

        # Plurality = most common regime across the window (ignore no-data months
        # when ranking). Share = plurality's % of scorable months.
        scorable_counts = {k: v for k, v in counts.items() if k != "Insufficient data"}
        if scorable_counts:
            plurality = max(scorable_counts, key=scorable_counts.get)
            scorable  = sum(scorable_counts.values())
            share     = scorable_counts[plurality] / scorable * 100
            exp_share = scorable_counts.get(expected, 0) / scorable * 100
        else:
            plurality, share, exp_share = "Insufficient data", 0.0, 0.0

        rows.append({
            "Episode":   label,
            "Period":    f"{start} → {end}",
            "Months":    total,
            "Expected":  expected,
            "Plurality": plurality,
            "Share":     f"{share:.0f}%",
            "Expected %": f"{exp_share:.0f}%",
            "Hit":       hit,
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
