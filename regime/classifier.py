# =============================================================================
# regime/classifier.py
#
# The regime classification engine. Takes a dict of macro signals and outputs
# one of four regime labels using a transparent scoring system.
#
# Three-layer framework (only Layer 1 is fully implemented):
#   Layer 1 — Growth/Inflation Quadrant  ← implemented here
#   Layer 2 — Monetary Policy Cycle      ← stubbed, to be added
#   Layer 3 — RORO (Risk-on/Risk-off)    ← stubbed, to be added
# =============================================================================

from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# THRESHOLDS
# All decision boundaries in one dict. Changing a threshold here affects
# the classifier without touching any logic code.
#
# growth_score_min     — need at least this many growth signals firing to
#                        classify growth as "accelerating" (out of 4 signals)
# inflation_score_min  — same for inflation signals (out of 4 signals)
# pmi_expansion        — PMI proxy above this = growth is expanding
# michigan_hot         — Michigan 1Y inflation expectations above this = hot
# -----------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    "growth_score_min":    2,     # out of 4 growth signals
    "inflation_score_min": 2,     # out of 4 inflation signals
    "pmi_expansion":       50.0,  # PMI proxy: above = expanding
    "michigan_hot":        3.0,   # Michigan 1Y expectations above 3% = elevated
}


# -----------------------------------------------------------------------------
# REGIME_COLOURS
# Used consistently across the regime flag, charts, and heatmap.
# -----------------------------------------------------------------------------
REGIME_COLOURS: dict[str, str] = {
    "Stagflation":      "#E05252",  # red   — rising inflation, falling growth
    "Overheating":      "#F5A623",  # amber — rising inflation, rising growth
    "Goldilocks":       "#4CAF50",  # green — falling inflation, rising growth
    "Deflation/Bust":   "#5B9BD5",  # blue  — falling inflation, falling growth
}


# -----------------------------------------------------------------------------
# REGIME_RETURNS
# Approximate annualised asset returns (%) per regime, 1972–present.
# Seeds the heatmap chart — can be refined with backtested data over time.
#
# Read as: "when the economy is in X regime, asset Y has historically returned
# Z% per year on average."
# -----------------------------------------------------------------------------
REGIME_RETURNS: dict[str, dict[str, float]] = {
    "Stagflation":    {"Gold": 22,  "Oil": 18,  "SPX": -8,  "TLT": -12, "DXY": 5,  "EM": -10},
    "Overheating":    {"Gold": 8,   "Oil": 15,  "SPX": 14,  "TLT": -5,  "DXY": -3, "EM": 18},
    "Goldilocks":     {"Gold": 3,   "Oil": 5,   "SPX": 18,  "TLT": 6,   "DXY": 0,  "EM": 12},
    "Deflation/Bust": {"Gold": 12,  "Oil": -20, "SPX": -25, "TLT": 20,  "DXY": 8,  "EM": -22},
}


# -----------------------------------------------------------------------------
# RegimeResult
# The output object from classify_regime(). Using a dataclass gives clean
# attribute access (result.regime) rather than key access (result["regime"]).
#
# Fields:
#   regime         — one of the four regime strings above
#   colour         — hex code for UI display
#   growth_score   — how many of the 4 growth signals fired (0–4)
#   inflation_score — how many of the 4 inflation signals fired (0–4)
#   growth_signals  — dict showing each signal's name and whether it fired
#   inflation_signals — same for inflation
# -----------------------------------------------------------------------------
@dataclass
class RegimeResult:
    regime: str
    colour: str
    growth_score: int
    inflation_score: int
    growth_signals: dict = field(default_factory=dict)
    inflation_signals: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# classify_regime() — Layer 1: Growth/Inflation Quadrant
#
# Uses a scoring engine rather than hard thresholds. Each signal votes +1.
# Aggregate the votes, threshold at 2/4. This is deliberately transparent —
# every signal can be explained to a non-technical interviewer in 30 seconds.
#
# Input: signals dict from fetch_macro_inputs() in data/fetcher.py
#
# Required keys:
#   Growth signals:
#     pmi_proxy          — INDPRO-derived PMI equivalent (0–100 scale)
#     pmi_mom_change     — MoM change in PMI proxy (positive = accelerating)
#     claims_wow_change  — WoW change in initial jobless claims (negative = good)
#     spread_10y2y_change — Change in 2s10s spread (positive = curve steepening = growth)
#   Inflation signals:
#     cpi_yoy            — Current CPI year-on-year %
#     cpi_yoy_lag        — CPI YoY from 3 months ago (for acceleration check)
#     ppi_mom            — PPI month-on-month % change
#     breakeven_5y5y     — 5Y5Y forward breakeven inflation rate (market's long-run view)
#     breakeven_5y5y_lag — 5Y5Y breakeven from 3 months ago (for acceleration check)
#     michigan_exp       — University of Michigan 1Y inflation expectations
# -----------------------------------------------------------------------------
def classify_regime(signals: dict[str, float]) -> RegimeResult:
    """Classify the macro regime using a transparent scoring engine (Layer 1)."""

    # -------------------------------------------------------------------------
    # GROWTH SCORING — 4 signals, each votes +1 if positive
    #
    # Signal 1: Is PMI above 50?
    #   PMI above 50 means the manufacturing sector is currently expanding.
    #   This is the most direct growth indicator.
    # -------------------------------------------------------------------------
    g1 = signals["pmi_proxy"] > THRESHOLDS["pmi_expansion"]

    # Signal 2: Is PMI accelerating (rising MoM)?
    #   Even if PMI is below 50, a rising PMI means conditions are improving.
    #   Direction of change matters as much as the level.
    g2 = signals["pmi_mom_change"] > 0

    # Signal 3: Are initial jobless claims falling week-on-week?
    #   Rising claims = people losing jobs = growth slowing.
    #   Falling claims = labour market tightening = growth signal.
    #   ICSA is weekly, so it's one of the fastest leading indicators we have.
    g3 = signals["claims_wow_change"] < 0

    # Signal 4: Is the yield curve steepening?
    #   A steepening curve (spread rising) means markets are pricing in better
    #   growth ahead. A flattening/inverting curve is a recession warning.
    g4 = signals["spread_10y2y_change"] > 0

    growth_score = sum([g1, g2, g3, g4])

    # -------------------------------------------------------------------------
    # INFLATION SCORING — 4 signals, each votes +1 if inflationary
    #
    # Signal 1: Is CPI YoY accelerating?
    #   We compare current CPI YoY to 3 months ago. If it's higher, inflation
    #   is re-accelerating — more dangerous than a stable high level.
    # -------------------------------------------------------------------------
    i1 = signals["cpi_yoy"] > signals["cpi_yoy_lag"]

    # Signal 2: Is PPI (producer prices) rising MoM?
    #   PPI leads CPI by 1–3 months — producers pass cost increases to consumers.
    #   Rising PPI is an early warning that CPI will follow.
    i2 = signals["ppi_mom"] > 0

    # Signal 3: Are 5Y5Y breakevens rising?
    #   The 5Y5Y breakeven is the bond market's view of average inflation 5–10
    #   years from now. Rising breakevens = market losing confidence in the Fed's
    #   ability to control long-run inflation. A critical signal for central banks.
    i3 = signals["breakeven_5y5y"] > signals["breakeven_5y5y_lag"]

    # Signal 4: Are consumer inflation expectations elevated?
    #   From the University of Michigan survey. Above 3% is considered elevated.
    #   Expectations matter because they're self-fulfilling — if people expect
    #   high inflation, they demand higher wages, which causes higher inflation.
    i4 = signals["michigan_exp"] > THRESHOLDS["michigan_hot"]

    inflation_score = sum([i1, i2, i3, i4])

    # -------------------------------------------------------------------------
    # QUADRANT CLASSIFICATION
    #
    # Map the two scores to one of four regimes:
    #
    #                  | Inflation ≥ 2 | Inflation < 2   |
    #  -----------------|---------------|-----------------|
    #  Growth ≥ 2       | Overheating   | Goldilocks      |
    #  Growth < 2       | Stagflation   | Deflation/Bust  |
    #
    # -------------------------------------------------------------------------
    g_threshold = THRESHOLDS["growth_score_min"]
    i_threshold = THRESHOLDS["inflation_score_min"]

    if growth_score >= g_threshold and inflation_score >= i_threshold:
        regime = "Overheating"
    elif growth_score < g_threshold and inflation_score >= i_threshold:
        regime = "Stagflation"
    elif growth_score >= g_threshold and inflation_score < i_threshold:
        regime = "Goldilocks"
    else:
        regime = "Deflation/Bust"

    return RegimeResult(
        regime=regime,
        colour=REGIME_COLOURS[regime],
        growth_score=growth_score,
        inflation_score=inflation_score,
        # Store each signal as name → True/False so the dashboard can show
        # exactly which signals fired and which didn't
        growth_signals={
            "PMI > 50":               g1,
            "PMI accelerating (MoM)": g2,
            "Claims falling (WoW)":   g3,
            "Curve steepening":       g4,
        },
        inflation_signals={
            "CPI YoY accelerating":       i1,
            "PPI rising (MoM)":           i2,
            "Breakevens rising":          i3,
            "Michigan exp > 3%":          i4,
        },
    )


# -----------------------------------------------------------------------------
# Layer 2 stub — Monetary Policy Cycle
# Four stances: early_easing → full_easing → early_tightening → peak_tightening
# The interaction with Layer 1 is critical:
#   Stagflation + peak_tightening (2022) = worst combination — Fed can't ease.
# TODO: implement using FEDFUNDS trend, NFCI, T10Y3M, DGS10 direction
# -----------------------------------------------------------------------------
def classify_monetary_cycle(signals: dict[str, float]) -> str:
    """Stub — monetary policy cycle classification (Layer 2). Not yet implemented."""
    return "unknown"


# -----------------------------------------------------------------------------
# Layer 3 stub — RORO (Risk-on / Risk-off)
# Fast overlay (hours–days). Doesn't change the regime but shows whether
# participants are expressing it through risk or safety assets.
# In risk-off: equities ↓, USD ↑, gold ↑, vol ↑, credit spreads widen.
# TODO: implement using VIX level/change, DXY, Gold/SPY ratio, HYG, EEM vs SPY
# -----------------------------------------------------------------------------
def classify_roro(signals: dict[str, float]) -> str:
    """Stub — risk-on/risk-off overlay (Layer 3). Not yet implemented."""
    return "unknown"
