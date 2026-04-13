# =============================================================================
# regime/classifier.py
#
# The core brain of the dashboard. Takes three macro inputs and outputs one of
# four regime labels. Everything else in the app (colours, charts, heatmap) is
# driven by what this file returns.
# =============================================================================

from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# THRESHOLDS
# These are the decision boundaries for the classifier. Kept in one dict so
# they're easy to find and adjust without digging through logic code.
#
# cpi_mom_hot    — CPI month-on-month % change. Above 0.3% is considered "hot"
#                  inflation (roughly 3.6% annualised). Below = "cool".
# pmi_expansion  — PMI (Purchasing Managers' Index) is a survey of business
#                  activity. 50 is the neutral line: above = economy expanding,
#                  below = contracting. This is the industry-standard threshold.
# spread_2s10s_warn — The 2s10s spread is the 10yr Treasury yield minus the 2yr.
#                  When it goes negative (inverted), it historically signals a
#                  recession is coming. We use 0 as the warning line.
# pmi_borderline — A "grey zone" around PMI 50 (48–52) where the reading is
#                  ambiguous. If PMI is in this band AND the yield curve is
#                  inverted, we treat growth as weak rather than strong.
# -----------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    "cpi_mom_hot":        0.3,
    "pmi_expansion":      50.0,
    "spread_2s10s_warn":  0.0,
    "pmi_borderline_low": 48.0,
    "pmi_borderline_high": 52.0,
}


# -----------------------------------------------------------------------------
# REGIME_COLOURS
# Hex colour codes for each regime — used consistently across the flag, charts,
# and heatmap so the visual language is coherent throughout the dashboard.
# -----------------------------------------------------------------------------
REGIME_COLOURS: dict[str, str] = {
    "Stagflation":        "#E05252",  # red   — worst outcome for most assets
    "Reflation":          "#F5A623",  # amber — hot but growing, mixed signals
    "Goldilocks":         "#4CAF50",  # green — ideal: low inflation, strong growth
    "Deflation/Risk-off": "#5B9BD5",  # blue  — falling prices, recession risk
}


# -----------------------------------------------------------------------------
# REGIME_RETURNS
# Approximate annualised asset returns (%) in each regime, based on historical
# analysis from 1972 to present. These seed the heatmap chart in the dashboard.
#
# Read each row as: "when the economy is in X regime, asset Y has historically
# returned Z% per year on average."
#
# Assets:
#   Gold  — GC=F     — inflation hedge, safe haven
#   Oil   — CL=F     — commodity, growth & supply-driven
#   SPX   — ^GSPC    — US equities (S&P 500)
#   TLT   — TLT      — long-duration US Treasury bonds (20yr+)
#   DXY   — DX-Y.NYB — US Dollar index vs basket of currencies
#   EM    — EEM      — emerging market equities
# -----------------------------------------------------------------------------
REGIME_RETURNS: dict[str, dict[str, float]] = {
    "Stagflation":        {"Gold": 22,  "Oil": 18,  "SPX": -8,  "TLT": -12, "DXY": 5,  "EM": -10},
    "Reflation":          {"Gold": 8,   "Oil": 15,  "SPX": 14,  "TLT": -5,  "DXY": -3, "EM": 18},
    "Goldilocks":         {"Gold": 3,   "Oil": 5,   "SPX": 18,  "TLT": 6,   "DXY": 0,  "EM": 12},
    "Deflation/Risk-off": {"Gold": 12,  "Oil": -20, "SPX": -25, "TLT": 20,  "DXY": 8,  "EM": -22},
}


# -----------------------------------------------------------------------------
# RegimeResult
# A dataclass — basically a lightweight container that holds the classifier's
# output. Using a dataclass instead of a plain dict means you get attribute
# access (result.regime) rather than key access (result["regime"]), which is
# cleaner when the app reads these values.
#
# Fields:
#   regime    — one of the four regime strings above
#   colour    — hex code for that regime (looked up from REGIME_COLOURS)
#   inflation — human-readable label: "Hot" or "Cool"
#   growth    — human-readable label: "Strong" or "Weak"
#   signals   — the raw input numbers, stored so the dashboard can display them
# -----------------------------------------------------------------------------
@dataclass
class RegimeResult:
    regime: str
    colour: str
    inflation: str   # "Hot" | "Cool"
    growth: str      # "Strong" | "Weak"
    signals: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# classify_regime()
# The main function. Takes three float inputs, runs them through the decision
# logic below, and returns a RegimeResult.
#
# Inputs:
#   cpi_mom      — CPI month-on-month % change (e.g. 0.4 means +0.4% that month)
#   pmi          — PMI reading (e.g. 51.2)
#   spread_2s10s — 10yr yield minus 2yr yield in % (e.g. -0.3 means inverted)
# -----------------------------------------------------------------------------
def classify_regime(
    cpi_mom: float,
    pmi: float,
    spread_2s10s: float,
) -> RegimeResult:
    """Classify the macro regime from CPI momentum, PMI, and the 2s10s yield spread."""

    # --- Step 1: Inflation signal ---
    # Simple threshold: is CPI rising faster than 0.3% this month?
    inflation_hot = cpi_mom > THRESHOLDS["cpi_mom_hot"]

    # --- Step 2: Growth signal ---
    # Primary: is PMI above 50 (expanding)?
    pmi_expanding = pmi >= THRESHOLDS["pmi_expansion"]

    # Supporting: is the yield curve inverted (2yr yield > 10yr yield)?
    # An inverted curve means bond markets are pricing in rate cuts ahead —
    # i.e. they expect the economy to slow or tip into recession.
    curve_inverted = spread_2s10s < THRESHOLDS["spread_2s10s_warn"]

    # If PMI is in the grey zone (48–52) it could go either way.
    # In that case, an inverted curve is the tiebreaker — we treat growth as weak.
    # Outside the grey zone, PMI alone decides.
    borderline = THRESHOLDS["pmi_borderline_low"] <= pmi < THRESHOLDS["pmi_borderline_high"]
    growth_strong = pmi_expanding and not (borderline and curve_inverted)

    # --- Step 3: Map to regime quadrant ---
    #
    #              | Inflation Hot | Inflation Cool  |
    #  ------------|---------------|-----------------|
    #  Growth Weak | Stagflation   | Deflation/Risk  |
    #  Growth Strong| Reflation    | Goldilocks      |
    #
    if inflation_hot and not growth_strong:
        regime = "Stagflation"
    elif inflation_hot and growth_strong:
        regime = "Reflation"
    elif not inflation_hot and growth_strong:
        regime = "Goldilocks"
    else:
        regime = "Deflation/Risk-off"

    # --- Step 4: Return a RegimeResult with everything the dashboard needs ---
    return RegimeResult(
        regime=regime,
        colour=REGIME_COLOURS[regime],
        inflation="Hot" if inflation_hot else "Cool",
        growth="Strong" if growth_strong else "Weak",
        # Store raw inputs so the UI can show "what drove this classification"
        signals={
            "cpi_mom": cpi_mom,
            "pmi": pmi,
            "spread_2s10s": spread_2s10s,
            "curve_inverted": curve_inverted,
        },
    )
