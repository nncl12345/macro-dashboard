from dataclasses import dataclass, field

# Thresholds are intentionally simple — explainable in 30 seconds to a non-technical interviewer
THRESHOLDS: dict[str, float] = {
    "cpi_mom_hot": 0.3,    # CPI MoM % — above = hot inflation
    "pmi_expansion": 50.0,  # PMI — above = expanding economy
    "spread_2s10s_warn": 0.0,  # 2s10s — below = inverted curve
    "pmi_borderline_low": 48.0,  # PMI band where curve inversion can override
    "pmi_borderline_high": 52.0,
}

REGIME_COLOURS: dict[str, str] = {
    "Stagflation":        "#E05252",  # red
    "Reflation":          "#F5A623",  # amber
    "Goldilocks":         "#4CAF50",  # green
    "Deflation/Risk-off": "#5B9BD5",  # blue
}

# Seed returns (annualised %) per regime — based on historical analysis 1972–present
REGIME_RETURNS: dict[str, dict[str, float]] = {
    "Stagflation":        {"Gold": 22,  "Oil": 18,  "SPX": -8,  "TLT": -12, "DXY": 5,  "EM": -10},
    "Reflation":          {"Gold": 8,   "Oil": 15,  "SPX": 14,  "TLT": -5,  "DXY": -3, "EM": 18},
    "Goldilocks":         {"Gold": 3,   "Oil": 5,   "SPX": 18,  "TLT": 6,   "DXY": 0,  "EM": 12},
    "Deflation/Risk-off": {"Gold": 12,  "Oil": -20, "SPX": -25, "TLT": 20,  "DXY": 8,  "EM": -22},
}


@dataclass
class RegimeResult:
    regime: str
    colour: str
    inflation: str   # "Hot" | "Cool"
    growth: str      # "Strong" | "Weak"
    signals: dict = field(default_factory=dict)


def classify_regime(
    cpi_mom: float,
    pmi: float,
    spread_2s10s: float,
) -> RegimeResult:
    """Classify the macro regime from CPI momentum, PMI, and the 2s10s yield spread."""
    inflation_hot = cpi_mom > THRESHOLDS["cpi_mom_hot"]

    # PMI is the primary growth signal.
    # An inverted yield curve downgrades borderline PMI readings (48–52) to weak.
    pmi_expanding = pmi >= THRESHOLDS["pmi_expansion"]
    curve_inverted = spread_2s10s < THRESHOLDS["spread_2s10s_warn"]
    borderline = THRESHOLDS["pmi_borderline_low"] <= pmi < THRESHOLDS["pmi_borderline_high"]
    growth_strong = pmi_expanding and not (borderline and curve_inverted)

    if inflation_hot and not growth_strong:
        regime = "Stagflation"
    elif inflation_hot and growth_strong:
        regime = "Reflation"
    elif not inflation_hot and growth_strong:
        regime = "Goldilocks"
    else:
        regime = "Deflation/Risk-off"

    return RegimeResult(
        regime=regime,
        colour=REGIME_COLOURS[regime],
        inflation="Hot" if inflation_hot else "Cool",
        growth="Strong" if growth_strong else "Weak",
        signals={
            "cpi_mom": cpi_mom,
            "pmi": pmi,
            "spread_2s10s": spread_2s10s,
            "curve_inverted": curve_inverted,
        },
    )
