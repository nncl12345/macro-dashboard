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

import math
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
    "inflation_score_min": 3,     # out of 5 inflation signals (majority-of-5 rule)
    "pmi_expansion":       50.0,  # PMI proxy: above = expanding
    "michigan_hot":        3.0,   # Michigan 1Y expectations above 3% = elevated
}


# -----------------------------------------------------------------------------
# SIGNAL_WEIGHTS — leading vs coincident vote weighting
#
# Leading indicators (surveys, market-priced expectations, forward claims, yield
# curve) turn *before* the economy does; coincident/lagging indicators (CPI,
# INDPRO) confirm what's already happened. Giving leaders 2x weight catches
# regime transitions sooner and downweights the "noise" from
# published-with-a-lag inflation data that dominates the tape after the fact.
#
# The key names here must match the signal dict keys populated in
# classify_regime() below.
# -----------------------------------------------------------------------------
SIGNAL_WEIGHTS: dict[str, int] = {
    # Softer 1.5x ratio: leading=3, coincident=2 (integer-scaled so we don't
    # lose the "n/m" readability in the UI). Previous 2x weighting over-
    # penalised growth in episodes where only 1-2 leading signals fired.
    # Growth axis
    "PMI > 50":                   2,  # coincident (INDPRO is backward-looking)
    "LEI rising (MoM)":           3,  # leading (Conference Board composite)
    "Claims falling (4w MA)":     3,  # leading (weekly, high-frequency)
    "Continuing claims falling":  2,  # coincident (slower-moving stock of unemployed)
    "WEI accelerating":           3,  # leading (NY Fed weekly nowcast)
    "Bear steepener":             3,  # leading (market's forward growth view)
    # Inflation axis
    "CPI YoY accelerating":       2,  # coincident (reports last month's CPI)
    "Core CPI YoY accelerating":  2,  # coincident (same)
    "PPI rising (MoM)":           3,  # leading (1–3m lead on CPI)
    "Breakevens rising":          3,  # leading (market's forward inflation view)
    "Michigan exp > 3%":          3,  # leading (survey of forward expectations)
}

# Max weighted total per axis, assuming every signal fires. Used by the UI
# breakdown and the confidence calculation.
MAX_GROWTH_SIGNALS    = 16  # 2 + 3 + 3 + 2 + 3 + 3  (added Continuing claims +2, WEI +3)
MAX_INFLATION_SIGNALS = 13  # 2 + 2 + 3 + 3 + 3


# -----------------------------------------------------------------------------
# REGIME_COLOURS
# Used consistently across the regime flag, charts, and heatmap.
# -----------------------------------------------------------------------------
REGIME_COLOURS: dict[str, str] = {
    # Warmer, desaturated palette. The vivid red/amber/emerald/blue set looks
    # like every Tailwind-default AI dashboard on the internet; terracotta,
    # ochre, sage, and slate-blue read as considered instead of generated.
    "Stagflation":      "#c9694d",  # terracotta
    "Overheating":      "#c49752",  # ochre / aged gold
    "Goldilocks":       "#7a9b7e",  # sage
    "Deflation/Bust":   "#6b8cae",  # slate blue
    "Insufficient data": "#555a66", # warm grey
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
    # Confidence: how many votes would need to flip to change the regime on
    # either axis. 1 = fragile (boundary call), 2 = moderate, 3+ = strong.
    votes_to_flip: int = 2
    confidence: str = "Moderate"   # "Fragile" | "Moderate" | "Strong"
    # How many signals were actually computable (vs missing due to historical
    # data gaps in backtest mode). In live mode these match MAX_* constants.
    growth_available: int = MAX_GROWTH_SIGNALS
    inflation_available: int = MAX_INFLATION_SIGNALS
    # Thresholds actually used (dynamic — scales with available signals)
    growth_threshold: int = 2
    inflation_threshold: int = 3


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
def _has(signals: dict, *keys: str) -> bool:
    """True only if every key is present AND non-NaN."""
    for k in keys:
        if k not in signals:
            return False
        v = signals[k]
        # NaN check that works without importing numpy/pandas here
        if v is None or (isinstance(v, float) and v != v):
            return False
    return True


def classify_regime(signals: dict[str, float]) -> RegimeResult:
    """Classify the macro regime using a transparent scoring engine (Layer 1).

    Each signal is evaluated only if the underlying data is present. Missing
    signals are skipped (not voted zero) — this lets the backtest run on
    historical dates where some series didn't yet exist (e.g. pre-2003
    breakevens). Thresholds scale to available-signal majorities.
    """

    # -------------------------------------------------------------------------
    # SAHM-RULE HARD OVERRIDE
    # Sahm's recession rule fires at the start of every US recession since 1950
    # with no false positives. When triggered, the quadrant math is irrelevant —
    # the economy is contracting, growth is negative, regime is Deflation/Bust.
    # We still compute scores below for the UI breakdown.
    # -------------------------------------------------------------------------
    sahm_override = bool(signals.get("sahm_trigger", False))

    # -------------------------------------------------------------------------
    # GROWTH SCORING — each signal is (name, required-keys, rule). Rules only
    # run when all required keys are present, so no KeyError if series missing.
    # -------------------------------------------------------------------------
    growth_signals: dict[str, bool] = {}

    # Signal: Is PMI above 50? (most direct growth indicator)
    if _has(signals, "pmi_proxy"):
        growth_signals["PMI > 50"] = signals["pmi_proxy"] > THRESHOLDS["pmi_expansion"]

    # Signal: Is the Conference Board LEI rising MoM?
    # Decorrelated from PMI — a 10-indicator composite of leading indicators.
    if _has(signals, "lei_mom"):
        growth_signals["LEI rising (MoM)"] = signals["lei_mom"] > 0

    # Signal: Are initial jobless claims trending down (4w MA vs prior 4w MA)?
    # Smoothed so we don't get whipsawed by holiday/seasonal noise.
    if _has(signals, "claims_trend_change"):
        growth_signals["Claims falling (4w MA)"] = signals["claims_trend_change"] < 0

    # Signal: Are continuing claims (CCSA) trending down? Same 4w/4w smoothing
    # as initial claims. Together with initial claims this forms a 2-vote
    # labour basket — initial = inflow into unemployment, continuing = stock.
    if _has(signals, "continuing_claims_trend_change"):
        growth_signals["Continuing claims falling"] = signals["continuing_claims_trend_change"] < 0

    # Signal: Is the NY Fed Weekly Economic Index accelerating vs its 4w avg?
    # WEI is a 10-input composite scaled to look like real GDP growth — the
    # cleanest real-time growth nowcast available for free. Weekly cadence
    # complements monthly INDPRO and quarterly LEI.
    if _has(signals, "wei_current", "wei_4w_avg"):
        growth_signals["WEI accelerating"] = signals["wei_current"] > signals["wei_4w_avg"]

    # Signal: BEAR steepener? (spread widening AND 10y leg leading the move)
    # Distinguishes growth-positive bear steepener from recessionary bull steepener.
    if _has(signals, "spread_10y2y_change", "yield_10y_change"):
        growth_signals["Bear steepener"] = (
            signals["spread_10y2y_change"] > 0 and signals["yield_10y_change"] > 0
        )

    # Weighted scoring: each present signal contributes its weight (1 for
    # coincident, 2 for leading). growth_available is the *weighted max*
    # so that `score/available` reads correctly in the UI breakdown.
    growth_signal_count = len(growth_signals)
    growth_available    = sum(SIGNAL_WEIGHTS.get(k, 1) for k in growth_signals)
    growth_score        = sum(SIGNAL_WEIGHTS.get(k, 1) for k, v in growth_signals.items() if v)

    # -------------------------------------------------------------------------
    # INFLATION SCORING — same graceful-skip pattern
    # -------------------------------------------------------------------------
    inflation_signals: dict[str, bool] = {}

    # Signal: Headline CPI YoY accelerating over 3 months
    if _has(signals, "cpi_yoy", "cpi_yoy_lag"):
        inflation_signals["CPI YoY accelerating"] = signals["cpi_yoy"] > signals["cpi_yoy_lag"]

    # Signal: Core CPI YoY accelerating — what the Fed actually reacts to
    if _has(signals, "core_cpi_yoy", "core_cpi_yoy_lag"):
        inflation_signals["Core CPI YoY accelerating"] = (
            signals["core_cpi_yoy"] > signals["core_cpi_yoy_lag"]
        )

    # Signal: PPI rising MoM — leads CPI by 1-3 months
    if _has(signals, "ppi_mom"):
        inflation_signals["PPI rising (MoM)"] = signals["ppi_mom"] > 0

    # Signal: 5Y5Y breakevens rising — market's long-run inflation view
    # Only available post-2003 when TIPS market existed.
    if _has(signals, "breakeven_5y5y", "breakeven_5y5y_lag"):
        inflation_signals["Breakevens rising"] = (
            signals["breakeven_5y5y"] > signals["breakeven_5y5y_lag"]
        )

    # Signal: Michigan 1Y inflation expectations elevated
    # Only available post-1978.
    if _has(signals, "michigan_exp"):
        inflation_signals["Michigan exp > 3%"] = signals["michigan_exp"] > THRESHOLDS["michigan_hot"]

    inflation_signal_count = len(inflation_signals)
    inflation_available    = sum(SIGNAL_WEIGHTS.get(k, 1) for k in inflation_signals)
    inflation_score        = sum(SIGNAL_WEIGHTS.get(k, 1) for k, v in inflation_signals.items() if v)

    # -------------------------------------------------------------------------
    # DISINFLATION OVERRIDE
    # When CPI has rolled over from its 12m peak AND 3m momentum is negative,
    # inflation is directionally falling regardless of what the level-oriented
    # level signals suggest. Zero the inflation score to force a disinflation
    # regime (Goldilocks or Deflation/Bust). Preserves the "rate of change is
    # everything" philosophy — level-hot but rolling-over is not "rising".
    # -------------------------------------------------------------------------
    disinflation_override = bool(
        signals.get("cpi_rolled_over") and signals.get("cpi_3m_decel")
    )
    if disinflation_override:
        inflation_score = 0
        inflation_signals["↓ Disinflation override"] = True

    # -------------------------------------------------------------------------
    # DYNAMIC THRESHOLDS — "plurality-leaning half" of weighted total.
    # Uses floor rather than ceil so the threshold is *just under* 50% of the
    # weighted max. With the 1.5x leading-vs-coincident weighting this keeps
    # the effective bar around 46%, comparable to the original unweighted
    # 50% threshold. Using ceil on a weighted max of 11/13 makes the bar
    # creep up to 55%+ and over-penalises the "up" regimes — verified in
    # the backtest.
    # -------------------------------------------------------------------------
    g_threshold = max(2, math.floor(growth_available / 2))
    i_threshold = max(2, math.floor(inflation_available / 2))

    # Insufficient-data guard — refuse to classify if fewer than 2 signals
    # available on either axis. Uses signal *count* (not weighted) so a single
    # heavy signal doesn't masquerade as two light ones.
    if growth_signal_count < 2 or inflation_signal_count < 2:
        return RegimeResult(
            regime="Insufficient data",
            colour="#4a5568",
            growth_score=growth_score,
            inflation_score=inflation_score,
            growth_signals=growth_signals,
            inflation_signals=inflation_signals,
            votes_to_flip=0,
            confidence="N/A",
            growth_available=growth_available,
            inflation_available=inflation_available,
            growth_threshold=g_threshold,
            inflation_threshold=i_threshold,
        )

    # -------------------------------------------------------------------------
    # QUADRANT CLASSIFICATION — same 2×2 as before, just using dynamic thresholds
    # -------------------------------------------------------------------------
    if sahm_override:
        # Recession detected — bypass scoring and force Deflation/Bust
        regime = "Deflation/Bust"
        growth_signals["⚠ Sahm rule triggered"] = True
    elif growth_score >= g_threshold and inflation_score >= i_threshold:
        regime = "Overheating"
    elif growth_score < g_threshold and inflation_score >= i_threshold:
        regime = "Stagflation"
    elif growth_score >= g_threshold and inflation_score < i_threshold:
        regime = "Goldilocks"
    else:
        regime = "Deflation/Bust"

    # -------------------------------------------------------------------------
    # CONFIDENCE — votes-to-flip on each axis
    #
    # How many signals would need to swap for the regime to change? The minimum
    # across the two axes is the overall fragility of the call.
    #   - score >= threshold  → votes_to_flip = score - threshold + 1
    #   - score <  threshold  → votes_to_flip = threshold - score
    #
    # 1 = Fragile (boundary call, single signal flips the regime)
    # 2 = Moderate
    # 3+ = Strong
    # -------------------------------------------------------------------------
    def _votes_to_flip(score: int, threshold: int) -> int:
        return score - threshold + 1 if score >= threshold else threshold - score

    growth_vtf    = _votes_to_flip(growth_score,    g_threshold)
    inflation_vtf = _votes_to_flip(inflation_score, i_threshold)
    votes_to_flip = min(growth_vtf, inflation_vtf)
    confidence    = "Fragile" if votes_to_flip <= 1 else "Strong" if votes_to_flip >= 3 else "Moderate"

    return RegimeResult(
        regime=regime,
        colour=REGIME_COLOURS[regime],
        growth_score=growth_score,
        inflation_score=inflation_score,
        growth_signals=growth_signals,
        inflation_signals=inflation_signals,
        votes_to_flip=votes_to_flip,
        confidence=confidence,
        growth_available=growth_available,
        inflation_available=inflation_available,
        growth_threshold=g_threshold,
        inflation_threshold=i_threshold,
    )


# -----------------------------------------------------------------------------
# Monetary cycle constants
# -----------------------------------------------------------------------------

CYCLE_COLOURS: dict[str, str] = {
    "Peak Tightening":  "#a04a3a",  # deep brick  — maximum restriction
    "Early Tightening": "#b8693f",  # burnt umber — hiking cycle underway
    "Early Easing":     "#5d7bae",  # dusty blue  — first cuts, relief beginning
    "Full Easing":      "#6a9270",  # sage green  — accommodative, fuel for risk assets
}

# Thresholds for the monetary cycle classifier
CYCLE_THRESHOLDS: dict[str, float] = {
    "hiking_threshold":   0.10,   # Fed Funds 6m change above this = hiking cycle
    "cutting_threshold": -0.10,   # Fed Funds 6m change below this = cutting cycle
    "near_peak_gap":      0.25,   # within 25bp of 12m high = near peak
    "full_easing_gap":    1.00,   # more than 100bp below 12m high = well into cuts
    "nfci_tight":         0.30,   # NFCI above this = tight financial conditions
    "nfci_loose":        -0.30,   # NFCI below this = loose financial conditions
    "high_rate_level":    4.00,   # Fed Funds above this = restrictive in absolute terms
}


@dataclass
class MonetaryCycleResult:
    stance: str          # one of the four cycle labels
    colour: str          # hex code for UI display
    signals: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# classify_monetary_cycle() — Layer 2
#
# Determines where the Fed is in its policy cycle. Logic:
#
#   Step 1: Is the Fed hiking, cutting, or on hold? (6-month Fed Funds change)
#   Step 2: Early or late in that direction? (proximity to 12-month high/low)
#   Step 3: If on hold, classify by absolute level + financial conditions
#
# Why this matters for Layer 1:
#   Same quadrant feels very different depending on the monetary stance.
#   Overheating + Peak Tightening = 2022 (brutal for everything).
#   Goldilocks + Full Easing = 1995, 2019 (best of all worlds).
#
# Required signal keys (from fetch_macro_inputs()):
#   fed_funds_current   — latest effective Fed Funds Rate
#   fed_funds_1m_change — MoM change (non-zero = FOMC just moved)
#   fed_funds_6m_change — 6-month change (direction of cycle)
#   fed_funds_12m_high  — peak rate over the past 12 months
#   nfci                — Chicago Fed financial conditions index
#   real_yield_current  — 10yr TIPS yield today
#   real_yield_3m_ago   — 10yr TIPS yield 3 months ago
# -----------------------------------------------------------------------------
def classify_monetary_cycle(signals: dict[str, float]) -> MonetaryCycleResult:
    """Classify the monetary policy cycle stance (Layer 2)."""

    ff_current    = signals["fed_funds_current"]
    ff_6m_change  = signals["fed_funds_6m_change"]
    ff_12m_high   = signals["fed_funds_12m_high"]
    nfci          = signals["nfci"]

    # --- Tier 1 additions (read with .get so missing data fails gracefully) ---
    # Net liquidity 3m % change — quantity-of-money read. Rising = stealth
    # easing (cash flooding the system) regardless of where Fed Funds sits.
    net_liq_3m_chg = signals.get("net_liq_3m_change_pct", float("nan"))
    # SOFR-IORB spread (bp). Positive >5bp = funding stress = override stance
    # to Peak Tightening regardless of Fed Funds direction.
    sofr_iorb_bp = signals.get("sofr_iorb_spread_bp", float("nan"))
    # MOVE Index — Treasury vol. Above 12m avg = elevated rates uncertainty.
    move_now    = signals.get("move_current", float("nan"))
    move_12m    = signals.get("move_12m_avg", float("nan"))

    def _is_num(x) -> bool:
        return x is not None and isinstance(x, (int, float)) and x == x   # NaN-safe

    move_elevated = _is_num(move_now) and _is_num(move_12m) and move_now > move_12m
    net_liq_expanding   = _is_num(net_liq_3m_chg) and net_liq_3m_chg >  2.0   # >+2% in 3m
    net_liq_contracting = _is_num(net_liq_3m_chg) and net_liq_3m_chg < -2.0   # <−2% in 3m
    funding_stress = _is_num(sofr_iorb_bp) and sofr_iorb_bp > 5.0   # SOFR > IORB by 5bp+

    # --- Step 1: Direction ---
    # Fed Funds 6-month change tells us whether we're in a hiking or cutting cycle.
    # Small moves (< ±0.10%) are treated as "on hold" — the Fed hasn't moved meaningfully.
    hiking  = ff_6m_change >  CYCLE_THRESHOLDS["hiking_threshold"]
    cutting = ff_6m_change <  CYCLE_THRESHOLDS["cutting_threshold"]
    on_hold = not hiking and not cutting

    # --- Step 2: Stage within the direction ---
    # Distance from 12-month high tells us early vs late in the cycle.
    #
    # Hiking cycle:
    #   Current rate ≈ 12m high → still near the peak → Peak Tightening
    #   Current rate well below 12m high → just started hiking → Early Tightening
    #
    # Cutting cycle:
    #   Current rate just below 12m high → first cut(s) just made → Early Easing
    #   Current rate well below 12m high → well into the cutting cycle → Full Easing
    gap_from_peak = ff_12m_high - ff_current   # how far below the peak we are
    near_peak     = gap_from_peak < CYCLE_THRESHOLDS["near_peak_gap"]
    full_easing   = gap_from_peak > CYCLE_THRESHOLDS["full_easing_gap"]

    if hiking and near_peak:
        # Hiking and at/near the top — Fed is at or approaching the terminal rate
        stance = "Peak Tightening"
    elif hiking and not near_peak:
        # Hiking but still well below recent highs — cycle just beginning
        stance = "Early Tightening"
    elif cutting and full_easing:
        # Multiple cuts in — well into the easing cycle
        stance = "Full Easing"
    elif cutting and not full_easing:
        # First cut(s) have landed but rate still close to the peak
        stance = "Early Easing"
    else:
        # On hold — classify by absolute rate level and financial conditions.
        # MOVE elevated counts as tight conditions alongside NFCI (rates-vol
        # stress sits next to broad financial-conditions stress).
        # Net liquidity is a *tilt*: stealth easing/tightening through the
        # back door even while Fed Funds is unchanged.
        tight_conditions = (
            ff_current >= CYCLE_THRESHOLDS["high_rate_level"]
            or nfci >= CYCLE_THRESHOLDS["nfci_tight"]
            or move_elevated
        )
        loose_conditions = (
            ff_current <= 1.0 or nfci <= CYCLE_THRESHOLDS["nfci_loose"]
        )
        if tight_conditions and not net_liq_expanding:
            stance = "Peak Tightening"
        elif loose_conditions or net_liq_expanding:
            # Either Fed dovish OR balance sheet expanding (stealth easing) →
            # treat as Full Easing even if rates haven't moved
            stance = "Full Easing"
        elif net_liq_contracting:
            # Stealth tightening through QT / TGA refill while Fed on hold
            stance = "Early Tightening"
        elif ff_6m_change > 0:
            stance = "Early Tightening"
        else:
            stance = "Early Easing"

    # --- Funding stress hard override (SOFR > IORB) ---
    # When the plumbing breaks, nothing else matters. Force Peak Tightening
    # regardless of where Fed Funds sits — dealer balance sheets are full,
    # Fed will likely intervene within days. Same gating-override pattern
    # as the Sahm rule on Layer 1.
    if funding_stress:
        stance = "Peak Tightening"

    # Real yield direction — supporting context (not used in classification,
    # but stored in signals so the UI can show it in the breakdown)
    real_yield_rising = signals["real_yield_current"] > signals["real_yield_3m_ago"]

    return MonetaryCycleResult(
        stance=stance,
        colour=CYCLE_COLOURS[stance],
        signals={
            "Fed Funds current":    ff_current,
            "6m change":            ff_6m_change,
            "Gap from 12m high":    round(gap_from_peak, 2),
            "NFCI":                 nfci,
            "Real yield rising":    real_yield_rising,
            # Tier 1 monetary additions
            "Net liquidity 3m %":   net_liq_3m_chg if _is_num(net_liq_3m_chg) else "n/a",
            "SOFR-IORB (bp)":       sofr_iorb_bp   if _is_num(sofr_iorb_bp)   else "n/a",
            "MOVE":                 move_now       if _is_num(move_now)       else "n/a",
            "MOVE elevated":        move_elevated,
            "Funding stress":       funding_stress,
        },
    )


# -----------------------------------------------------------------------------
# Layer 3 — RORO (Risk-on / Risk-off)
#
# A fast-moving overlay that sits on top of Layers 1 and 2. It doesn't change
# the regime label but tells you how participants are *expressing* it right now:
# through risk assets (equities, EM, HY credit) or safety assets (USD, gold, vol).
#
# Why this matters:
#   Overheating + Risk-On  → rally still has legs, stay long equities
#   Overheating + Risk-Off → distribution phase, consider reducing exposure
#   Goldilocks + Risk-On   → classic bull; 1995, 2019 felt like this
#   Stagflation + Risk-Off → 2022 scenario — brutal for everything except commodities
#
# 5-signal voting engine (same pattern as Layer 1):
#   Each signal votes +1 for Risk-Off. Total votes → stance:
#     ≥ 3 votes = Risk-Off   (majority of signals screaming caution)
#       2 votes = Neutral     (mixed signals — no clear direction)
#     ≤ 1 vote  = Risk-On    (risk appetite intact)
#
# The 5 signals:
#   r1 — VIX 5-day change positive  (fear index rising → risk-off)
#   r2 — DXY 5-day change positive  (USD strengthening → safe-haven bid)
#   r3 — Gold/SPY ratio 5d rising   (gold outpacing equities → flight to safety)
#   r4 — HY OAS 5-day widening      (credit spreads blowing out → credit stress)
#   r5 — EEM underperforming SPY    (EM selling off relative to US → de-risking)
#
# Required signal keys (from fetch_roro_signals() in data/fetcher.py):
#   vix_5d_change              — VIX point change over last 5 trading days
#   dxy_5d_change              — DXY % change over last 5 trading days
#   gold_spy_ratio_5d_change   — Gold/SPY ratio % change over 5 trading days
#   hyg_5d_change              — HYG ETF % change over 5 trading days
#   eem_vs_spy_5d              — EEM 5d return minus SPY 5d return (relative %)
# -----------------------------------------------------------------------------

RORO_COLOURS: dict[str, str] = {
    "Risk-Off": "#a04a3a",   # deep brick  — caution, safety trade
    "Neutral":  "#8a5e2e",   # bronze      — mixed signals
    "Risk-On":  "#6a9270",   # sage green  — risk appetite intact
}

RORO_THRESHOLDS: dict[str, int] = {
    "risk_off_min": 3,   # ≥3 votes = Risk-Off
    "neutral_min":  2,   # 2 votes  = Neutral
}


@dataclass
class RoroResult:
    stance: str          # "Risk-On", "Neutral", or "Risk-Off"
    colour: str          # hex code for UI display
    score: int           # raw vote count (0–5, higher = more risk-off)
    signals: dict = field(default_factory=dict)  # signal name → numeric value
    votes: dict = field(default_factory=dict)    # signal name → True if voted risk-off


def classify_roro(signals: dict[str, float]) -> RoroResult:
    """Classify the risk-on/risk-off overlay (Layer 3) using a 5-signal voting engine."""

    # Signal 1: VIX rising over the past 5 days?
    # The VIX is the options market's implied volatility for the S&P 500.
    # A rising VIX means investors are buying protection → fear is increasing.
    r1 = signals.get("vix_5d_change", 0) > 0

    # Signal 2: USD (DXY) rising over 5 days?
    # In risk-off episodes, capital flees to the dollar as the world's reserve
    # currency. DXY up = global de-risking underway.
    r2 = signals.get("dxy_5d_change", 0) > 0

    # Signal 3: Gold/SPY ratio rising over 5 days?
    # Gold outperforming equities is the classic risk-off signature.
    # Ratio rising means gold is gaining ground on stocks.
    r3 = signals.get("gold_spy_ratio_5d_change", 0) > 0

    # Signal 4: HY OAS (option-adjusted spread) widening over 5 days?
    # Direct read on credit-market risk appetite — when investors get nervous,
    # they demand more spread to hold junk bonds vs. Treasuries. Replaces HYG
    # price (which embeds duration noise); OAS is the institutional standard.
    # Falls back to HYG-falling proxy if HY OAS data isn't available yet.
    if "hy_oas_5d_change" in signals:
        r4 = signals.get("hy_oas_5d_change", 0) > 0   # widening = risk-off
    else:
        r4 = signals.get("hyg_5d_change", 0) < 0

    # Signal 5: EM equities underperforming US equities over 5 days?
    # Emerging markets are the highest-beta risk assets globally. When risk
    # appetite fades, EM sells off faster than the US — EEM lags SPY.
    r5 = signals.get("eem_vs_spy_5d", 0) < 0

    score = sum([r1, r2, r3, r4, r5])

    if score >= RORO_THRESHOLDS["risk_off_min"]:
        stance = "Risk-Off"
    elif score >= RORO_THRESHOLDS["neutral_min"]:
        stance = "Neutral"
    else:
        stance = "Risk-On"

    # Bundle each signal's numeric value and its vote together so the UI
    # can display both without re-deriving the direction logic.
    # Choose label for credit signal based on which data source is live.
    # HY OAS (bp) is the preferred read; HYG % change is the legacy fallback.
    if "hy_oas_5d_change" in signals:
        credit_label = "HY OAS 5d chg (bp)"
        credit_value = round(signals.get("hy_oas_5d_change", float("nan")), 1)
    else:
        credit_label = "HYG 5d chg %"
        credit_value = round(signals.get("hyg_5d_change", float("nan")), 2)

    _signal_data = {
        "VIX 5d chg (pts)":    (round(signals.get("vix_5d_change", float("nan")), 2),  r1),
        "DXY 5d chg %":        (round(signals.get("dxy_5d_change", float("nan")), 2),  r2),
        "Gold/SPY 5d chg %":   (round(signals.get("gold_spy_ratio_5d_change", float("nan")), 2), r3),
        credit_label:          (credit_value,  r4),
        "EEM vs SPY 5d %":     (round(signals.get("eem_vs_spy_5d", float("nan")), 2),  r5),
    }

    return RoroResult(
        stance=stance,
        colour=RORO_COLOURS[stance],
        score=score,
        signals={k: v[0] for k, v in _signal_data.items()},
        votes={k: v[1] for k, v in _signal_data.items()},
    )
