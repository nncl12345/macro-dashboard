# =============================================================================
# app.py
#
# Main Streamlit application. This file is intentionally thin — it only
# handles layout and wiring. All business logic lives in:
#   data/fetcher.py       — pulls live data from FRED and yfinance
#   regime/classifier.py  — scores signals and returns the current regime
#   charts/plots.py       — builds Plotly figures from the data
#
# Layout (four rows, matching CLAUDE.md spec):
#   Row 1: Regime flag + 4 KPI metric cards
#   Row 2: Yield curve chart | CPI trend chart
#   Row 3: Asset return heatmap
#   Row 4: Market snapshot table
# =============================================================================

import streamlit as st

from charts.plots import (
    plot_cpi_trend,
    plot_episode_table,
    plot_market_snapshot,
    plot_regime_heatmap,
    plot_yield_curve,
)
from data.fetcher import (
    compute_episode_returns,
    compute_regime_returns,
    fetch_cpi_trend,
    fetch_historical_panel,
    fetch_kpi_data,
    fetch_macro_inputs,
    fetch_market_snapshot,
    fetch_regime_price_history,
    fetch_roro_signals,
    fetch_yield_curve,
)
from regime.backtest import backtest_episodes, hit_rate_summary
from regime.classifier import REGIME_COLOURS, REGIME_RETURNS, classify_regime, classify_monetary_cycle, classify_roro

# -----------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the script.
# Sets the browser tab title, favicon, and default layout width.
# "wide" layout uses the full browser width, better for a data dashboard.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom CSS — Bloomberg-style dark terminal theme
#
# config.toml sets the base dark palette; these overrides fine-tune elements
# that Streamlit's theme system doesn't expose (expanders, plotly wrappers,
# scrollbars, fonts). JetBrains Mono is used for numeric values — tabular
# figures prevent layout jitter as numbers update.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ── Base ── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #080d19;
    }
    [data-testid="stHeader"] {
        background-color: #080d19 !important;
        border-bottom: 1px solid #1a2a3a;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        background-color: #080d19;
    }

    /* ── Dividers ── */
    [data-testid="stDivider"] hr, hr {
        border-color: #1a2a3a !important;
        margin: 0.6rem 0;
    }

    /* ── Plotly chart wrappers ── */
    [data-testid="stPlotlyChart"] {
        border-radius: 8px;
        border: 1px solid #1a2a3a;
        overflow: hidden;
        padding: 0;
        background-color: #0e1726;
    }

    /* ── Expander (signal breakdown footer) ── */
    [data-testid="stExpander"] {
        background-color: #0e1726 !important;
        border: 1px solid #1a2a3a !important;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary {
        color: #8899aa !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #e2e8f0 !important;
    }
    /* Text inside the expander */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #8899aa;
        font-size: 0.82rem;
        font-family: 'Inter', sans-serif;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div {
        color: #60a5fa !important;
    }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar              { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track        { background: #080d19; }
    ::-webkit-scrollbar-thumb        { background: #1a2a3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover  { background: #243b53; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# All fetch functions are cached with @st.cache_data(ttl=3600) in fetcher.py,
# so these calls are fast after the first load — they return cached results
# until the 1-hour TTL expires.
# =============================================================================

# Show a spinner while data loads — disappears once everything is ready
with st.spinner("Loading live macro data…"):
    macro_signals  = fetch_macro_inputs()
    kpi            = fetch_kpi_data()
    curve_now, curve_ago = fetch_yield_curve()
    cpi_df         = fetch_cpi_trend()
    snapshot_df    = fetch_market_snapshot()
    roro_signals   = fetch_roro_signals()
    # Historical price data for the regime heatmap — fetched once, cached 1hr.
    # compute_regime_returns() is pure compute (no cache) so it runs on each load
    # but is fast — it only slices the already-fetched DataFrame.
    regime_prices  = fetch_regime_price_history()
    regime_returns, episode_counts = compute_regime_returns(regime_prices, fallback=REGIME_RETURNS)
    episode_rows   = compute_episode_returns(regime_prices, fallback=REGIME_RETURNS)

# Run all three classifiers on the live signals
result        = classify_regime(macro_signals)
cycle_result  = classify_monetary_cycle(macro_signals)
roro_result   = classify_roro(roro_signals)

# Colour used throughout this render for the current regime
regime_colour = REGIME_COLOURS[result.regime]


# =============================================================================
# HELPERS — reusable UI components
# =============================================================================

# -----------------------------------------------------------------------------
# KPI card helper — returns a styled HTML card with a colour-coded background.
# status:  "good" (green) | "bad" (red) | "warn" (amber) | "neutral" (grey)
# Using HTML instead of st.metric gives us full control over the tint colour.
# -----------------------------------------------------------------------------
def _kpi_card(label: str, value: str, delta: str = "", status: str = "neutral") -> str:
    """Return an HTML string for a dark-theme colour-coded KPI card.

    status colours follow Bloomberg convention: green = healthy, red = risk,
    amber = warning/elevated, neutral = informational.
    """
    # Dark card backgrounds — subtle tint, not screaming neon
    bg     = {"good": "#0d2818", "bad": "#2d0a0a", "warn": "#1f1500", "neutral": "#0e1726"}[status]
    border = {"good": "#34d39960", "bad": "#f8717160", "warn": "#fbbf2460", "neutral": "#1a2a3a"}[status]
    # Value uses the status colour so the number itself carries the signal
    val_c  = {"good": "#34d399", "bad": "#f87171", "warn": "#fbbf24", "neutral": "#e2e8f0"}[status]
    delta_html = (
        f'<div style="font-size:0.69rem; color:#8899aa; margin-top:0.25rem;'
        f' font-family:Inter,sans-serif;">{delta}</div>'
        if delta else ""
    )
    return (
        f'<div style="background:{bg}; border-radius:8px; border:1px solid {border};'
        f' padding:0.75rem 1rem; height:100%;">'
        f'<div style="font-size:0.65rem; color:#8899aa; font-weight:600; letter-spacing:0.09em;'
        f' text-transform:uppercase; margin-bottom:0.3rem; font-family:Inter,sans-serif;">{label}</div>'
        f'<div style="font-size:1.45rem; font-weight:700; color:{val_c};'
        f' font-family:\'JetBrains Mono\',monospace; letter-spacing:-0.01em;">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def _signal_values(signals: dict) -> tuple[dict, dict]:
    """Return (growth_values, inflation_values) dicts mapping signal name → driving number.

    For boolean threshold signals (e.g. PMI > 50), return the raw value being compared.
    For acceleration signals (e.g. CPI accelerating), return the delta (current − lag)
    because the sign of the delta is what the classifier actually checks.
    """
    growth = {
        "PMI > 50":               signals["pmi_proxy"],
        "LEI rising (MoM)":       signals["lei_mom"],
        "Claims falling (4w MA)": signals["claims_trend_change"],
        # Bear steepener = spread widened AND 10y rose; display the 10y move
        # as the most informative single number behind the vote.
        "Bear steepener":         signals["yield_10y_change"],
    }
    inflation = {
        "CPI YoY accelerating":      signals["cpi_yoy"] - signals["cpi_yoy_lag"],
        "Core CPI YoY accelerating": signals["core_cpi_yoy"] - signals["core_cpi_yoy_lag"],
        "PPI rising (MoM)":          signals["ppi_mom"],
        "Breakevens rising":         signals["breakeven_5y5y"] - signals["breakeven_5y5y_lag"],
        "Michigan exp > 3%":         signals["michigan_exp"],
    }
    return growth, inflation


def _fmt_signal_val(signal_name: str, val: float) -> str:
    """Format a signal value for display next to its name.

    PMI proxy and Michigan expectations are absolute levels — no sign prefix.
    Claims trend change is in raw count — format with thousand separators.
    Everything else is a rate of change where direction matters, so use +/−.
    """
    absolute_signals = {"PMI > 50", "Michigan exp > 3%"}
    if signal_name in absolute_signals:
        return f"{val:.1f}"
    if signal_name == "Claims falling (4w MA)":
        return f"{val:+,.0f}"
    return f"{val:+.2f}"


def _confidence_badge_html(confidence: str, votes_to_flip: int) -> str:
    """Small pill showing how fragile the regime call is (votes-to-flip)."""
    colour = {
        "Fragile":  "#ff5757",   # red — boundary call, single signal could flip it
        "Moderate": "#fbbf24",   # amber — some buffer, but not robust
        "Strong":   "#34d399",   # green — firm majority on both axes
    }[confidence]
    plural = "s" if votes_to_flip != 1 else ""
    return (
        f'<div style="margin-top:0.55rem;">'
        f'<span style="font-size:0.6rem; font-weight:700; letter-spacing:0.1em;'
        f' color:#4a5568; text-transform:uppercase; font-family:Inter,sans-serif;">'
        f'Confidence&nbsp;</span>'
        f'<span style="font-size:0.72rem; font-weight:600; padding:0.14rem 0.44rem;'
        f' border-radius:4px; background-color:{colour}18; color:{colour};'
        f' border:1px solid {colour}50; font-family:\'JetBrains Mono\',monospace;">'
        f'{confidence} · {votes_to_flip} vote{plural} to flip</span>'
        f'</div>'
    )


def _regime_flag_html(
    result,
    cycle_result,
    roro_result,
    regime_colour: str,
) -> str:
    """Return the HTML string for the regime flag card (regime + badges + score tally)."""
    cycle_colour = cycle_result.colour
    roro_colour  = roro_result.colour
    return (
        f'<div style="padding:1rem 1.1rem; border-radius:8px;'
        f' border:1px solid {regime_colour}40; background:#0e1726;'
        f' display:inline-block; min-width:270px;">'

        f'<div style="font-size:0.62rem; font-weight:700; letter-spacing:0.12em;'
        f' color:#4a5568; text-transform:uppercase; margin-bottom:0.45rem;'
        f' font-family:Inter,sans-serif;">Macro Regime</div>'

        f'<div style="font-size:1.65rem; font-weight:800; letter-spacing:0.04em;'
        f' padding:0.28rem 0.8rem; border-radius:6px; display:inline-block;'
        f' background-color:{regime_colour}18; color:{regime_colour};'
        f' border:1.5px solid {regime_colour}60; margin-bottom:0.65rem;'
        f' font-family:\'JetBrains Mono\',monospace;">'
        f'{result.regime.upper()}</div>'

        f'<div style="display:flex; gap:0.55rem; align-items:flex-start;">'

        f'<div>'
        f'<div style="font-size:0.6rem; font-weight:700; letter-spacing:0.1em;'
        f' color:#4a5568; text-transform:uppercase; margin-bottom:0.18rem;'
        f' font-family:Inter,sans-serif;">Fed Cycle</div>'
        f'<div style="font-size:0.76rem; font-weight:600; padding:0.18rem 0.48rem;'
        f' border-radius:4px; background-color:{cycle_colour}18; color:{cycle_colour};'
        f' border:1px solid {cycle_colour}50; white-space:nowrap;'
        f' font-family:\'JetBrains Mono\',monospace;">'
        f'{cycle_result.stance}</div>'
        f'</div>'

        f'<div>'
        f'<div style="font-size:0.6rem; font-weight:700; letter-spacing:0.1em;'
        f' color:#4a5568; text-transform:uppercase; margin-bottom:0.18rem;'
        f' font-family:Inter,sans-serif;">Sentiment</div>'
        f'<div style="font-size:0.76rem; font-weight:600; padding:0.18rem 0.48rem;'
        f' border-radius:4px; background-color:{roro_colour}18; color:{roro_colour};'
        f' border:1px solid {roro_colour}50; white-space:nowrap;'
        f' font-family:\'JetBrains Mono\',monospace;">'
        f'{roro_result.stance} {roro_result.score}/5</div>'
        f'</div>'

        f'</div>'

        # Confidence colour: Fragile = red, Moderate = amber, Strong = green.
        # A Fragile call means the regime would flip if a single signal changed.
        f'{_confidence_badge_html(result.confidence, result.votes_to_flip)}'

        f'<div style="font-size:0.68rem; color:#4a5568; margin-top:0.55rem;'
        f' font-family:Inter,sans-serif;">'
        f'Growth {result.growth_score}/{result.growth_available} (thr&nbsp;{result.growth_threshold})'
        f' &nbsp;&middot;&nbsp; '
        f'Inflation {result.inflation_score}/{result.inflation_available} (thr&nbsp;{result.inflation_threshold})'
        f'</div>'

        f'</div>'
    )


# =============================================================================
# ROW 1 — Regime flag + KPI cards
# =============================================================================

flag_col, kpi1, kpi2, kpi3, kpi4 = st.columns([2, 1, 1, 1, 1])

with flag_col:
    st.markdown(
        _regime_flag_html(result, cycle_result, roro_result, regime_colour),
        unsafe_allow_html=True,
    )

with kpi1:
    # CPI above 2% target = bad (red), below = good (green)
    cpi_status = "bad" if kpi['cpi_yoy'] > 2.0 else "good"
    st.markdown(_kpi_card(
        label="CPI YoY",
        value=f"{kpi['cpi_yoy']:.2f}%",
        delta=f"{kpi['cpi_yoy'] - 2.0:+.2f}% vs 2% target",
        status=cpi_status,
    ), unsafe_allow_html=True)

with kpi2:
    # Fed Funds above 4% = restrictive (warn), below 2% = accommodative (good)
    ff = kpi['fed_funds']
    ff_status = "warn" if ff >= 4.0 else "good" if ff < 2.0 else "neutral"
    ff_delta = "Restrictive" if ff >= 4.0 else "Accommodative" if ff < 2.0 else "Neutral"
    st.markdown(_kpi_card(
        label="Fed Funds",
        value=f"{ff:.2f}%",
        delta=ff_delta,
        status=ff_status,
    ), unsafe_allow_html=True)

with kpi3:
    # 2s10s Spread — negative (inverted) = bad, positive = good
    spread = kpi['spread_2s10s']
    spread_status = "bad" if spread < 0 else "good"
    spread_delta = "Inverted — recession signal" if spread < 0 else "Normal"
    st.markdown(_kpi_card(
        label="2s10s Spread",
        value=f"{spread:.2f}%",
        delta=spread_delta,
        status=spread_status,
    ), unsafe_allow_html=True)

with kpi4:
    # 10yr Real Yield — negative = financial repression (warn), positive = good
    real_yield = kpi['real_yield_10y']
    ry_status = "warn" if real_yield < 0 else "good"
    ry_delta = "Financial repression" if real_yield < 0 else "Positive real return"
    st.markdown(_kpi_card(
        label="10yr Real Yield",
        value=f"{real_yield:.2f}%",
        delta=ry_delta,
        status=ry_status,
    ), unsafe_allow_html=True)

st.divider()


# =============================================================================
# ROW 2 — Yield curve (left) | CPI trend (right)
# Two side-by-side charts using a 1:1 column split
# =============================================================================

left_col, right_col = st.columns(2)

with left_col:
    # Yield curve: current vs 1yr ago.
    # The shape tells you a lot: normal (upward slope) = healthy growth priced in,
    # flat = uncertainty, inverted = recession warning.
    st.plotly_chart(
        plot_yield_curve(curve_now, curve_ago),
        use_container_width=True,
        theme=None,   # don't let Streamlit override paper_bgcolor/plot_bgcolor
    )

with right_col:
    # CPI and Core CPI over 24 months with the Fed 2% target as a dashed line.
    # Lets you see at a glance whether inflation is converging to or diverging
    # from target.
    st.plotly_chart(
        plot_cpi_trend(cpi_df),
        use_container_width=True,
        theme=None,
    )

st.divider()


# =============================================================================
# ROW 3 — Regime heatmap (full width)
# Shows average annualised returns for Gold, Oil, SPX, TLT, DXY, EM across
# all four regimes. Current regime is highlighted with a coloured border.
# =============================================================================

st.plotly_chart(
    plot_regime_heatmap(result.regime, regime_returns=regime_returns, episode_counts=episode_counts),
    use_container_width=True,
    theme=None,
)

st.divider()


# =============================================================================
# ROW 3b — Episode returns table (full width)
# The heatmap shows regime averages; this table shows the named events behind
# them so you can see exactly what happened during each historical episode.
# =============================================================================

st.plotly_chart(
    plot_episode_table(episode_rows),
    use_container_width=True,
    theme=None,
)

st.divider()


# =============================================================================
# ROW 4 — Market snapshot table (full width)
# Live prices + 1D/1W/1M/YTD % changes, green/red coloured cells
# =============================================================================

st.plotly_chart(
    plot_market_snapshot(snapshot_df),
    use_container_width=True,
    theme=None,
)


# =============================================================================
# FOOTER — signal breakdown expander
# Collapsed by default so it doesn't clutter the view, but available for
# anyone who wants to see exactly which signals drove the regime call.
# This is the "show your working" section — important for CV credibility.
# =============================================================================

with st.expander("📋  Signal breakdown — what drove this regime classification?"):
    g_col, i_col, m_col, r_col = st.columns(4)

    live_g_vals, live_i_vals = _signal_values(macro_signals)

    with g_col:
        st.markdown(f"**Layer 1 — Growth: {result.growth_score}/{result.growth_available} (thr {result.growth_threshold})**")
        for signal, fired in result.growth_signals.items():
            icon = "✅" if fired else "❌"
            val_str = _fmt_signal_val(signal, live_g_vals[signal])
            st.markdown(f"{icon} {signal} `{val_str}`")

    with i_col:
        st.markdown(f"**Layer 1 — Inflation: {result.inflation_score}/{result.inflation_available} (thr {result.inflation_threshold})**")
        for signal, fired in result.inflation_signals.items():
            icon = "✅" if fired else "❌"
            val_str = _fmt_signal_val(signal, live_i_vals[signal])
            st.markdown(f"{icon} {signal} `{val_str}`")

    with m_col:
        st.markdown(f"**Layer 2 — Monetary Cycle: {cycle_result.stance}**")
        for key, val in cycle_result.signals.items():
            if isinstance(val, bool):
                icon = "✅" if val else "❌"
                st.markdown(f"{icon} {key}")
            else:
                st.markdown(f"• {key}: **{val}**")

    with r_col:
        st.markdown(f"**Layer 3 — RORO: {roro_result.stance} ({roro_result.score}/5)**")
        for key, val in roro_result.signals.items():
            is_risk_off = roro_result.votes[key]
            icon  = "🔴" if is_risk_off else "🟢"
            label = "Risk-Off" if is_risk_off else "Risk-On"
            # Format with explicit + sign so direction is unambiguous at a glance
            val_str = f"{val:+.2f}" if val == val else "n/a"  # nan check
            st.markdown(f"{icon} **{key}**: {val_str} — {label}")


# =============================================================================
# SCENARIO BUILDER
# Lets you override any input signal and see the resulting regime in real time.
# Pre-populated from live data — tweak a slider and the flag updates instantly.
#
# For acceleration-check inputs (CPI lag, breakeven lag, real yield lag) we
# expose a "3-month change" slider and derive: lag = current − delta.
# This maps to how analysts think: "CPI is X% and has risen Y pp in 3 months."
# =============================================================================

st.divider()
st.markdown(
    '<div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; letter-spacing:0.04em;'
    ' font-family:Inter,sans-serif; margin-bottom:0.1rem;">🔮  Scenario Builder</div>'
    '<div style="font-size:0.78rem; color:#4a5568; font-family:Inter,sans-serif;'
    ' margin-bottom:1.2rem;">Adjust any signal below — the regime flags update instantly.</div>',
    unsafe_allow_html=True,
)

# --- Input columns (three layers side by side) ---
l1g_col, l1i_col, l2_col, l3_col = st.columns(4)

with l1g_col:
    st.markdown(
        '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
        ' text-transform:uppercase; margin-bottom:0.6rem; font-family:Inter,sans-serif;">'
        'Layer 1 — Growth</div>',
        unsafe_allow_html=True,
    )
    wi_pmi = st.slider(
        "PMI proxy (>50 = expanding)",
        min_value=20.0, max_value=75.0,
        value=float(macro_signals["pmi_proxy"]),
        step=0.5,
        help="INDPRO-derived PMI equivalent. Above 50 = manufacturing expanding.",
    )
    wi_lei_mom = st.slider(
        "LEI MoM change (%)",
        min_value=-2.0, max_value=2.0,
        value=float(macro_signals["lei_mom"]),
        step=0.05,
        format="%.2f",
        help="Conference Board Leading Economic Index month-on-month % change. Positive = leading indicators signal growth ahead.",
    )
    wi_claims_trend = st.slider(
        "Initial claims — 4w MA change",
        min_value=-100_000, max_value=100_000,
        value=int(macro_signals["claims_trend_change"]),
        step=1_000,
        help="Change in the 4-week moving average of initial jobless claims. Negative = labour market tightening (smoothed, unlike raw WoW).",
    )
    wi_10y_chg = st.slider(
        "10y yield change (%)",
        min_value=-1.0, max_value=1.0,
        value=float(macro_signals["yield_10y_change"]),
        step=0.01,
        format="%.2f",
        help="Change in 10y Treasury yield over the past month. Bear steepener requires this to be positive.",
    )
    wi_2y_chg = st.slider(
        "2y yield change (%)",
        min_value=-1.0, max_value=1.0,
        value=float(macro_signals["yield_2y_change"]),
        step=0.01,
        format="%.2f",
        help="Change in 2y Treasury yield over the past month. Bear steepener requires the 10y to rise by MORE than the 2y.",
    )

with l1i_col:
    st.markdown(
        '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
        ' text-transform:uppercase; margin-bottom:0.6rem; font-family:Inter,sans-serif;">'
        'Layer 1 — Inflation</div>',
        unsafe_allow_html=True,
    )
    wi_cpi = st.slider(
        "CPI YoY (%)",
        min_value=0.0, max_value=12.0,
        value=float(macro_signals["cpi_yoy"]),
        step=0.1,
        format="%.1f",
        help="Current CPI year-on-year % change.",
    )
    # Expose the 3m acceleration rather than the raw lag — easier to reason about
    wi_cpi_3m_delta = st.slider(
        "CPI YoY — 3m change (pp)",
        min_value=-3.0, max_value=3.0,
        value=round(float(macro_signals["cpi_yoy"]) - float(macro_signals["cpi_yoy_lag"]), 2),
        step=0.05,
        format="%.2f",
        help="How much CPI YoY has moved over the past 3 months. Positive = re-accelerating.",
    )
    wi_core_cpi = st.slider(
        "Core CPI YoY (%)",
        min_value=0.0, max_value=10.0,
        value=float(macro_signals["core_cpi_yoy"]),
        step=0.1,
        format="%.1f",
        help="Core CPI (ex food & energy) YoY. Cleaner gauge of underlying inflation — what the Fed actually reacts to.",
    )
    wi_core_cpi_3m_delta = st.slider(
        "Core CPI YoY — 3m change (pp)",
        min_value=-2.0, max_value=2.0,
        value=round(float(macro_signals["core_cpi_yoy"]) - float(macro_signals["core_cpi_yoy_lag"]), 2),
        step=0.05,
        format="%.2f",
        help="How much core CPI YoY has moved in 3 months. Positive = sticky inflation re-accelerating.",
    )
    wi_ppi_mom = st.slider(
        "PPI MoM (%)",
        min_value=-5.0, max_value=5.0,
        value=float(macro_signals["ppi_mom"]),
        step=0.1,
        format="%.1f",
        help="Producer Price Index month-on-month % change. PPI leads CPI by 1–3 months.",
    )
    wi_breakeven = st.slider(
        "5Y5Y breakeven (%)",
        min_value=1.0, max_value=5.0,
        value=float(macro_signals["breakeven_5y5y"]),
        step=0.05,
        format="%.2f",
        help="Market's long-run inflation expectation (5–10yr forward). Rising = losing confidence in the Fed.",
    )
    wi_breakeven_3m_delta = st.slider(
        "Breakeven — 3m change (pp)",
        min_value=-1.0, max_value=1.0,
        value=round(float(macro_signals["breakeven_5y5y"]) - float(macro_signals["breakeven_5y5y_lag"]), 2),
        step=0.02,
        format="%.2f",
        help="How much the 5Y5Y breakeven has moved in 3 months. Positive = market pricing in more inflation.",
    )
    wi_michigan = st.slider(
        "Michigan 1Y inflation exp (%)",
        min_value=1.0, max_value=7.0,
        value=float(macro_signals["michigan_exp"]),
        step=0.1,
        format="%.1f",
        help="University of Michigan 1-year inflation expectations survey. Above 3% = elevated.",
    )

with l2_col:
    st.markdown(
        '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
        ' text-transform:uppercase; margin-bottom:0.6rem; font-family:Inter,sans-serif;">'
        'Layer 2 — Monetary</div>',
        unsafe_allow_html=True,
    )
    wi_ff_current = st.slider(
        "Fed Funds rate (%)",
        min_value=0.0, max_value=10.0,
        value=float(macro_signals["fed_funds_current"]),
        step=0.25,
        format="%.2f",
        help="Current effective Federal Funds Rate.",
    )
    wi_ff_6m_chg = st.slider(
        "Fed Funds 6m change (%)",
        min_value=-3.0, max_value=3.0,
        value=float(macro_signals["fed_funds_6m_change"]),
        step=0.25,
        format="%.2f",
        help="6-month change in Fed Funds. Positive = hiking cycle, negative = cutting cycle.",
    )
    wi_ff_12m_high = st.slider(
        "Fed Funds 12m peak (%)",
        min_value=0.0, max_value=10.0,
        value=float(macro_signals["fed_funds_12m_high"]),
        step=0.25,
        format="%.2f",
        help="The highest Fed Funds rate over the past 12 months. Used to detect early/late cycle stage.",
    )
    wi_nfci = st.slider(
        "NFCI (Chicago Fed)",
        min_value=-1.0, max_value=2.0,
        value=float(macro_signals["nfci"]),
        step=0.05,
        format="%.2f",
        help="National Financial Conditions Index. Negative = loose (easy credit), positive = tight.",
    )
    wi_real_yield = st.slider(
        "10yr real yield (%)",
        min_value=-2.0, max_value=4.0,
        value=float(macro_signals["real_yield_current"]),
        step=0.05,
        format="%.2f",
        help="10-year TIPS yield (inflation-adjusted). Rising real yields = financial conditions tightening.",
    )
    wi_real_yield_3m_delta = st.slider(
        "Real yield — 3m change (pp)",
        min_value=-1.5, max_value=1.5,
        value=round(float(macro_signals["real_yield_current"]) - float(macro_signals["real_yield_3m_ago"]), 2),
        step=0.05,
        format="%.2f",
        help="How much the real yield has moved in 3 months. Positive = tightening impulse.",
    )

with l3_col:
    st.markdown(
        '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
        ' text-transform:uppercase; margin-bottom:0.6rem; font-family:Inter,sans-serif;">'
        'Layer 3 — RORO</div>',
        unsafe_allow_html=True,
    )
    wi_vix_5d = st.slider(
        "VIX 5-day change (pts)",
        min_value=-15.0, max_value=25.0,
        value=float(roro_signals.get("vix_5d_change", 0.0)),
        step=0.5,
        format="%.1f",
        help="5-day change in the VIX fear index. Positive = fear rising = risk-off signal.",
    )
    wi_dxy_5d = st.slider(
        "DXY 5-day change (%)",
        min_value=-3.0, max_value=3.0,
        value=float(roro_signals.get("dxy_5d_change", 0.0)),
        step=0.1,
        format="%.1f",
        help="5-day % change in the US Dollar Index. Rising = safe-haven bid = risk-off signal.",
    )
    wi_gold_spy_5d = st.slider(
        "Gold/SPY ratio 5-day change (%)",
        min_value=-5.0, max_value=5.0,
        value=float(roro_signals.get("gold_spy_ratio_5d_change", 0.0)),
        step=0.1,
        format="%.1f",
        help="5-day % change in Gold/SPY ratio. Rising = gold outperforming equities = flight to safety.",
    )
    wi_hyg_5d = st.slider(
        "HYG 5-day change (%)",
        min_value=-5.0, max_value=5.0,
        value=float(roro_signals.get("hyg_5d_change", 0.0)),
        step=0.1,
        format="%.1f",
        help="5-day % change in HYG (high-yield bond ETF). Falling = credit risk aversion = risk-off.",
    )
    wi_eem_vs_spy_5d = st.slider(
        "EEM vs SPY 5-day rel. perf (%)",
        min_value=-5.0, max_value=5.0,
        value=float(roro_signals.get("eem_vs_spy_5d", 0.0)),
        step=0.1,
        format="%.1f",
        help="EEM 5-day return minus SPY 5-day return. Negative = EM underperforming = de-risking.",
    )

# --- Build hypothetical signal dicts from slider values ---
wi_macro_signals = {
    # Growth
    "pmi_proxy":           wi_pmi,
    "lei_mom":             wi_lei_mom,
    "claims_trend_change": float(wi_claims_trend),
    "yield_10y_change":    wi_10y_chg,
    "yield_2y_change":     wi_2y_chg,
    # Derived: bear steepener test uses both the spread change and the 10y move
    "spread_10y2y_change": wi_10y_chg - wi_2y_chg,
    # Inflation
    "cpi_yoy":             wi_cpi,
    "cpi_yoy_lag":         wi_cpi - wi_cpi_3m_delta,
    "core_cpi_yoy":        wi_core_cpi,
    "core_cpi_yoy_lag":    wi_core_cpi - wi_core_cpi_3m_delta,
    "ppi_mom":             wi_ppi_mom,
    "breakeven_5y5y":      wi_breakeven,
    "breakeven_5y5y_lag":  wi_breakeven - wi_breakeven_3m_delta,
    "michigan_exp":        wi_michigan,
    # Monetary cycle (Layer 2)
    "fed_funds_current":   wi_ff_current,
    "fed_funds_1m_change": 0.0,  # not exposed — doesn't affect stance classification
    "fed_funds_6m_change": wi_ff_6m_chg,
    "fed_funds_12m_high":  max(wi_ff_12m_high, wi_ff_current),  # peak ≥ current
    "nfci":                wi_nfci,
    "real_yield_current":  wi_real_yield,
    "real_yield_3m_ago":   wi_real_yield - wi_real_yield_3m_delta,
}

wi_roro_signals = {
    "vix_5d_change":             wi_vix_5d,
    "dxy_5d_change":             wi_dxy_5d,
    "gold_spy_ratio_5d_change":  wi_gold_spy_5d,
    "hyg_5d_change":             wi_hyg_5d,
    "eem_vs_spy_5d":             wi_eem_vs_spy_5d,
}

# Run all three classifiers on the hypothetical inputs
wi_result       = classify_regime(wi_macro_signals)
wi_cycle_result = classify_monetary_cycle(wi_macro_signals)
wi_roro_result  = classify_roro(wi_roro_signals)
wi_regime_colour = REGIME_COLOURS[wi_result.regime]

# --- Results row ---
st.markdown(
    '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
    ' text-transform:uppercase; margin: 0.8rem 0 0.5rem; font-family:Inter,sans-serif;">'
    'Scenario Output</div>',
    unsafe_allow_html=True,
)
res_flag_col, res_breakdown_col = st.columns([1, 2])

with res_flag_col:
    st.markdown(
        _regime_flag_html(wi_result, wi_cycle_result, wi_roro_result, wi_regime_colour),
        unsafe_allow_html=True,
    )

with res_breakdown_col:
    # Show signal-level breakdown so the user can see exactly which votes changed
    bd_g, bd_i, bd_m, bd_r = st.columns(4)

    wi_g_vals, wi_i_vals = _signal_values(wi_macro_signals)

    with bd_g:
        st.markdown(f"**Growth: {wi_result.growth_score}/{wi_result.growth_available} (thr {wi_result.growth_threshold})**")
        for signal, fired in wi_result.growth_signals.items():
            val_str = _fmt_signal_val(signal, wi_g_vals[signal])
            st.markdown(f"{'✅' if fired else '❌'} {signal} `{val_str}`")

    with bd_i:
        st.markdown(f"**Inflation: {wi_result.inflation_score}/{wi_result.inflation_available} (thr {wi_result.inflation_threshold})**")
        for signal, fired in wi_result.inflation_signals.items():
            val_str = _fmt_signal_val(signal, wi_i_vals[signal])
            st.markdown(f"{'✅' if fired else '❌'} {signal} `{val_str}`")

    with bd_m:
        st.markdown(f"**Monetary: {wi_cycle_result.stance}**")
        for key, val in wi_cycle_result.signals.items():
            if isinstance(val, bool):
                st.markdown(f"{'✅' if val else '❌'} {key}")
            else:
                st.markdown(f"• {key}: **{val}**")

    with bd_r:
        st.markdown(f"**RORO: {wi_roro_result.stance} ({wi_roro_result.score}/5)**")
        for key, val in wi_roro_result.signals.items():
            is_risk_off = wi_roro_result.votes[key]
            val_str = f"{val:+.2f}" if val == val else "n/a"
            st.markdown(f"{'🔴' if is_risk_off else '🟢'} **{key}**: {val_str}")


# =============================================================================
# METHODOLOGY — explainer for the three-layer classification framework.
# Positioned after the evidence (axis breakdowns) and before the backtest so
# the page reads: answer → evidence → method → validation.
# =============================================================================

st.markdown("---")
st.markdown(
    '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
    ' text-transform:uppercase; margin: 1.2rem 0 0.5rem; font-family:Inter,sans-serif;">'
    'Methodology</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
The framework nests three layers running on different horizons. Each layer
constrains the one inside it — the quadrant tells you *what* regime you're in,
the monetary cycle tells you *how forcefully* it will express itself, and RORO
tells you *how participants are trading it right now*.
"""
)

meth_l1, meth_l2, meth_l3 = st.tabs([
    "Layer 1 — Growth/Inflation Quadrant",
    "Layer 2 — Monetary Cycle",
    "Layer 3 — RORO overlay",
])

with meth_l1:
    st.markdown(
        """
**Horizon: weeks–quarters.** Classifies the economy into one of four regimes
based on the *direction of change* of growth and inflation — not their absolute
level. A 3% GDP economy decelerating is stagflationary; a 1% GDP economy
accelerating is Goldilocks.

|  | Inflation ↑ | Inflation ↓ |
|---|---|---|
| **Growth ↑** | Overheating | Goldilocks |
| **Growth ↓** | Stagflation | Deflation/Bust |

Each axis runs a panel of directional signals that each vote **+weight if
rising/accelerating**. Leading indicators (surveys, forward-market prices,
weekly claims, yield curve) carry **weight 3**; coincident/lagging indicators
(PMI from INDPRO, headline CPI, core CPI) carry **weight 2**. The 1.5× ratio
rewards early-warning signals without letting a single leader masquerade as
a full axis verdict. Weighted tallies cross a majority threshold; axis above
= "up", below = "down". The two axis verdicts pick the quadrant.

**Growth signals — weighted max 11:** PMI proxy (INDPRO-derived) > 50 *(coincident, 2)*;
Conference Board LEI rising MoM *(leading, 3)*; initial claims 4-week trend
falling *(leading, 3)*; bear steepener (10y–2y spread widening *and* 10y
leading) *(leading, 3)*.

**Inflation signals — weighted max 13:** headline CPI YoY accelerating vs 3m ago
*(coincident, 2)*; core CPI YoY accelerating vs 3m ago *(coincident, 2)*; PPI
rising MoM *(leading, 3)*; 5Y5Y breakevens rising vs 3m ago *(leading, 3)*;
Michigan 1Y expectations > 3% *(leading, 3)*.

**Threshold:** `floor(weighted_max / 2)` — sits just under 50% of the available
weighted max (5 for growth, 6 for inflation on the full signal set). Rescales
to available signals so pre-2003 backtests don't fake TIPS data that didn't
exist yet.

**Two hard overrides (v3):**

1. **Sahm rule.** If the 3-month average of U3 unemployment has risen ≥0.5pp
   above its trailing 12-month low, force the regime to **Deflation/Bust**.
   Fires at the start of every US recession since 1950 with zero false positives.
2. **Disinflation override.** If headline CPI YoY has rolled over ≥0.5pp from
   its 12-month peak *and* is below its 3m-ago value, zero the inflation score.
   Catches late-cycle disinflation where the *level* is still hot but the
   *direction* has clearly turned (late 2022 onward).

**Confidence metric.** Every call reports how many votes would need to flip to
change the regime — 1 = Fragile, 2 = Moderate, 3+ = Strong. Flags knife-edge
calls at a glance.
"""
    )

with meth_l2:
    st.markdown(
        """
**Horizon: months–years.** The slow outer constraint. Sets the liquidity
environment and the discount rate on every risk asset. Four stances:

- **Early Tightening** — Fed has begun hiking but rate is still well below
  12m high (cycle just beginning).
- **Peak Tightening** — Fed still hiking and rate is at/near 12m high
  (terminal rate zone).
- **Early Easing** — Fed has started cutting but rate is still close to peak
  (first cuts just delivered).
- **Full Easing** — multiple cuts in, rate well below 12m peak (accommodative
  stance).

**Logic.** First look at the 6-month Fed Funds change to get *direction*
(hiking / cutting / on hold). Then use distance from the 12-month high to
determine *stage* within that direction. If on hold, classify by the absolute
rate level and Chicago Fed NFCI (tight financial conditions → Peak Tightening,
loose → Full Easing).

**Why it matters for Layer 1.** Same quadrant feels very different depending
on the monetary stance:

- **Overheating + Peak Tightening** = 2022 (brutal for everything — Fed can't
  rescue anything).
- **Goldilocks + Full Easing** = 1995, 2019 (best-case — disinflation with
  a tailwind from cuts).
- **Stagflation + Peak Tightening** = the worst combination: Fed can't ease,
  everything reprices lower.

**Supporting signal.** 10y real yield direction (3-month change in TIPS yield).
Rising real yields confirm tightening impulse; falling reals confirm easing.
Stored in the breakdown but not used in the classification itself.
"""
    )

with meth_l3:
    st.markdown(
        """
**Horizon: hours–days.** The fast overlay. Doesn't change the underlying
regime — tells you whether participants are expressing it through *risk* or
*safety* right now. Three stances: **Risk-On**, **Neutral**, **Risk-Off**.

**5-signal voting engine.** Each signal votes +1 for Risk-Off:

- **VIX 5d change positive** — fear index rising → investors buying protection.
- **DXY 5d change positive** — USD safe-haven bid → global de-risking.
- **Gold/SPY ratio 5d rising** — gold outpacing equities → flight to safety.
- **HYG 5d change negative** — high-yield bonds falling → credit stress.
- **EEM vs SPY 5d negative** — EM underperforming US → risk appetite fading.

**Thresholds.** ≥3 votes = Risk-Off, 2 = Neutral, ≤1 = Risk-On.

**Why it matters.** In risk-off episodes, normally uncorrelated assets converge
toward safety (equities ↓, USD ↑, gold ↑, vol ↑, spreads widen). The RORO
signal usually reverts to the regime signal — it's the tactical overlay that
tells you whether to press or fade the regime call on a given day.

**Reading the combinations:**

- **Overheating + Risk-On** → the rally still has legs; stay long equities.
- **Overheating + Risk-Off** → distribution phase; consider reducing exposure.
- **Goldilocks + Risk-On** → classic bull market (1995, 2019 felt like this).
- **Stagflation + Risk-Off** → 2022 — brutal for everything except commodities.
"""
    )

st.markdown(
    '<div style="color:#667085; font-size:0.78rem; margin: 0.8rem 0 0; '
    'font-family:Inter,sans-serif; line-height:1.5;">'
    '<b>Why rules-based, not ML.</b> Every vote can be explained to a PM in '
    '30 seconds. A gradient-boosted tree would likely score higher in-sample but '
    'is indefensible when it misclassifies live. Transparency beats accuracy for '
    'a framework meant to guide discretionary macro positioning.'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# BACKTEST VALIDATION
# Runs the Layer 1 classifier against 14 hand-labelled historical episodes
# (1974 Oil Shock → 2024 Disinflation). Face-validity check — does the
# framework correctly identify the regimes an interviewer would expect it to?
#
# V1 scope: single-midpoint sample per episode, full-hit + partial-hit counts.
# Pre-2003 episodes have degraded signal sets (no TIPS breakevens, no Michigan
# survey pre-1978, no LEI pre-1982) — the classifier scales thresholds to the
# available signal count so earlier decades still produce a verdict.
# =============================================================================

st.markdown("---")
st.markdown(
    '<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; color:#8899aa;'
    ' text-transform:uppercase; margin: 1.2rem 0 0.5rem; font-family:Inter,sans-serif;">'
    'Backtest Validation · 1974–2024</div>',
    unsafe_allow_html=True,
)

bt_panel   = fetch_historical_panel()
bt_results = backtest_episodes(bt_panel)
bt_summary = hit_rate_summary(bt_results)

# Summary metrics — four compact cards so the top-line numbers are unmissable.
bt_m1, bt_m2, bt_m3, bt_m4 = st.columns(4)
bt_m1.metric("Episodes", bt_summary["total"])
bt_m2.metric("Hit rate", f"{bt_summary['hit_rate_pct']}%",
             help="Full-match regime label as % of scorable episodes.")
bt_m3.metric("Weighted score", f"{bt_summary['weighted_pct']}%",
             help="Full hit = 1.0, partial (one axis right) = 0.5, miss = 0.")
bt_m4.metric("Partial / Miss",
             f"{bt_summary['partials']} / {bt_summary['misses']}",
             help="Partial = one axis correct; Miss = diagonal-opposite quadrant.")

# Colour-coded table. Streamlit's dataframe styler handles the conditional
# background per cell — green hits, amber partials, red misses.
def _colour_hit(val: str) -> str:
    return {
        "Hit":     "background-color: #065f46; color: white;",
        "Partial": "background-color: #92400E; color: white;",
        "Miss":    "background-color: #B91C1C; color: white;",
        "No data": "background-color: #4a5568; color: white;",
    }.get(val, "")

styled = bt_results.style.map(_colour_hit, subset=["Hit"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption(
    "Partial = one of growth/inflation axes matches; Miss = opposite quadrant. "
    "Pre-2003 episodes run on a reduced signal set (no TIPS breakevens); "
    "pre-1978 episodes also lack Michigan survey and LEI — the classifier scales "
    "its voting thresholds to the available signal count."
)
