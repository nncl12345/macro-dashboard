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
    plot_market_snapshot,
    plot_regime_heatmap,
    plot_yield_curve,
)
from data.fetcher import (
    fetch_cpi_trend,
    fetch_kpi_data,
    fetch_macro_inputs,
    fetch_market_snapshot,
    fetch_roro_signals,
    fetch_yield_curve,
)
from regime.classifier import REGIME_COLOURS, classify_regime, classify_monetary_cycle, classify_roro

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

# Run all three classifiers on the live signals
result        = classify_regime(macro_signals)
cycle_result  = classify_monetary_cycle(macro_signals)
roro_result   = classify_roro(roro_signals)

# Colour used throughout this render for the current regime
regime_colour = REGIME_COLOURS[result.regime]


# =============================================================================
# ROW 1 — Regime flag + KPI cards
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


# Regime flag on the left, four KPI metrics across the right
flag_col, kpi1, kpi2, kpi3, kpi4 = st.columns([2, 1, 1, 1, 1])

with flag_col:
    cycle_colour = cycle_result.colour
    roro_colour  = roro_result.colour

    # The regime flag is the primary signal on the page. Dark card with a
    # coloured border that matches the active regime — so the card itself
    # "glows" the regime colour. Monospace font on the regime label gives
    # the Bloomberg terminal look.
    flag_html = (
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

        # Fed Cycle badge
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

        # RORO badge
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

        # Score tally — small, muted, below the badges
        f'<div style="font-size:0.68rem; color:#4a5568; margin-top:0.55rem;'
        f' font-family:Inter,sans-serif;">'
        f'Growth {result.growth_score}/4 &nbsp;&middot;&nbsp; Inflation {result.inflation_score}/4'
        f'</div>'

        f'</div>'
    )
    st.markdown(flag_html, unsafe_allow_html=True)

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
    plot_regime_heatmap(result.regime),
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

    with g_col:
        st.markdown(f"**Layer 1 — Growth: {result.growth_score}/4**")
        for signal, fired in result.growth_signals.items():
            icon = "✅" if fired else "❌"
            st.markdown(f"{icon} {signal}")

    with i_col:
        st.markdown(f"**Layer 1 — Inflation: {result.inflation_score}/4**")
        for signal, fired in result.inflation_signals.items():
            icon = "✅" if fired else "❌"
            st.markdown(f"{icon} {signal}")

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
