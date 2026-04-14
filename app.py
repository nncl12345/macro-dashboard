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
    fetch_yield_curve,
)
from regime.classifier import REGIME_COLOURS, classify_regime

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
# Custom CSS
# Streamlit's default styling is functional but plain. These overrides:
#   - Make the regime flag text larger and bolder
#   - Remove default top padding so Row 1 sits high on the page
#   - Style the KPI metric cards (st.metric) to match the regime colour
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; }

    /* Regime flag — large centred label */
    .regime-flag {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 0.25rem;
    }

    /* Sub-label under the regime flag */
    .regime-sub {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.1rem;
    }

    /* Make st.metric values slightly larger */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
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

# Run the regime classifier on the live signals
result = classify_regime(macro_signals)

# Colour used throughout this render for the current regime
regime_colour = REGIME_COLOURS[result.regime]


# =============================================================================
# ROW 1 — Regime flag + KPI cards
# =============================================================================

# Regime flag on the left, four KPI metrics across the right
flag_col, kpi1, kpi2, kpi3, kpi4 = st.columns([2, 1, 1, 1, 1])

with flag_col:
    # Large colour-coded regime label — the headline of the whole dashboard
    st.markdown(
        f'<div class="regime-flag" style="background-color:{regime_colour}22; '
        f'color:{regime_colour}; border: 2px solid {regime_colour};">'
        f'{result.regime.upper()}'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Score breakdown shown as a sub-label so the viewer knows how confident
    # the classification is (e.g. 4/4 growth signals vs 2/4)
    st.markdown(
        f'<div class="regime-sub">'
        f'Growth {result.growth_score}/4 · Inflation {result.inflation_score}/4'
        f'</div>',
        unsafe_allow_html=True,
    )

with kpi1:
    # CPI YoY — the headline inflation number most people refer to
    # Delta shows whether it's above or below the Fed's 2% target
    st.metric(
        label="CPI YoY",
        value=f"{kpi['cpi_yoy']:.2f}%",
        delta=f"{kpi['cpi_yoy'] - 2.0:.2f}% vs target",
        delta_color="inverse",   # inverse: above target = red (bad), below = green
    )

with kpi2:
    # Fed Funds Rate — where the Fed has set the overnight lending rate
    st.metric(
        label="Fed Funds",
        value=f"{kpi['fed_funds']:.2f}%",
    )

with kpi3:
    # 2s10s Spread — positive = normal curve, negative = inverted (recession signal)
    spread = kpi['spread_2s10s']
    st.metric(
        label="2s10s Spread",
        value=f"{spread:.2f}%",
        delta="inverted" if spread < 0 else "normal",
        delta_color="inverse" if spread < 0 else "normal",
    )

with kpi4:
    # 10yr Real Yield — return on Treasuries after stripping out inflation
    # Positive = real return available in bonds, negative = financial repression
    st.metric(
        label="10yr Real Yield",
        value=f"{kpi['real_yield_10y']:.2f}%",
        delta_color="normal",
    )

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
    )

with right_col:
    # CPI and Core CPI over 24 months with the Fed 2% target as a dashed line.
    # Lets you see at a glance whether inflation is converging to or diverging
    # from target.
    st.plotly_chart(
        plot_cpi_trend(cpi_df),
        use_container_width=True,
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
)

st.divider()


# =============================================================================
# ROW 4 — Market snapshot table (full width)
# Live prices + 1D/1W/1M/YTD % changes, green/red coloured cells
# =============================================================================

st.plotly_chart(
    plot_market_snapshot(snapshot_df),
    use_container_width=True,
)


# =============================================================================
# FOOTER — signal breakdown expander
# Collapsed by default so it doesn't clutter the view, but available for
# anyone who wants to see exactly which signals drove the regime call.
# This is the "show your working" section — important for CV credibility.
# =============================================================================

with st.expander("📋  Signal breakdown — what drove this regime classification?"):
    g_col, i_col = st.columns(2)

    with g_col:
        st.markdown(f"**Growth score: {result.growth_score}/4**")
        for signal, fired in result.growth_signals.items():
            icon = "✅" if fired else "❌"
            st.markdown(f"{icon} {signal}")

    with i_col:
        st.markdown(f"**Inflation score: {result.inflation_score}/4**")
        for signal, fired in result.inflation_signals.items():
            icon = "✅" if fired else "❌"
            st.markdown(f"{icon} {signal}")

    st.caption(
        "Regime = Layer 1 (Growth/Inflation Quadrant). "
        "Monetary cycle (Layer 2) and RORO overlay (Layer 3) coming soon."
    )
