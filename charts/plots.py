# =============================================================================
# charts/plots.py
#
# All Plotly chart functions for the dashboard. Each function takes prepared
# data (from data/fetcher.py) and returns a go.Figure ready for st.plotly_chart().
#
# Four charts:
#   plot_yield_curve()    — current curve vs 1yr ago overlay
#   plot_cpi_trend()      — CPI & Core CPI YoY with 2% Fed target line
#   plot_regime_heatmap() — asset returns by regime, current regime highlighted
#   plot_market_snapshot()— price + % change table for all tracked assets
# =============================================================================

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from regime.classifier import REGIME_COLOURS, REGIME_RETURNS


# -----------------------------------------------------------------------------
# Shared styling constants
# Kept here so all charts have a consistent look without repeating values.
# -----------------------------------------------------------------------------
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
FONT_COLOR  = "#000000"
GRID_COLOR  = "#ebebeb"
BG_COLOR    = "white"

# Colour palette for chart lines
COLOR_PRIMARY   = "#2563EB"   # blue — current data
COLOR_SECONDARY = "#94a3b8"   # grey — historical comparison
COLOR_CPI       = "#f97316"   # orange — CPI
COLOR_CORE_CPI  = "#8b5cf6"   # purple — Core CPI
COLOR_FED_TARGET = "#ef4444"  # red — Fed 2% target line

def _base_layout(title: str = "", height: int = 380, margin: Optional[dict] = None, **kwargs) -> dict:
    """
    Return a base Plotly layout dict shared by all charts.
    Applying this to every chart ensures consistent fonts, colours, and margins.
    Pass margin= to override the default.
    """
    # axis_style is applied to both xaxis and yaxis unless the caller overrides them.
    # Explicitly setting tickfont and title font forces black text — Plotly's
    # white theme otherwise renders axis labels in a washed-out grey.
    axis_style = dict(
        tickfont=dict(color=FONT_COLOR, family=FONT_FAMILY),
        title_font=dict(color=FONT_COLOR, family=FONT_FAMILY),
    )
    return dict(
        title=dict(text=title, font=dict(size=14, color=FONT_COLOR, family=FONT_FAMILY), x=0, xanchor="left"),
        font=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        height=height,
        margin=margin if margin is not None else dict(l=50, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(color=FONT_COLOR, family=FONT_FAMILY),
        ),
        xaxis=axis_style,
        yaxis=axis_style,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# plot_yield_curve()
#
# Shows the Treasury yield curve at two points in time: today and one year ago.
# Overlaying them reveals how the curve has shifted — flattened, steepened,
# or inverted. An inverted curve (short rates > long rates) is a classic
# recession warning signal.
#
# Inputs:
#   current   — pd.Series, index = maturity labels ('3M','2Y','5Y','10Y','30Y')
#   year_ago  — pd.Series, same structure but from 12 months prior
# -----------------------------------------------------------------------------
def plot_yield_curve(current: pd.Series, year_ago: pd.Series) -> go.Figure:
    """Return a Plotly figure showing the current yield curve vs 1yr ago."""
    fig = go.Figure()

    # Current yield curve — solid line, primary colour
    fig.add_trace(go.Scatter(
        x=current.index.tolist(),
        y=current.values.tolist(),
        mode="lines+markers",
        name="Today",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        marker=dict(size=7),
    ))

    # 1yr ago yield curve — dashed grey line for comparison
    fig.add_trace(go.Scatter(
        x=year_ago.index.tolist(),
        y=year_ago.values.tolist(),
        mode="lines+markers",
        name="1yr ago",
        line=dict(color=COLOR_SECONDARY, width=2, dash="dash"),
        marker=dict(size=6),
    ))

    # Shade the area between the two curves to make the shift visually obvious.
    # fill='tonexty' fills between this trace and the one above it.
    fig.add_trace(go.Scatter(
        x=year_ago.index.tolist(),
        y=year_ago.values.tolist(),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.08)",  # faint blue fill
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(**_base_layout(title="Yield Curve"))
    fig.update_yaxes(title="Yield (%)", gridcolor=GRID_COLOR, zeroline=False)
    fig.update_xaxes(
        title="Maturity",
        showgrid=False,
        categoryorder="array",
        categoryarray=["3M", "2Y", "5Y", "10Y", "30Y"],
    )

    return fig


# -----------------------------------------------------------------------------
# plot_cpi_trend()
#
# Shows CPI and Core CPI year-on-year % change over the last 24 months.
# The dashed 2% line is the Fed's inflation target — visually shows how far
# above or below target inflation is running.
#
# Input:
#   df — pd.DataFrame with columns ['CPI YoY', 'Core CPI YoY'],
#        monthly DatetimeIndex, from fetch_cpi_trend()
# -----------------------------------------------------------------------------
def plot_cpi_trend(df: pd.DataFrame) -> go.Figure:
    """Return a Plotly figure showing CPI and Core CPI YoY trend with Fed target."""
    fig = go.Figure()

    # CPI YoY — headline number, more volatile (includes food & energy)
    fig.add_trace(go.Scatter(
        x=df.index.tolist(),
        y=df["CPI YoY"].tolist(),
        mode="lines",
        name="CPI YoY",
        line=dict(color=COLOR_CPI, width=2.5),
    ))

    # Core CPI YoY — strips out food and energy, watched more closely by the Fed
    fig.add_trace(go.Scatter(
        x=df.index.tolist(),
        y=df["Core CPI YoY"].tolist(),
        mode="lines",
        name="Core CPI YoY",
        line=dict(color=COLOR_CORE_CPI, width=2.5),
    ))

    # Fed 2% target line — dashed red horizontal line
    # This is the Fed's official inflation target. Persistent readings above this
    # line is what triggers rate hikes.
    fig.add_hline(
        y=2.0,
        line=dict(color=COLOR_FED_TARGET, width=1.5, dash="dash"),
        annotation_text="Fed target 2%",
        annotation_position="bottom right",
        annotation_font=dict(color=COLOR_FED_TARGET, size=11),
    )

    fig.update_layout(**_base_layout(title="CPI Trend (YoY %)"))
    fig.update_yaxes(title="YoY %", gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%")
    fig.update_xaxes(showgrid=False, title="")

    return fig


# -----------------------------------------------------------------------------
# plot_regime_heatmap()
#
# The centrepiece chart. Shows average annualised asset returns (%) for each
# of the four regimes, colour-coded green (positive) to red (negative).
#
# The current regime column is highlighted with a box so the viewer immediately
# sees: "we're here — this is what history says about asset performance."
#
# Inputs:
#   current_regime — string, e.g. "Overheating" (from RegimeResult.regime)
# -----------------------------------------------------------------------------
def plot_regime_heatmap(current_regime: str) -> go.Figure:
    """Return a Plotly heatmap of asset returns by regime, current regime highlighted."""

    # Build a DataFrame from the REGIME_RETURNS dict in classifier.py.
    # Rows = assets (Gold, Oil, SPX...), Cols = regimes.
    df = pd.DataFrame(REGIME_RETURNS).T          # regimes as rows
    df = df.T                                     # transpose: assets as rows, regimes as cols

    regimes = df.columns.tolist()
    assets  = df.index.tolist()
    z_values = df.values.tolist()  # 2D list of return values for the heatmap

    # Custom diverging colorscale: red (negative) → white (zero) → green (positive)
    # Using explicit midpoint at 0 ensures white = breakeven return
    colorscale = [
        [0.0,  "#d32f2f"],   # deep red  — very negative (e.g. -25%)
        [0.35, "#ef9a9a"],   # light red
        [0.5,  "#ffffff"],   # white     — zero return
        [0.65, "#a5d6a7"],   # light green
        [1.0,  "#1b5e20"],   # deep green — very positive (e.g. +22%)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=regimes,
        y=assets,
        colorscale=colorscale,
        zmid=0,              # anchor the midpoint colour (white) at zero
        text=[[f"{v:+.0f}%" for v in row] for row in z_values],
        texttemplate="%{text}",
        textfont=dict(size=13, color=FONT_COLOR),
        showscale=True,
        colorbar=dict(
            title=dict(text="Ann. Return %", side="right"),
            ticksuffix="%",
            thickness=12,
        ),
        hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>Avg annualised return: %{text}<extra></extra>",
    ))

    # Highlight the current regime with a coloured border box.
    # find the x-axis index of the current regime column.
    if current_regime in regimes:
        col_idx = regimes.index(current_regime)
        regime_colour = REGIME_COLOURS.get(current_regime, "#333")

        # Plotly shapes use paper coordinates for x when xref='x domain',
        # but it's simpler to use the category index directly.
        # x0/x1 are offset by 0.5 to frame the column exactly.
        fig.add_shape(
            type="rect",
            x0=col_idx - 0.5,
            x1=col_idx + 0.5,
            y0=-0.5,
            y1=len(assets) - 0.5,
            line=dict(color=regime_colour, width=3),
            fillcolor="rgba(0,0,0,0)",   # transparent fill — border only
        )

        # Label above the highlighted column
        fig.add_annotation(
            x=current_regime,
            y=len(assets) - 0.5,
            text="◀ NOW",
            showarrow=False,
            font=dict(color=regime_colour, size=11, family=FONT_FAMILY),
            yanchor="bottom",
            yshift=6,
        )

    fig.update_layout(**_base_layout(title="Historical Asset Returns by Regime (Ann. %)", height=420))
    fig.update_xaxes(side="bottom", showgrid=False, title="")
    fig.update_yaxes(showgrid=False, title="", autorange="reversed")

    return fig


# -----------------------------------------------------------------------------
# plot_market_snapshot()
#
# A styled Plotly table showing live prices and % changes for all tracked
# assets. % change cells are coloured green (positive) or red (negative)
# so you can read the risk-on/risk-off tone at a glance.
#
# Input:
#   df — pd.DataFrame from fetch_market_snapshot(), indexed by asset name
# -----------------------------------------------------------------------------
def plot_market_snapshot(df: pd.DataFrame) -> go.Figure:
    """Return a styled Plotly table of current prices and % changes."""

    # Reset index so 'Asset' becomes a regular column
    table_df = df.reset_index()

    columns  = table_df.columns.tolist()
    pct_cols = ["1D %", "1W %", "1M %", "YTD %"]

    # Build cell values — format % columns with + sign and % suffix
    cell_values = []
    for col in columns:
        if col in pct_cols:
            cell_values.append([
                f"+{v:.2f}%" if v > 0 else f"{v:.2f}%" if v != 0 else "0.00%"
                for v in table_df[col]
            ])
        elif col == "Price":
            cell_values.append([f"{v:,.2f}" for v in table_df[col]])
        else:
            cell_values.append(table_df[col].tolist())

    # Colour each % cell: green if positive, red if negative, grey if zero/NaN
    def cell_colours(col_name: str) -> list[str]:
        """Return a list of background hex colours for one column."""
        if col_name not in pct_cols:
            return ["#f8f8f8"] * len(table_df)
        colours = []
        for v in table_df[col_name]:
            if v > 0:
                colours.append("#d4edda")   # light green
            elif v < 0:
                colours.append("#f8d7da")   # light red
            else:
                colours.append("#f8f8f8")   # neutral grey
        return colours

    fill_colors = [cell_colours(col) for col in columns]

    fig = go.Figure(data=go.Table(
        columnwidth=[80, 80, 60, 60, 60, 70],
        header=dict(
            values=[f"<b>{c}</b>" for c in columns],
            fill_color="#1e3a5f",         # dark navy header
            font=dict(color="white", size=12, family=FONT_FAMILY),
            align="center",
            height=36,
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(color=FONT_COLOR, size=12, family=FONT_FAMILY),
            align=["left"] + ["center"] * (len(columns) - 1),
            height=32,
        ),
    ))

    fig.update_layout(
        **_base_layout(title="Market Snapshot", height=320, margin=dict(l=10, r=10, t=50, b=10)),
    )

    return fig
