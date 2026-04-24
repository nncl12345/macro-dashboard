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

from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go

from regime.classifier import REGIME_COLOURS


# -----------------------------------------------------------------------------
# Shared styling constants
# Kept here so all charts have a consistent look without repeating values.
# -----------------------------------------------------------------------------
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex colour string to rgba() for use in Plotly shape fills."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


FONT_FAMILY = "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif"
MONO_FAMILY = "'JetBrains Mono', ui-monospace, SFMono-Regular, monospace"

# Dark terminal theme — matches the app background (#080d19) and card surface (#0e1726)
FONT_COLOR  = "#e2e8f0"   # near-white text on dark bg
GRID_COLOR  = "#1a2a3a"   # very subtle grid lines — don't compete with data
BG_COLOR    = "#0e1726"   # chart background matches card surface

# Palette — shifted off the Tailwind defaults. Warmer terracotta/ochre/sage
# replacing the vivid red/amber/emerald triad that every AI dashboard ships with.
COLOR_PRIMARY    = "#6b8cae"   # slate blue — current data
COLOR_SECONDARY  = "#4a5568"   # muted slate — historical comparison
COLOR_CPI        = "#c9694d"   # terracotta — headline CPI
COLOR_CORE_CPI   = "#9c8ac7"   # muted plum — Core CPI
COLOR_FED_TARGET = "#c96a5a"   # brick — Fed 2% target line

def _base_layout(title: str = "", height: int = 380, margin: Optional[dict] = None, **kwargs) -> dict:
    """
    Return a base Plotly layout dict shared by all charts.
    Applying this to every chart ensures consistent fonts, colours, and margins.
    Pass margin= to override the default.
    """
    # Dark theme: all axis chrome (ticks, lines, zero-lines) uses GRID_COLOR so
    # they're visible but don't distract from the data traces.
    axis_style = dict(
        tickfont=dict(color=FONT_COLOR, family=FONT_FAMILY, size=11),
        title_font=dict(color=FONT_COLOR, family=FONT_FAMILY, size=12),
        gridcolor=GRID_COLOR,
        linecolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )
    return dict(
        title=dict(
            text=title,
            font=dict(size=13, color="#8899aa", family=FONT_FAMILY, weight=600),
            x=0,
            xanchor="left",
        ),
        font=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        height=height,
        margin=margin if margin is not None else dict(l=50, r=20, t=45, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(color="#8899aa", family=FONT_FAMILY, size=11),
            bgcolor="rgba(0,0,0,0)",   # transparent — let chart bg show through
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
        fillcolor="rgba(96, 165, 250, 0.07)",  # faint blue fill — subtle on dark bg
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
def plot_regime_heatmap(
    current_regime: str,
    regime_returns: dict[str, dict[str, float]],
    episode_counts: dict[str, dict[str, int]],
) -> go.Figure:
    """
    Return a Plotly heatmap of asset returns by regime, current regime highlighted.

    Args:
        current_regime: The active regime label (e.g. 'Overheating').
        regime_returns: {regime: {asset: avg_annualised_pct}} — computed or fallback.
        episode_counts: {regime: {asset: n}} — shown in hover tooltips.
    """

    # Build a DataFrame from the computed returns dict.
    # Rows = assets (Gold, Oil, SPX...), Cols = regimes.
    df = pd.DataFrame(regime_returns).T          # regimes as rows
    df = df.T                                    # transpose: assets as rows, regimes as cols

    regimes  = df.columns.tolist()
    assets   = df.index.tolist()
    z_values = df.values.tolist()  # 2D list of return values for the heatmap

    # Build a matching 2D list of episode counts for the hover tooltip.
    # n=0 means the value is a hardcoded fallback, not from real data.
    n_values = [
        [episode_counts.get(regime, {}).get(asset, 0) for regime in regimes]
        for asset in assets
    ]

    # Custom diverging colorscale: dark red → dark neutral → dark green.
    # The dark neutral (#1a2a3a) at midpoint (zero return) keeps near-zero cells
    # consistent with the chart background so they don't "pop" falsely.
    # Vivid endpoints make the extreme return cells immediately readable.
    colorscale = [
        [0.0,  "#c9694d"],   # terracotta   — worst negative
        [0.35, "#4a1f17"],   # deep terracotta
        [0.5,  "#1a2a3a"],   # dark neutral — zero return blends with chart bg
        [0.65, "#1f3327"],   # deep sage
        [1.0,  "#7a9b7e"],   # sage         — best positive
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=regimes,
        y=assets,
        colorscale=colorscale,
        zmid=0,              # anchor the midpoint colour at zero
        text=[[f"{v:+.0f}%" for v in row] for row in z_values],
        texttemplate="%{text}",
        textfont=dict(size=13, color=FONT_COLOR, family=MONO_FAMILY),
        customdata=n_values,
        showscale=True,
        colorbar=dict(
            title=dict(text="Ann. Return %", side="right", font=dict(color="#8899aa", size=11)),
            ticksuffix="%",
            tickfont=dict(color="#8899aa", size=10),
            thickness=10,
            outlinewidth=0,
            bgcolor=BG_COLOR,
        ),
        # %{customdata} shows episode count; n=0 means hardcoded fallback was used
        hovertemplate=(
            "<b>%{y}</b> in <b>%{x}</b><br>"
            "Avg annualised return: %{text}<br>"
            "Episodes: %{customdata}<extra></extra>"
        ),
    ))

    # Highlight the current regime with a coloured border box.
    # find the x-axis index of the current regime column.
    if current_regime in regimes:
        col_idx = regimes.index(current_regime)
        regime_colour = REGIME_COLOURS.get(current_regime, "#333")

        # Plotly shapes use paper coordinates for x when xref='x domain',
        # but it's simpler to use the category index directly.
        # x0/x1 are offset by 0.5 to frame the column exactly.
        # The subtle tint (alpha=0.12) sits on top of the cell colours without
        # drowning them out — the green/red heatmap still reads through clearly.
        fig.add_shape(
            type="rect",
            x0=col_idx - 0.5,
            x1=col_idx + 0.5,
            y0=-0.5,
            y1=len(assets) - 0.5,
            line=dict(color=regime_colour, width=3),
            fillcolor=_hex_to_rgba(regime_colour, 0.12),
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

    # Dark theme table: uniform dark cell backgrounds, with colored *text* on %
    # columns instead of colored backgrounds. This is the Bloomberg/terminal
    # convention — green numbers on dark, red numbers on dark. Much more readable
    # than the light-bg tint approach used in the old white theme.
    def cell_fill(col_name: str) -> list[str]:
        """Uniform dark fill — colour information carried by text, not bg."""
        # Alternate very subtle row tinting for scannability
        return [BG_COLOR if i % 2 == 0 else "#0c1520" for i in range(len(table_df))]

    def cell_font_color(col_name: str) -> list[str]:
        """Return per-cell font colours: green/red for % cols, default elsewhere."""
        if col_name not in pct_cols:
            return [FONT_COLOR] * len(table_df)
        colors = []
        for v in table_df[col_name]:
            if v > 0:
                colors.append("#7a9b7e")   # sage — positive return
            elif v < 0:
                colors.append("#c9694d")   # terracotta — negative return
            else:
                colors.append("#8899aa")   # muted         — flat
        return colors

    fill_colors      = [cell_fill(col) for col in columns]
    font_color_cells = [cell_font_color(col) for col in columns]

    fig = go.Figure(data=go.Table(
        columnwidth=[80, 80, 60, 60, 60, 70],
        header=dict(
            values=columns,               # plain strings — no HTML tags
            fill_color="#1e3a5f",         # distinct navy — clearly separates header from cells
            font=dict(color="#ffffff", size=12, family=FONT_FAMILY),
            align="center",
            height=38,
            line=dict(color=GRID_COLOR, width=1),
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(color=font_color_cells, size=12, family=MONO_FAMILY),
            align=["left"] + ["center"] * (len(columns) - 1),
            height=30,
            line=dict(color=GRID_COLOR, width=0.5),
        ),
    ))

    fig.update_layout(
        **_base_layout(title="Market Snapshot", height=320, margin=dict(l=10, r=10, t=50, b=10)),
    )

    return fig


# -----------------------------------------------------------------------------
# plot_episode_table()
#
# A Plotly table showing per-episode asset returns, grouped by regime.
# Each named event (e.g. "Global Financial Crisis") is a row, with annualised
# returns for Gold, Oil, SPX, TLT, DXY, EM shown in green/red.
# A "Regime Average" summary row follows each group.
#
# This sits below the heatmap and is the granular view — the heatmap shows
# the cross-regime picture, this table shows the specific events behind it.
#
# Input:
#   episode_data — list of dicts from compute_episode_returns() in fetcher.py
# -----------------------------------------------------------------------------
def plot_episode_table(episode_data: list[dict]) -> go.Figure:
    """Return a Plotly table of per-episode asset returns, grouped by regime."""

    assets = ["Gold", "Oil", "SPX", "TLT", "DXY", "EM"]

    # Build cell value and colour lists (one list per column)
    col_regime   = [r["regime"]     for r in episode_data]
    col_name     = [r["name"]       for r in episode_data]
    col_period   = [r["period"]     for r in episode_data]

    def _fmt_ret(v: Optional[float]) -> str:
        if v is None:
            return "—"
        return f"{v:+.1f}%"

    def _ret_color(v: Optional[float]) -> str:
        if v is None:
            return "#334155"   # very muted — no data
        if v > 0:
            return "#7a9b7e"   # sage
        if v < 0:
            return "#c9694d"   # terracotta
        return "#8899aa"

    # Per-row fill colours: average rows get a distinct background
    def _fill(row: dict) -> str:
        return "#131f35" if row["is_average"] else BG_COLOR

    fill_col = [_fill(r) for r in episode_data]

    # Regime name column — colour text to match regime colour; blank on average rows
    regime_text   = [r["regime"] if not r["is_average"] else "" for r in episode_data]
    regime_colors = [REGIME_COLOURS.get(r["regime"], FONT_COLOR) for r in episode_data]

    # Episode name: bold the average rows via HTML isn't supported in Plotly tables,
    # so we use ALL CAPS for average rows to distinguish them visually.
    name_text   = [r["name"].upper() if r["is_average"] else r["name"] for r in episode_data]
    name_colors = ["#8899aa" if r["is_average"] else FONT_COLOR for r in episode_data]

    # Build per-column formatted values and per-cell font colours for return columns
    asset_vals   = {a: [_fmt_ret(r.get(a)) for r in episode_data] for a in assets}
    asset_colors = {a: [_ret_color(r.get(a)) for r in episode_data] for a in assets}

    fig = go.Figure(data=go.Table(
        columnwidth=[90, 130, 105, 55, 55, 55, 55, 55, 55],
        header=dict(
            values=["Regime", "Episode", "Period"] + assets,
            fill_color="#1e3a5f",
            font=dict(color="#ffffff", size=12, family=FONT_FAMILY),
            align=["left", "left", "left"] + ["center"] * len(assets),
            height=36,
            line=dict(color=GRID_COLOR, width=1),
        ),
        cells=dict(
            values=[
                regime_text,
                name_text,
                col_period,
            ] + [asset_vals[a] for a in assets],
            fill_color=[
                fill_col,
                fill_col,
                fill_col,
            ] + [fill_col] * len(assets),
            font=dict(
                color=[
                    regime_colors,
                    name_colors,
                    ["#8899aa"] * len(episode_data),  # period: always muted
                ] + [asset_colors[a] for a in assets],
                size=11,
                family=MONO_FAMILY,
            ),
            align=["left", "left", "left"] + ["center"] * len(assets),
            height=28,
            line=dict(color=GRID_COLOR, width=0.5),
        ),
    ))

    # Height: 36 header + 28 per row + small margin
    table_height = 36 + len(episode_data) * 28 + 60
    fig.update_layout(
        **_base_layout(
            title="Historical Episode Returns (Ann. %)",
            height=table_height,
            margin=dict(l=10, r=10, t=50, b=10),
        )
    )

    return fig
