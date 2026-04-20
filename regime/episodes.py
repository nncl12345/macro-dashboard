# =============================================================================
# regime/episodes.py
#
# Canonical historical regime episode definitions and asset data source map.
# Pure constants — no logic, no imports, no side effects.
#
# These episodes are the anchors for the historical performance attribution
# in the regime heatmap. They are hand-picked, widely-agreed periods that
# define each regime in macro history. Returns computed from these windows
# are real, not estimated — which is the point.
# =============================================================================


# -----------------------------------------------------------------------------
# REGIME_EPISODES
#
# Each entry is a list of (start, end) ISO month strings defining a canonical
# historical episode for that regime. Dates are month-start strings but the
# fetcher resamples everything to month-end before slicing, so the boundary
# is the last trading day of that month.
#
# Episode selection rationale:
#
# Stagflation:
#   1973-10 → 1975-03  First oil shock (OPEC embargo, Nixon shock aftermath)
#   1979-01 → 1982-06  Second oil shock + Volcker tightening — the definitive
#                       stagflation episode; gold's greatest bull run
#   2021-10 → 2022-12  Post-COVID supply shock + energy crisis + Fed behind curve
#
# Goldilocks:
#   1995-01 → 1999-12  The mid-90s expansion — falling inflation, strong growth,
#                       Fed managed a soft landing in 1995
#   2003-07 → 2007-06  Post dot-com recovery; benign inflation, EM boom
#   2013-01 → 2015-12  Mid-decade expansion; QE taper tantrum was brief
#   2019-01 → 2019-12  Late-cycle Goldilocks; Fed pivoted to cuts, no recession
#
# Deflation/Bust:
#   2000-03 → 2002-09  Dot-com bust; tech wreck, mild recession, UST rally
#   2007-12 → 2009-06  GFC — the defining modern deflationary bust
#   2020-02 → 2020-04  COVID shock; fastest bear market in history, UST surge
#
# Overheating:
#   1986-01 → 1987-09  Reagan reflation; strong growth, rising inflation,
#                       ended with Black Monday (Oct 1987, excluded)
#   1999-01 → 2000-02  Late dot-com overheating; SPX blowoff, Fed hiking
#   2020-05 → 2021-09  Reopening boom; fiscal stimulus, demand surge pre-inflation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# EPISODE_NAMES
#
# Human-readable names for each episode, in the same order as REGIME_EPISODES.
# Used as row labels in the episode-level returns table on the dashboard.
# -----------------------------------------------------------------------------
EPISODE_NAMES: dict[str, list[str]] = {
    "Stagflation":    ["First Oil Shock",        "Volcker Era",                  "Post-COVID Energy Crisis"],
    "Goldilocks":     ["Mid-90s Expansion",       "Post Dot-com Recovery",        "Mid-decade Expansion",    "2019 Soft Landing"],
    "Deflation/Bust": ["Dot-com Bust",            "Global Financial Crisis",      "COVID Crash"],
    "Overheating":    ["Reagan Reflation",         "Dot-com Frenzy",              "Reopening Boom"],
}


REGIME_EPISODES: dict[str, list[tuple[str, str]]] = {
    "Stagflation": [
        ("1973-10", "1975-03"),
        ("1979-01", "1982-06"),
        ("2021-10", "2022-12"),
    ],
    "Goldilocks": [
        ("1995-01", "1999-12"),
        ("2003-07", "2007-06"),
        ("2013-01", "2015-12"),
        ("2019-01", "2019-12"),
    ],
    "Deflation/Bust": [
        ("2000-03", "2002-09"),
        ("2007-12", "2009-06"),
        ("2020-02", "2020-04"),
    ],
    "Overheating": [
        ("1986-01", "1987-09"),
        ("1999-01", "2000-02"),
        ("2020-05", "2021-09"),
    ],
}


# -----------------------------------------------------------------------------
# ASSET_SOURCES
#
# Maps each human-readable asset name (matching the keys in REGIME_RETURNS)
# to its data source, ticker/series ID, and earliest available date.
#
# The fetcher uses this to:
#   1. Know which API to call for each asset
#   2. Skip episodes that pre-date data availability
#
# Sources:
#   fred     — fetched via fredapi, returns monthly price series
#   yfinance — fetched via yf.download(), daily, resampled to month-end
#
# EM note: VEIEX (Vanguard EM Index Fund, launched May 1994) is used for
# 1994-04-14 through 2003-04-13, then stitched to EEM (launched 2003-04-14).
# The stitch is normalised so there's no level jump at the join date.
# This gives continuous EM coverage from 1994 — covering the 1995-99 Goldilocks
# episode that EEM alone would miss.
#
# TLT note: No pre-2002 proxy is used. FRED DGS30 (30yr yield, from 1977) could
# approximate bond returns via duration math, but that introduces assumptions that
# are hard to explain concisely. TLT data starts 2002-07-26; episodes before that
# are simply skipped for this asset (n≥2 for most regimes via GFC + COVID).
# -----------------------------------------------------------------------------
ASSET_SOURCES: dict[str, dict] = {
    "Gold": {
        "source":    "fred",
        "series_id": "GOLDAMGBD228NLBM",   # London AM gold fixing, USD/troy oz, daily from 1968
        "available_from": "1968-03-20",
    },
    "Oil": {
        "source":    "fred",
        "series_id": "WTISPLC",            # WTI spot price, monthly from 1946
        "available_from": "1946-01-01",
    },
    "SPX": {
        "source":    "yfinance",
        "ticker":    "^GSPC",
        "available_from": "1927-01-01",
    },
    "TLT": {
        "source":    "yfinance",
        "ticker":    "TLT",
        "available_from": "2002-07-26",    # ETF inception date
    },
    "DXY": {
        "source":    "yfinance",
        "ticker":    "DX-Y.NYB",
        "available_from": "1971-01-04",
    },
    "EM": {
        "source":    "yfinance",
        # Two tickers stitched together — logic handled in fetch_regime_price_history()
        "ticker":    "EEM",                # primary (post-2003)
        "ticker_pre": "VEIEX",            # pre-stitch (1994-05 to 2003-04)
        "stitch_date": "2003-04-14",      # EEM inception; switch from VEIEX to EEM here
        "available_from": "1994-05-04",   # VEIEX inception date
    },
}
