# =============================================================================
# data/fetcher.py
#
# Every piece of external data the dashboard needs comes through here.
# Two sources: FRED (macro series via API) and yfinance (market prices).
#
# All public functions are decorated with @st.cache_data(ttl=3600), which
# means Streamlit will cache the result in memory for 1 hour. After that it
# re-fetches. This keeps the dashboard fast — API calls happen once per hour,
# not on every page interaction.
# =============================================================================

import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred


# -----------------------------------------------------------------------------
# Constants — all series IDs and ticker symbols in one place.
# If a series ever changes ID (FRED occasionally retires series), update here.
# -----------------------------------------------------------------------------

# FRED series IDs. Each string is FRED's internal code for that data series.
# You can find any series ID by searching fred.stlouisfed.org — it's in the URL.
FRED_SERIES: dict[str, str] = {
    "cpi":            "CPIAUCSL",  # CPI All Items — the headline inflation number
    "core_cpi":       "CPILFESL",  # Core CPI — strips out food & energy (less volatile)
    "pce":            "PCEPI",     # PCE — Fed's preferred inflation gauge (smoother than CPI)
    "fed_funds":      "FEDFUNDS",  # Federal Funds Rate — the overnight rate the Fed controls
    "spread_2s10s":   "T10Y2Y",    # 10yr minus 2yr spread, pre-calculated by FRED
    "yield_10y":      "DGS10",     # 10-year Treasury yield
    "yield_2y":       "DGS2",      # 2-year Treasury yield
    "yield_5y":       "DGS5",      # 5-year Treasury yield
    "yield_30y":      "DGS30",     # 30-year Treasury yield
    "yield_3m":       "DGS3MO",    # 3-month Treasury yield
    "real_yield_10y": "DFII10",    # 10yr TIPS yield — real return after inflation
    "unemployment":   "UNRATE",    # US unemployment rate
    "indpro":         "INDPRO",    # Industrial Production Index — used to derive a PMI-like growth proxy
}

# The five maturities we plot on the yield curve chart, in order short → long.
# Tuple format: (display label, FRED series ID)
YIELD_CURVE_MATURITIES: list[tuple[str, str]] = [
    ("3M",  "DGS3MO"),
    ("2Y",  "DGS2"),
    ("5Y",  "DGS5"),
    ("10Y", "DGS10"),
    ("30Y", "DGS30"),
]

# yfinance ticker symbols for the six assets in the market snapshot.
# These are Yahoo Finance's codes — searchable at finance.yahoo.com.
TICKERS: dict[str, str] = {
    "Gold":  "GC=F",      # Gold futures (front-month contract)
    "Oil":   "CL=F",      # WTI crude oil futures
    "DXY":   "DX-Y.NYB",  # US Dollar index vs basket of six major currencies
    "SPX":   "^GSPC",     # S&P 500 index
    "EM":    "EEM",       # iShares MSCI Emerging Markets ETF
    "TLT":   "TLT",       # iShares 20+ Year Treasury Bond ETF (long-duration bonds)
}


# -----------------------------------------------------------------------------
# FRED client
# Reads the API key and returns a connected Fred client object.
# Called inside fetch_fred_series() — not cached itself because the Fred
# object can't be serialised, but fetch_fred_series() caches its *output*
# so this only runs when the cache is cold or expired.
# -----------------------------------------------------------------------------
def _get_fred_client() -> Fred:
    """Initialise and return a FRED API client using the stored API key."""
    try:
        # On Streamlit Cloud, secrets are set in the app dashboard and injected
        # at runtime via st.secrets — no file needed on the server.
        api_key = st.secrets["FRED_KEY"]
    except (FileNotFoundError, KeyError):
        # Locally, load from the .env file in the project root.
        load_dotenv()
        api_key = os.getenv("FRED_KEY")

    if not api_key:
        raise ValueError(
            "FRED_KEY not found. Add it to .env (local) or Streamlit secrets (cloud)."
        )

    return Fred(api_key=api_key)


# -----------------------------------------------------------------------------
# Generic FRED fetcher
# Every specific fetch function below calls this one. Caching here means we
# never hit the FRED API more than once per series per hour regardless of how
# many functions call it.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, periods: int = 36) -> pd.Series:
    """Fetch the last N observations of a FRED series. Returns a dated pd.Series."""
    fred = _get_fred_client()
    # fred.get_series() returns a pandas Series with a DatetimeIndex.
    # FRED sometimes has trailing NaN values (data not yet released), so we drop them.
    data = fred.get_series(series_id)
    return data.dropna().tail(periods)


# -----------------------------------------------------------------------------
# fetch_macro_inputs()
# The bridge between the data layer and the regime classifier.
# Returns exactly the three floats that classify_regime() in classifier.py
# expects: cpi_mom, pmi, spread_2s10s.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_macro_inputs() -> dict[str, float]:
    """
    Return the latest CPI MoM %, PMI reading, and 2s10s spread.
    These are the three inputs consumed by classify_regime().
    """
    # --- CPI Month-on-Month % change ---
    # CPIAUCSL is an index level (e.g. 305.1), not a % change.
    # pct_change() calculates the % change from the prior row.
    # Multiply by 100 to convert from decimal (0.003) → percentage (0.3).
    cpi = fetch_fred_series("CPIAUCSL", periods=3)
    cpi_mom = float(cpi.pct_change().dropna().iloc[-1] * 100)

    # --- PMI (proxy via Industrial Production) ---
    # The ISM Manufacturing PMI isn't freely available on FRED, so we derive a
    # PMI-equivalent signal from INDPRO (Industrial Production Index).
    #
    # Method: calculate the 6-month annualised % change in INDPRO, then map it
    # onto the PMI 0–100 scale by centring on 50:
    #   pmi_proxy = 50 + (indpro_annualised_growth * scaling_factor)
    #
    # Scaling factor = 2: a healthy 2% annualised growth rate in industrial
    # production maps to PMI ~54 (clearly above the 50 expansion line).
    # Examples:
    #   flat production  (0% growth)      → PMI proxy = 50  (neutral)
    #   +2% annualised growth             → PMI proxy = 54  (expanding)
    #   -2% annualised contraction        → PMI proxy = 46  (contracting)
    indpro = fetch_fred_series("INDPRO", periods=10)
    # pct_change(6) = % change vs 6 months prior. ×2 annualises it (×12/6).
    indpro_6m_pct = float(indpro.pct_change(6).dropna().iloc[-1] * 100)
    indpro_annualised = indpro_6m_pct * 2
    pmi = round(50 + (indpro_annualised * 2), 1)

    # --- 2s10s Spread ---
    # T10Y2Y = 10yr Treasury yield minus 2yr yield, in percentage points.
    # Positive = normal (long rates > short rates).
    # Negative = inverted (short rates > long rates) — historically a recession signal.
    spread = fetch_fred_series("T10Y2Y", periods=5)
    spread_2s10s = float(spread.iloc[-1])

    return {
        "cpi_mom":      round(cpi_mom, 3),
        "pmi":          round(pmi, 1),
        "spread_2s10s": round(spread_2s10s, 2),
    }


# -----------------------------------------------------------------------------
# fetch_kpi_data()
# The four headline numbers shown in the metric cards at the top of the
# dashboard. Each card has a single current value and that's it — no history.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_kpi_data() -> dict[str, float]:
    """Return current values for the four dashboard KPI metric cards."""
    # CPI year-on-year — pct_change(12) compares each month vs the same month
    # 12 months prior (i.e. true year-on-year, not annualised MoM).
    # We fetch 14 periods: 12 for the YoY base + 1 current + 1 buffer.
    cpi = fetch_fred_series("CPIAUCSL", periods=14)
    cpi_yoy = float(cpi.pct_change(12).dropna().iloc[-1] * 100)

    # Fed Funds Rate — the overnight rate the Fed sets. Currently expressed as
    # the effective rate (i.e. what banks actually charge each other overnight).
    fed_funds = fetch_fred_series("FEDFUNDS", periods=2)
    fed_funds_rate = float(fed_funds.iloc[-1])

    # 2s10s spread — same series as used in the classifier
    spread = fetch_fred_series("T10Y2Y", periods=5)
    spread_2s10s = float(spread.iloc[-1])

    # 10yr Real Yield (TIPS) — the yield on inflation-linked Treasuries.
    # This is the return an investor earns AFTER inflation is stripped out.
    # When real yields rise, it tightens financial conditions broadly
    # (expensive to borrow in real terms). Closely watched by markets.
    real_yield = fetch_fred_series("DFII10", periods=5)
    real_yield_10y = float(real_yield.iloc[-1])

    return {
        "cpi_yoy":        round(cpi_yoy, 2),
        "fed_funds":      round(fed_funds_rate, 2),
        "spread_2s10s":   round(spread_2s10s, 2),
        "real_yield_10y": round(real_yield_10y, 2),
    }


# -----------------------------------------------------------------------------
# fetch_yield_curve()
# Returns two snapshots of the yield curve — today and one year ago — as
# pd.Series objects. The chart in plots.py overlays them so you can see
# visually whether the curve has steepened, flattened, or inverted.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_yield_curve() -> tuple[pd.Series, pd.Series]:
    """
    Return (current_curve, one_year_ago_curve) as pd.Series.
    Index = maturity label (e.g. '10Y'), values = yield in %.
    """
    current_yields: dict[str, float] = {}
    year_ago_yields: dict[str, float] = {}

    one_year_ago = datetime.today() - timedelta(days=365)

    for label, series_id in YIELD_CURVE_MATURITIES:
        # Fetch ~400 trading days (about 18 months) to cover both today
        # and one year ago in a single API call per maturity.
        data = fetch_fred_series(series_id, periods=400)

        # Current: the most recent available data point
        current_yields[label] = float(data.iloc[-1])

        # 1yr ago: find the last available date on or before one_year_ago.
        # We can't look up an exact date because markets close on weekends
        # and holidays — so we take the nearest previous trading day.
        past_data = data[data.index <= pd.Timestamp(one_year_ago)]
        year_ago_yields[label] = (
            float(past_data.iloc[-1]) if not past_data.empty else float("nan")
        )

    return pd.Series(current_yields), pd.Series(year_ago_yields)


# -----------------------------------------------------------------------------
# fetch_cpi_trend()
# Returns 24 months of CPI and Core CPI as year-on-year % changes.
# The chart in plots.py draws these as lines with a dashed 2% Fed target.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_cpi_trend(months: int = 24) -> pd.DataFrame:
    """
    Return monthly CPI and Core CPI YoY % change for the last N months.
    Columns: ['CPI YoY', 'Core CPI YoY']. Index: monthly DatetimeIndex.
    """
    # We need extra history beyond the display window to calculate 12-month
    # % changes. E.g. to show Jan 2024 YoY, we need Jan 2023 as the base.
    extra = 14
    cpi      = fetch_fred_series("CPIAUCSL", periods=months + extra)
    core_cpi = fetch_fred_series("CPILFESL", periods=months + extra)

    df = pd.DataFrame({
        "CPI YoY":      cpi.pct_change(12) * 100,
        "Core CPI YoY": core_cpi.pct_change(12) * 100,
    }).dropna().tail(months)

    return df


# -----------------------------------------------------------------------------
# fetch_market_snapshot()
# Downloads ~1 year of daily price history for all six assets via yfinance,
# then calculates % changes over four windows: 1 day, 1 week, 1 month, YTD.
#
# Why yfinance instead of FRED for this?
# FRED is great for macro series but doesn't carry commodity futures or ETF
# prices. yfinance pulls from Yahoo Finance, which has all of these for free.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_snapshot() -> pd.DataFrame:
    """
    Return current price and 1D/1W/1M/YTD % changes for all dashboard tickers.
    Returns a DataFrame indexed by asset name (Gold, Oil, DXY, SPX, EM, TLT).
    """
    rows: list[dict] = []

    for name, ticker in TICKERS.items():
        # Download roughly 1 year of adjusted daily closes.
        # auto_adjust=True bakes in splits/dividends so prices are comparable.
        raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

        if raw.empty:
            continue

        # yfinance returns a DataFrame for single tickers — squeeze to a 1D Series
        prices = raw["Close"].squeeze().dropna()

        if len(prices) < 2:
            continue

        current_price = float(prices.iloc[-1])

        def pct_vs(n_days: int) -> float:
            """Return % change from n trading days ago to today."""
            if len(prices) < n_days + 1:
                return float("nan")
            return float((prices.iloc[-1] / prices.iloc[-(n_days + 1)] - 1) * 100)

        # YTD: find the last available trading day before Jan 1 of this year,
        # then calculate % change from that close to today.
        year_start = pd.Timestamp(datetime.today().year, 1, 1)
        ytd_base = prices[prices.index < year_start]
        ytd_pct = (
            float((prices.iloc[-1] / ytd_base.iloc[-1] - 1) * 100)
            if not ytd_base.empty
            else float("nan")
        )

        rows.append({
            "Asset": name,
            "Price": round(current_price, 2),
            "1D %":  round(pct_vs(1), 2),
            "1W %":  round(pct_vs(5), 2),   # 5 trading days ≈ 1 calendar week
            "1M %":  round(pct_vs(21), 2),  # 21 trading days ≈ 1 calendar month
            "YTD %": round(ytd_pct, 2),
        })

    return pd.DataFrame(rows).set_index("Asset")


# -----------------------------------------------------------------------------
# Quick test — run `python3 data/fetcher.py` to verify live data is coming in.
# Will print each dataset to the terminal so you can eyeball the values.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Macro Inputs (classifier inputs) ---")
    print(fetch_macro_inputs())

    print("\n--- KPI Data ---")
    print(fetch_kpi_data())

    print("\n--- CPI Trend (last 6 months) ---")
    print(fetch_cpi_trend(months=6))

    print("\n--- Yield Curve ---")
    current, year_ago = fetch_yield_curve()
    print("Current:    ", current.to_dict())
    print("1yr ago:    ", year_ago.to_dict())

    print("\n--- Market Snapshot ---")
    print(fetch_market_snapshot())
