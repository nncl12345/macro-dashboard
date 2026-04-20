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
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred


# -----------------------------------------------------------------------------
# Constants — all series IDs and ticker symbols in one place.
# If a series ever changes ID (FRED occasionally retires series), update here.
# -----------------------------------------------------------------------------

# FRED series IDs. Each string is FRED's internal code.
# Find any ID by searching fred.stlouisfed.org — it appears in the URL.
FRED_SERIES: dict[str, str] = {
    # --- Inflation ---
    "cpi":              "CPIAUCSL",  # CPI All Items (monthly)
    "core_cpi":         "CPILFESL",  # Core CPI — strips food & energy
    "pce":              "PCEPI",     # PCE — Fed's preferred inflation gauge
    "ppi":              "PPIACO",    # PPI All Commodities — leads CPI by 1–3 months
    "breakeven_5y5y":   "T5YIFR",   # 5Y5Y forward breakeven — market's long-run inflation view
    "michigan_exp":     "MICH",      # U Michigan 1Y inflation expectations (survey)

    # --- Growth ---
    "indpro":           "INDPRO",    # Industrial Production Index (PMI proxy)
    "claims":           "ICSA",      # Initial Jobless Claims (weekly — fastest leading indicator)
    "lei":              "USSLIND",   # Conference Board Leading Economic Index
    "retail_sales":     "RSXFS",     # Retail Sales ex Food Services (consumer demand)

    # --- Rates & Yield Curve ---
    "fed_funds":        "FEDFUNDS",  # Federal Funds Rate
    "spread_2s10s":     "T10Y2Y",    # 10yr – 2yr spread (classic recession signal)
    "spread_10y3m":     "T10Y3M",    # 10yr – 3m spread (often more reliable predictor)
    "yield_10y":        "DGS10",     # 10-year Treasury yield
    "yield_2y":         "DGS2",      # 2-year Treasury yield
    "yield_5y":         "DGS5",      # 5-year Treasury yield
    "yield_30y":        "DGS30",     # 30-year Treasury yield
    "yield_3m":         "DGS3MO",    # 3-month Treasury yield
    "real_yield_10y":   "DFII10",    # 10yr TIPS real yield (inflation-adjusted return)

    # --- Financial Conditions ---
    "nfci":             "NFCI",      # Chicago Fed National Financial Conditions Index
                                     # Negative = loose (easy to borrow), positive = tight

    # --- Labour ---
    "unemployment":     "UNRATE",    # Unemployment rate
}

# Maturities for the yield curve chart, ordered short → long.
YIELD_CURVE_MATURITIES: list[tuple[str, str]] = [
    ("3M",  "DGS3MO"),
    ("2Y",  "DGS2"),
    ("5Y",  "DGS5"),
    ("10Y", "DGS10"),
    ("30Y", "DGS30"),
]

# yfinance tickers — Yahoo Finance codes.
# Market snapshot assets + RORO/Layer 3 signals.
TICKERS: dict[str, str] = {
    # Market snapshot
    "Gold":  "GC=F",      # Gold futures (front-month)
    "Oil":   "CL=F",      # WTI crude oil futures
    "DXY":   "DX-Y.NYB",  # US Dollar index
    "SPX":   "^GSPC",     # S&P 500
    "EM":    "EEM",       # Emerging market equities ETF
    "TLT":   "TLT",       # 20+ year Treasury bond ETF

    # RORO / Layer 3 signals
    "HYG":   "HYG",       # High yield corporate bond ETF — credit risk appetite
    "VIX":   "^VIX",      # CBOE Volatility Index — primary fear gauge
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
        # On Streamlit Cloud, secrets are injected via the app dashboard
        api_key = st.secrets["FRED_KEY"]
    except (FileNotFoundError, KeyError):
        # Locally, load from .env in the project root
        load_dotenv()
        api_key = os.getenv("FRED_KEY")

    if not api_key:
        raise ValueError(
            "FRED_KEY not found. Add it to .env (local) or Streamlit secrets (cloud)."
        )

    return Fred(api_key=api_key)


# -----------------------------------------------------------------------------
# Generic FRED fetcher
# All specific fetch functions call this. Caching here means we never hit
# the FRED API more than once per series per hour.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, periods: int = 36) -> pd.Series:
    """Fetch the last N observations of a FRED series. Returns a dated pd.Series."""
    fred = _get_fred_client()
    # FRED sometimes has trailing NaN values (data not yet released), so we drop them.
    data = fred.get_series(series_id)
    return data.dropna().tail(periods)


# -----------------------------------------------------------------------------
# fetch_macro_inputs()
# The bridge between the data layer and the regime classifier.
# Returns all signals needed by classify_regime() in regime/classifier.py.
#
# Organised into two groups matching the scoring engine:
#   - 4 growth signals
#   - 4 inflation signals (+ lag values for acceleration checks)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_macro_inputs() -> dict[str, float]:
    """
    Return all signals consumed by classify_regime().
    Includes current values and lag values for acceleration comparisons.
    """

    # =========================================================================
    # GROWTH SIGNALS
    # =========================================================================

    # --- PMI proxy (via INDPRO) ---
    # ISM Manufacturing PMI isn't freely available on FRED, so we derive a
    # PMI-equivalent from INDPRO (Industrial Production Index).
    #
    # Method: 6-month annualised % change, mapped onto the 0–100 PMI scale:
    #   pmi_proxy = 50 + (annualised_growth * 2)
    #   flat production → 50 (neutral), +2% annualised → 54, -2% → 46
    #
    # For the MoM acceleration signal, we compare pmi_proxy this month to last.
    indpro = fetch_fred_series("INDPRO", periods=12)
    indpro_6m_pct = float(indpro.pct_change(6).dropna().iloc[-1] * 100)
    pmi_proxy = round(50 + (indpro_6m_pct * 2 * 2), 1)

    # PMI MoM change: compare this month's proxy to last month's proxy
    # Positive = PMI rising = growth accelerating
    indpro_6m_prev = float(indpro.pct_change(6).dropna().iloc[-2] * 100)
    pmi_proxy_prev = round(50 + (indpro_6m_prev * 2 * 2), 1)
    pmi_mom_change = round(pmi_proxy - pmi_proxy_prev, 2)

    # --- Initial Jobless Claims (ICSA) ---
    # Weekly series. Negative WoW change = fewer people filing for unemployment
    # = labour market tightening = growth signal.
    # We fetch 4 weeks to ensure we have at least 2 valid readings.
    claims = fetch_fred_series("ICSA", periods=4)
    claims_wow_change = float(claims.iloc[-1] - claims.iloc[-2])
    # Expressed in thousands (ICSA reports in thousands of people)

    # --- 2s10s Yield Spread change ---
    # We look at whether the spread is widening (steepening) or narrowing.
    # A steepening curve = markets pricing in better growth ahead.
    # We compare the latest reading to 4 weeks ago (T10Y2Y is daily).
    spread_2s10s = fetch_fred_series("T10Y2Y", periods=30)
    spread_10y2y_change = round(
        float(spread_2s10s.iloc[-1]) - float(spread_2s10s.iloc[-20]), 3
    )
    # iloc[-20] ≈ 4 weeks ago on a daily series

    # =========================================================================
    # INFLATION SIGNALS
    # =========================================================================

    # --- CPI YoY (current and lag for acceleration check) ---
    # We compare current CPI YoY to 3 months ago. If it's higher, inflation
    # is re-accelerating — more dangerous than a stable high number.
    cpi = fetch_fred_series("CPIAUCSL", periods=18)
    cpi_yoy_series = cpi.pct_change(12) * 100  # 12-month % change = YoY
    cpi_yoy = round(float(cpi_yoy_series.dropna().iloc[-1]), 3)
    cpi_yoy_lag = round(float(cpi_yoy_series.dropna().iloc[-4]), 3)
    # iloc[-4] = 3 months ago (monthly series)

    # --- PPI MoM ---
    # Producer Price Index month-on-month % change.
    # PPI leads CPI — producers raise prices first, then pass them to consumers.
    ppi = fetch_fred_series("PPIACO", periods=3)
    ppi_mom = round(float(ppi.pct_change().dropna().iloc[-1] * 100), 3)

    # --- 5Y5Y Breakeven (current and lag) ---
    # The bond market's expectation of average inflation between 5 and 10 years
    # from now. Rising breakevens = market losing confidence in the Fed.
    # T5YIFR is daily, so we use iloc[-1] and iloc[-60] (≈3 months ago).
    be = fetch_fred_series("T5YIFR", periods=70)
    breakeven_5y5y = round(float(be.iloc[-1]), 3)
    breakeven_5y5y_lag = round(float(be.iloc[-60]), 3)

    # --- Michigan 1Y Inflation Expectations ---
    # Monthly consumer survey. Above 3% = elevated expectations.
    # Self-fulfilling: high expectations → wage demands → actual inflation.
    mich = fetch_fred_series("MICH", periods=3)
    michigan_exp = round(float(mich.iloc[-1]), 2)

    # =========================================================================
    # MONETARY CYCLE SIGNALS (Layer 2)
    # =========================================================================

    # --- Fed Funds Rate trend ---
    # Fetch 14 months so we can calculate 1-month, 6-month, and 12-month changes.
    # The direction and pace of Fed Funds tells us where we are in the cycle.
    ff = fetch_fred_series("FEDFUNDS", periods=14)
    fed_funds_current = round(float(ff.iloc[-1]), 2)

    # 1-month change: is the Fed actively hiking or cutting right now?
    # Non-zero = the FOMC just moved rates at their last meeting.
    fed_funds_1m_change = round(float(ff.iloc[-1] - ff.iloc[-2]), 2)

    # 6-month change: tells us the direction of the current cycle.
    # Positive = hiking cycle, negative = cutting cycle, near zero = on hold.
    fed_funds_6m_change = round(float(ff.iloc[-1] - ff.iloc[-7]), 2)

    # 12-month high: the peak rate over the past year.
    # If current rate is near this high, the Fed is at or near peak tightening.
    # If current rate is well below this high, cuts have begun (early/full easing).
    fed_funds_12m_high = round(float(ff.tail(12).max()), 2)

    # --- Chicago Fed NFCI (National Financial Conditions Index) ---
    # Weekly index measuring tightness of US financial conditions.
    # Negative = loose (easy to borrow, credit abundant, spreads tight)
    # Positive = tight (hard to borrow, spreads wide, credit contracting)
    # This captures the real-world impact of Fed policy beyond just the rate level.
    nfci_series = fetch_fred_series("NFCI", periods=5)
    nfci = round(float(nfci_series.iloc[-1]), 3)

    # --- 10yr Real Yield direction ---
    # Rising real yields = financial conditions tightening (bonds getting cheaper
    # in real terms). Falling real yields = easing impulse.
    # We compare current to 3 months ago to get the direction of travel.
    real_yield_series = fetch_fred_series("DFII10", periods=70)
    real_yield_current = round(float(real_yield_series.iloc[-1]), 2)
    real_yield_3m_ago  = round(float(real_yield_series.iloc[-60]), 2)
    # iloc[-60] ≈ 3 months ago on a daily series

    return {
        # Growth
        "pmi_proxy":           pmi_proxy,
        "pmi_mom_change":      pmi_mom_change,
        "claims_wow_change":   claims_wow_change,
        "spread_10y2y_change": spread_10y2y_change,
        # Inflation
        "cpi_yoy":             cpi_yoy,
        "cpi_yoy_lag":         cpi_yoy_lag,
        "ppi_mom":             ppi_mom,
        "breakeven_5y5y":      breakeven_5y5y,
        "breakeven_5y5y_lag":  breakeven_5y5y_lag,
        "michigan_exp":        michigan_exp,
        # Monetary cycle (Layer 2)
        "fed_funds_current":   fed_funds_current,
        "fed_funds_1m_change": fed_funds_1m_change,
        "fed_funds_6m_change": fed_funds_6m_change,
        "fed_funds_12m_high":  fed_funds_12m_high,
        "nfci":                nfci,
        "real_yield_current":  real_yield_current,
        "real_yield_3m_ago":   real_yield_3m_ago,
    }


# -----------------------------------------------------------------------------
# fetch_kpi_data()
# Returns the four headline numbers for the KPI metric cards at the top of
# the dashboard: CPI YoY, Fed Funds, 2s10s spread, 10yr Real Yield.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_kpi_data() -> dict[str, float]:
    """Return current values for the four dashboard KPI metric cards."""
    cpi = fetch_fred_series("CPIAUCSL", periods=14)
    cpi_yoy = float(cpi.pct_change(12).dropna().iloc[-1] * 100)

    fed_funds = fetch_fred_series("FEDFUNDS", periods=2)
    fed_funds_rate = float(fed_funds.iloc[-1])

    spread = fetch_fred_series("T10Y2Y", periods=5)
    spread_2s10s = float(spread.iloc[-1])

    # 10yr Real Yield (TIPS): positive = earning real return above inflation,
    # negative = losing purchasing power even in Treasuries.
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
# Returns two yield curve snapshots: today and one year ago.
# Used for the yield curve chart — overlaying past vs present shows whether
# the curve has steepened, flattened, or inverted over the year.
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
        # Fetch ~18 months to cover both today and 1yr ago in one API call
        data = fetch_fred_series(series_id, periods=400)
        current_yields[label] = float(data.iloc[-1])

        # Find nearest available trading day on or before one_year_ago
        past_data = data[data.index <= pd.Timestamp(one_year_ago)]
        year_ago_yields[label] = (
            float(past_data.iloc[-1]) if not past_data.empty else float("nan")
        )

    return pd.Series(current_yields), pd.Series(year_ago_yields)


# -----------------------------------------------------------------------------
# fetch_cpi_trend()
# Returns 24 months of CPI and Core CPI as year-on-year % changes.
# Used for the CPI trend chart with the 2% Fed target line overlaid.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_cpi_trend(months: int = 24) -> pd.DataFrame:
    """
    Return monthly CPI and Core CPI YoY % for the last N months.
    Columns: ['CPI YoY', 'Core CPI YoY']. Index: monthly DatetimeIndex.
    """
    extra = 14  # Extra periods needed to calculate 12-month % change
    cpi      = fetch_fred_series("CPIAUCSL", periods=months + extra)
    core_cpi = fetch_fred_series("CPILFESL", periods=months + extra)

    df = pd.DataFrame({
        "CPI YoY":      cpi.pct_change(12) * 100,
        "Core CPI YoY": core_cpi.pct_change(12) * 100,
    }).dropna().tail(months)

    return df


# -----------------------------------------------------------------------------
# fetch_market_snapshot()
# Downloads ~1 year of daily prices for all tickers via yfinance and
# calculates % changes over four windows: 1D, 1W, 1M, YTD.
#
# Includes RORO assets (HYG, VIX) alongside the main market snapshot assets.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_snapshot() -> pd.DataFrame:
    """
    Return current price and 1D/1W/1M/YTD % changes for all tickers.
    Returns a DataFrame indexed by asset name.
    """
    # Market snapshot assets only (exclude VIX from price table — it's an index)
    snapshot_tickers = {k: v for k, v in TICKERS.items() if k != "VIX"}
    rows: list[dict] = []

    for name, ticker in snapshot_tickers.items():
        raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

        if raw.empty:
            continue

        # yfinance returns a DataFrame for single tickers — squeeze to 1D Series
        prices = raw["Close"].squeeze().dropna()

        if len(prices) < 2:
            continue

        current_price = float(prices.iloc[-1])

        def pct_vs(n_days: int) -> float:
            """Return % change from n trading days ago to today."""
            if len(prices) < n_days + 1:
                return float("nan")
            return float((prices.iloc[-1] / prices.iloc[-(n_days + 1)] - 1) * 100)

        # YTD: % change from last trading day of prior year to today
        year_start = pd.Timestamp(datetime.today().year, 1, 1)
        ytd_base = prices[prices.index < year_start]
        ytd_pct = (
            float((prices.iloc[-1] / ytd_base.iloc[-1] - 1) * 100)
            if not ytd_base.empty else float("nan")
        )

        rows.append({
            "Asset": name,
            "Price": round(current_price, 2),
            "1D %":  round(pct_vs(1), 2),
            "1W %":  round(pct_vs(5), 2),    # 5 trading days ≈ 1 week
            "1M %":  round(pct_vs(21), 2),   # 21 trading days ≈ 1 month
            "YTD %": round(ytd_pct, 2),
        })

    return pd.DataFrame(rows).set_index("Asset")


# -----------------------------------------------------------------------------
# fetch_roro_signals()
# Returns the raw data needed for the Layer 3 RORO classification.
# VIX level + 5d change, DXY, Gold/SPY ratio, HYG price, EEM vs SPY.
# Not yet wired into the classifier — consumed by the RORO stub in classifier.py.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_roro_signals() -> dict[str, float]:
    """
    Return current RORO signal values for Layer 3 classification.
    VIX, DXY, HYG, Gold/SPY ratio, EEM/SPY relative performance.
    """
    roro_tickers = {
        "^VIX":   "vix",
        "DX-Y.NYB": "dxy",
        "HYG":    "hyg",
        "GC=F":   "gold",
        "^GSPC":  "spy",
        "EEM":    "eem",
    }

    prices: dict[str, pd.Series] = {}
    for ticker, name in roro_tickers.items():
        raw = yf.download(ticker, period="1mo", auto_adjust=True, progress=False)
        if not raw.empty:
            prices[name] = raw["Close"].squeeze().dropna()

    signals: dict[str, float] = {}

    # VIX level + 5-day point change (e.g. +3.2 = fear rising = risk-off signal)
    if "vix" in prices and len(prices["vix"]) >= 6:
        signals["vix_level"]     = round(float(prices["vix"].iloc[-1]), 2)
        signals["vix_5d_change"] = round(
            float(prices["vix"].iloc[-1] - prices["vix"].iloc[-6]), 2
        )

    # DXY 5-day % change: USD strengthening = safe-haven bid = risk-off
    if "dxy" in prices and len(prices["dxy"]) >= 6:
        signals["dxy_5d_change"] = round(
            float((prices["dxy"].iloc[-1] / prices["dxy"].iloc[-6] - 1) * 100), 3
        )

    # HYG price level (for display) + 5-day % change: falling HYG = credit spreads
    # widening = risk-off (investors dumping high-yield bonds)
    if "hyg" in prices:
        signals["hyg_price"] = round(float(prices["hyg"].iloc[-1]), 2)
        if len(prices["hyg"]) >= 6:
            signals["hyg_5d_change"] = round(
                float((prices["hyg"].iloc[-1] / prices["hyg"].iloc[-6] - 1) * 100), 3
            )

    # Gold/SPY ratio: current level + 5-day % change.
    # Rising ratio = gold outperforming equities = classic risk-off signature.
    if "gold" in prices and "spy" in prices:
        ratio_now = float(prices["gold"].iloc[-1] / prices["spy"].iloc[-1])
        signals["gold_spy_ratio"] = round(ratio_now, 4)
        if len(prices["gold"]) >= 6 and len(prices["spy"]) >= 6:
            ratio_5d_ago = float(prices["gold"].iloc[-6] / prices["spy"].iloc[-6])
            signals["gold_spy_ratio_5d_change"] = round(
                (ratio_now / ratio_5d_ago - 1) * 100, 3
            )

    # EEM vs SPY 5-day relative return: negative = EM underperforming US = risk-off
    if "eem" in prices and "spy" in prices and len(prices["eem"]) >= 6:
        eem_ret = float(prices["eem"].iloc[-1] / prices["eem"].iloc[-6] - 1)
        spy_ret = float(prices["spy"].iloc[-1] / prices["spy"].iloc[-6] - 1)
        signals["eem_vs_spy_5d"] = round((eem_ret - spy_ret) * 100, 3)

    return signals


# -----------------------------------------------------------------------------
# fetch_regime_price_history()
#
# The data backbone for the historical performance heatmap. Fetches long-run
# monthly price history for all six regime assets from multiple sources:
#
#   Gold  — FRED GOLDAMGBD228NLBM (London AM fixing, daily from 1968)
#   Oil   — FRED WTISPLC (WTI spot price, monthly from 1946)
#   SPX   — yfinance ^GSPC
#   TLT   — yfinance TLT (from 2002-07; no pre-2002 proxy — see episodes.py)
#   DXY   — yfinance DX-Y.NYB
#   EM    — yfinance VEIEX (1994-2003) stitched to EEM (2003-present)
#
# Returns a single unified DataFrame resampled to month-end, which
# compute_regime_returns() slices into canonical episode windows.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_regime_price_history() -> pd.DataFrame:
    """
    Return monthly historical prices for all six regime assets (1946–present).
    Columns: ['Gold', 'Oil', 'SPX', 'TLT', 'DXY', 'EM']. Index: month-end dates.
    """
    fred = _get_fred_client()

    # -------------------------------------------------------------------------
    # FRED: Gold and Oil — attempt full-history fetch, fall back to yfinance.
    #
    # GOLDAMGBD228NLBM (London AM fixing, from 1968) and WTISPLC (WTI spot,
    # from 1946) are ideal for covering the 1970s stagflation episodes, but
    # some FRED series IDs visible on the website are locked behind data
    # licensing agreements and return 400 from the API. If either fetch fails,
    # we fall back to yfinance GC=F / CL=F and accept reduced historical depth.
    # The episode-slicing logic in compute_regime_returns() handles missing data
    # gracefully (empty slices are skipped, n=0 triggers the hardcoded fallback).
    # -------------------------------------------------------------------------
    try:
        gold_daily   = fred.get_series("GOLDAMGBD228NLBM").dropna()
        gold_monthly = gold_daily.resample("ME").last()
    except Exception:
        gold_monthly = pd.Series(dtype=float)   # populated from yfinance GC=F below

    try:
        oil_raw      = fred.get_series("WTISPLC").dropna()
        oil_monthly  = oil_raw.resample("ME").last()
    except Exception:
        oil_monthly = pd.Series(dtype=float)    # populated from yfinance CL=F below

    # -------------------------------------------------------------------------
    # yfinance: SPX, TLT, DXY, VEIEX, EEM — plus GC=F and CL=F as fallbacks
    # for Gold and Oil if the FRED series above were unavailable.
    # -------------------------------------------------------------------------
    yf_tickers = ["^GSPC", "TLT", "DX-Y.NYB", "VEIEX", "EEM", "GC=F", "CL=F"]
    raw = yf.download(
        yf_tickers,
        start="1970-01-01",
        auto_adjust=True,
        progress=False,
    )

    # Multi-ticker download gives MultiIndex columns: (price_type, ticker).
    # Extracting "Close" gives a single-level DataFrame with ticker symbols.
    closes = raw["Close"]
    closes_monthly = closes.resample("ME").last()

    # Helper to safely extract a column from the monthly closes DataFrame
    def _col(ticker: str) -> pd.Series:
        return closes_monthly[ticker] if ticker in closes_monthly.columns else pd.Series(dtype=float)

    # -------------------------------------------------------------------------
    # EM series: stitch VEIEX (pre-2003-04) → EEM (post-2003-04).
    # Normalise EEM to VEIEX level at the stitch month-end so there's no
    # artificial level jump — preserving relative returns in both eras.
    #
    # Why this matters: if we didn't normalise, the level discontinuity at the
    # stitch date would distort returns for any episode that starts immediately
    # after the join. (None of our current episodes span the join, but it's
    # correct practice regardless.)
    # -------------------------------------------------------------------------
    stitch_ts = pd.Timestamp("2003-04-14") + pd.offsets.MonthEnd(0)

    veiex = _col("VEIEX")
    eem   = _col("EEM")

    if not veiex.empty and not eem.empty:
        # Compute scale factor so EEM equals VEIEX at the stitch month
        v_level = veiex.get(stitch_ts)
        e_level = eem.get(stitch_ts)
        scale = (v_level / e_level) if (pd.notna(v_level) and pd.notna(e_level) and e_level != 0) else 1.0

        em_series = veiex.copy()
        eem_scaled = eem * scale
        # Overwrite with scaled EEM for all months from the stitch date onward
        em_series.update(eem_scaled[eem_scaled.index >= stitch_ts])
    else:
        em_series = veiex if not veiex.empty else eem

    # -------------------------------------------------------------------------
    # Assemble and return the unified monthly price DataFrame.
    # Use FRED series for Gold/Oil where available; yfinance as fallback.
    # -------------------------------------------------------------------------
    gold_col = gold_monthly if not gold_monthly.empty else _col("GC=F")
    oil_col  = oil_monthly  if not oil_monthly.empty  else _col("CL=F")

    df = pd.DataFrame({
        "Gold": gold_col,
        "Oil":  oil_col,
        "SPX":  _col("^GSPC"),
        "TLT":  _col("TLT"),
        "DXY":  _col("DX-Y.NYB"),
        "EM":   em_series,
    })

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# -----------------------------------------------------------------------------
# compute_regime_returns()
#
# Takes the monthly price history from fetch_regime_price_history() and the
# canonical episode windows from regime/episodes.py, and computes average
# annualised returns per (regime, asset).
#
# This is a pure compute function — no network calls, no caching needed.
# The expensive I/O is already cached upstream in fetch_regime_price_history().
#
# Annualisation formula (per episode):
#   annualised_return = ((end_price / start_price) ^ (365.25 / days)) - 1
#
# Averaging: simple arithmetic mean across valid episodes. Each episode counts
# once regardless of its duration — this is labelled regime attribution, not
# time-weighted portfolio accounting.
#
# Fallback: if an asset has zero valid episodes for a regime (e.g. TLT in
# the pre-2002 stagflation episodes), the hardcoded REGIME_RETURNS value is
# used so the heatmap always shows something.
# -----------------------------------------------------------------------------
def compute_regime_returns(
    prices: pd.DataFrame,
    fallback: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """
    Compute average annualised returns per regime from historical episode windows.

    Args:
        prices:   Monthly price DataFrame from fetch_regime_price_history().
        fallback: Hardcoded REGIME_RETURNS dict — used when n=0 for an asset.

    Returns:
        regime_returns  — {regime: {asset: avg_annualised_pct}}
        episode_counts  — {regime: {asset: n_valid_episodes}}
    """
    from regime.episodes import REGIME_EPISODES, ASSET_SOURCES

    assets  = list(ASSET_SOURCES.keys())
    regimes = list(REGIME_EPISODES.keys())

    regime_returns: dict[str, dict[str, float]] = {}
    episode_counts: dict[str, dict[str, int]]   = {}

    for regime in regimes:
        regime_returns[regime] = {}
        episode_counts[regime] = {}
        episodes = REGIME_EPISODES[regime]

        for asset in assets:
            available_from = pd.Timestamp(ASSET_SOURCES[asset]["available_from"])

            if asset not in prices.columns:
                # Column missing entirely — use fallback
                regime_returns[regime][asset] = fallback.get(regime, {}).get(asset, 0.0)
                episode_counts[regime][asset] = 0
                continue

            price_series = prices[asset].dropna()
            valid_returns: list[float] = []

            for (start_str, end_str) in episodes:
                start_ts = pd.Timestamp(start_str)
                # end_str is month-start (e.g. "2009-06") — push to month-end
                end_ts = pd.Timestamp(end_str) + pd.offsets.MonthEnd(0)

                # Skip episodes that pre-date this asset's data availability
                if start_ts < available_from:
                    continue

                # Slice to the episode window and drop any NaN gaps
                episode_prices = price_series.loc[
                    (price_series.index >= start_ts) &
                    (price_series.index <= end_ts)
                ].dropna()

                if len(episode_prices) < 2:
                    continue

                start_price = float(episode_prices.iloc[0])
                end_price   = float(episode_prices.iloc[-1])

                if start_price <= 0:
                    continue

                # Use actual calendar span between first and last observed price
                # (not the nominal episode dates) so short data gaps don't inflate
                # annualised returns
                days = (episode_prices.index[-1] - episode_prices.index[0]).days
                if days <= 0:
                    continue

                total_return = end_price / start_price - 1
                annualised   = ((1 + total_return) ** (365.25 / days) - 1) * 100
                valid_returns.append(annualised)

            if not valid_returns:
                # No valid episodes — fall back to hardcoded seed value
                regime_returns[regime][asset] = fallback.get(regime, {}).get(asset, 0.0)
                episode_counts[regime][asset] = 0
            else:
                avg = sum(valid_returns) / len(valid_returns)
                regime_returns[regime][asset] = round(avg, 1)
                episode_counts[regime][asset] = len(valid_returns)

    return regime_returns, episode_counts


# -----------------------------------------------------------------------------
# compute_episode_returns()
#
# Like compute_regime_returns() but returns one row per individual episode
# rather than regime-level averages. Used for the episode detail table that
# shows the specific named events (GFC, COVID crash, Volcker era, etc.) and
# the actual annualised return of each asset during that episode.
#
# Also appends a regime-average row after each group so the table contains
# both granular and summary information in one view.
#
# Returns a list of dicts, each representing one table row:
#   {
#     "regime":     "Deflation/Bust",
#     "name":       "Global Financial Crisis",
#     "period":     "Dec 2007 – Jun 2009",
#     "is_average": False,
#     "Gold": 18.4, "Oil": -52.1, "SPX": -38.6, "TLT": 28.3, "DXY": 7.1, "EM": -47.2
#   }
# Values are annualised % returns; None means no data for that asset/episode.
# -----------------------------------------------------------------------------
def compute_episode_returns(
    prices: pd.DataFrame,
    fallback: dict[str, dict[str, float]],
) -> list[dict]:
    """
    Return per-episode annualised returns for the episode detail table.
    Appends a regime-average row after each episode group.
    """
    from datetime import datetime
    from regime.episodes import REGIME_EPISODES, EPISODE_NAMES, ASSET_SOURCES

    assets  = list(ASSET_SOURCES.keys())
    rows: list[dict] = []

    def _fmt_period(start_str: str, end_str: str) -> str:
        """Convert '1973-10'/'1975-03' to 'Oct 1973 – Mar 1975'."""
        s = datetime.strptime(start_str, "%Y-%m").strftime("%b %Y")
        e = datetime.strptime(end_str,   "%Y-%m").strftime("%b %Y")
        return f"{s} – {e}"

    for regime, episodes in REGIME_EPISODES.items():
        names = EPISODE_NAMES.get(regime, [])
        regime_sums:  dict[str, list[float]] = {a: [] for a in assets}

        for i, (start_str, end_str) in enumerate(episodes):
            name      = names[i] if i < len(names) else f"Episode {i + 1}"
            period    = _fmt_period(start_str, end_str)
            start_ts  = pd.Timestamp(start_str)
            end_ts    = pd.Timestamp(end_str) + pd.offsets.MonthEnd(0)
            row: dict = {"regime": regime, "name": name, "period": period, "is_average": False}

            for asset in assets:
                available_from = pd.Timestamp(ASSET_SOURCES[asset]["available_from"])
                value = None  # type: Optional[float]

                if asset in prices.columns and start_ts >= available_from:
                    series = prices[asset].dropna()
                    ep     = series.loc[(series.index >= start_ts) & (series.index <= end_ts)].dropna()

                    if len(ep) >= 2 and ep.iloc[0] > 0:
                        days = (ep.index[-1] - ep.index[0]).days
                        if days > 0:
                            total  = ep.iloc[-1] / ep.iloc[0] - 1
                            value  = round(((1 + total) ** (365.25 / days) - 1) * 100, 1)

                row[asset] = value
                if value is not None:
                    regime_sums[asset].append(value)

            rows.append(row)

        # Regime-average summary row
        avg_row: dict = {"regime": regime, "name": "Regime Average", "period": "", "is_average": True}
        for asset in assets:
            vals = regime_sums[asset]
            if vals:
                avg_row[asset] = round(sum(vals) / len(vals), 1)
            else:
                avg_row[asset] = fallback.get(regime, {}).get(asset)
        rows.append(avg_row)

    return rows


# -----------------------------------------------------------------------------
# Quick test — run `python3 data/fetcher.py` to verify live data is coming in.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Macro Inputs (all classifier signals) ---")
    inputs = fetch_macro_inputs()
    for k, v in inputs.items():
        print(f"  {k:<25} {v}")

    print("\n--- KPI Data ---")
    print(fetch_kpi_data())

    print("\n--- CPI Trend (last 6 months) ---")
    print(fetch_cpi_trend(months=6))

    print("\n--- Yield Curve ---")
    current, year_ago = fetch_yield_curve()
    print("  Current:   ", current.to_dict())
    print("  1yr ago:   ", year_ago.to_dict())

    print("\n--- Market Snapshot ---")
    print(fetch_market_snapshot())

    print("\n--- RORO Signals ---")
    print(fetch_roro_signals())
