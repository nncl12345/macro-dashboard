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
    "continuing_claims": "CCSA",     # Continuing Claims — stock of unemployed (slower-moving than initial)
    "lei":              "USSLIND",   # Conference Board Leading Economic Index
    "retail_sales":     "RSXFS",     # Retail Sales ex Food Services (consumer demand)
    "wei":              "WEI",       # NY Fed Weekly Economic Index (real-time GDP nowcast)
    "new_orders":       "NEWORDER",  # Core capital goods orders (ex defence, ex aircraft)
                                     # — leading-side complement to INDPRO; INDPRO captures
                                     # production, NEWORDER captures the order book behind it.

    # --- Net liquidity (Fed balance-sheet quantity-of-money read) ---
    "walcl":            "WALCL",     # Fed total assets (weekly H.4.1, $M)
    "tga":              "WTREGEN",   # Treasury General Account (weekly, $M) — cash sitting outside system
    "rrp":              "RRPONTSYD", # Overnight Reverse Repo balance (daily, $B) — MMF cash parked at Fed
    # Net liquidity = WALCL − TGA − RRP. Rising = stealth easing, falling = stealth tightening.

    # --- Funding-market stress (gating override) ---
    "sofr":             "SOFR",      # Secured Overnight Financing Rate (from 2018-04)
    "iorb":             "IORB",      # Interest on Reserve Balances (from 2021-07; pre-2021 = IOER)
    # SOFR > IORB by >5bp = dealer balance sheets full = funding stress = forces "Peak Tightening".

    # --- Credit / risk overlay ---
    "hy_oas":           "BAMLH0A0HYM2",  # ICE BofA US HY OAS (daily, from 1996) — replaces HYG price as RORO signal

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
# fetch_historical_panel()
# Pulls the FULL history of every series the classifier uses, from 1965 onward.
# Used by the backtest — one-shot cached fetch so we don't hit the FRED API
# 13× (once per historical episode).
#
# Cache TTL is 24h: historical data doesn't change meaningfully intra-day, and
# this fetch is ~14 API calls so we want to avoid repeating it on every page load.
# -----------------------------------------------------------------------------
HISTORICAL_SERIES: list[str] = [
    "INDPRO",     # PMI proxy source
    "USSLIND",    # LEI (from 1982)
    "ICSA",       # Initial claims (from 1967)
    "DGS10",      # 10y Treasury
    "DGS2",       # 2y Treasury (from 1976)
    "T10Y2Y",     # 2s10s spread (from 1976)
    "CPIAUCSL",   # Headline CPI
    "CPILFESL",   # Core CPI (from 1957)
    "PPIACO",     # PPI
    "T5YIFR",     # 5Y5Y breakeven (from 2003)
    "MICH",       # Michigan expectations (from 1978)
    "FEDFUNDS",   # Fed Funds rate
    "NFCI",       # Chicago Fed financial conditions (from 1971)
    "DFII10",     # 10y TIPS real yield (from 2003)
    "UNRATE",     # Unemployment rate — Sahm-rule recession detector
]


@st.cache_data(ttl=86400)   # 24h — historical data doesn't change often
def fetch_historical_panel(start: str = "1965-01-01") -> dict[str, pd.Series]:
    """Fetch the full history of every classifier-relevant FRED series.

    Returns a dict of series_id → pd.Series. Series that don't exist until a
    later start date will naturally begin partway through the returned range.
    The backtest uses `.loc[:date]` slicing to get point-in-time data.
    """
    fred = _get_fred_client()
    panel: dict[str, pd.Series] = {}
    for sid in HISTORICAL_SERIES:
        series = fred.get_series(sid, observation_start=start)
        panel[sid] = series.dropna()
    return panel


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

    # --- Conference Board LEI (Leading Economic Index) MoM ---
    # USSLIND is a composite of 10 forward-looking indicators (claims, building
    # permits, new orders, yield curve, equities, etc). MoM > 0 = leading
    # indicators point to growth ahead. Used in place of a second PMI signal to
    # avoid double-counting INDPRO — LEI captures a broader forward view.
    lei = fetch_fred_series("USSLIND", periods=3)
    lei_mom = round(float(lei.pct_change().dropna().iloc[-1] * 100), 3)

    # --- Initial Jobless Claims (ICSA) — 4-week moving average trend ---
    # Weekly series. Raw WoW changes are dominated by holiday/seasonal noise,
    # so we compare the latest 4-week average to the prior 4-week average.
    # Negative = claims trending down = labour market tightening = growth signal.
    claims = fetch_fred_series("ICSA", periods=8)
    claims_4w_now  = float(claims.iloc[-4:].mean())
    claims_4w_prev = float(claims.iloc[-8:-4].mean())
    claims_trend_change = round(claims_4w_now - claims_4w_prev, 0)
    # Expressed in level of claims (ICSA reports actual count, not thousands)

    # --- Continuing Claims (CCSA) — stock of unemployed ---
    # Initial claims captures the inflow into unemployment; CCSA captures the
    # *stock* — people who've filed and still can't find work. Together they
    # form a 2-vote labour basket: initial says "are people losing jobs?",
    # continuing says "are they getting hired back?". Same 4w/4w smoothing.
    cc = fetch_fred_series("CCSA", periods=8)
    cc_4w_now   = float(cc.iloc[-4:].mean())
    cc_4w_prev  = float(cc.iloc[-8:-4].mean())
    continuing_claims_trend_change = round(cc_4w_now - cc_4w_prev, 0)

    # --- Core capital goods orders (NEWORDER) — 3-month % change ---
    # The most forward-looking macro signal that's free and FRED-native.
    # NEWORDER = Manufacturers' New Orders for Nondefense Capital Goods
    # excluding aircraft. Strips out the noisy bits (Boeing one-offs, defence
    # spending) and leaves you with what businesses are actually committing to
    # spend on equipment. This is the institutional read on capex intent.
    #
    # Why both this AND INDPRO: INDPRO captures realised production; NEWORDER
    # captures the order book backing the next quarter of production. INDPRO
    # is a proxy for the PMI *production* sub-index; NEWORDER fills the
    # *new orders* sub-index gap that INDPRO doesn't see.
    #
    # 3-month % change to filter single-month noise (NEWORDER is volatile MoM).
    no = fetch_fred_series("NEWORDER", periods=8)
    new_orders_3m_chg_pct = round(float(no.pct_change(3).dropna().iloc[-1] * 100), 2)

    # --- WEI (Weekly Economic Index, NY Fed) ---
    # Composite of 10 weekly indicators (claims, retail, fuel, payroll tax,
    # steel, etc.) scaled to look like real GDP growth. Real-time nowcast —
    # complements monthly INDPRO with a weekly-cadence growth read.
    # Vote: WEI > its 4-week MA = growth accelerating.
    wei = fetch_fred_series("WEI", periods=8)
    wei_current = round(float(wei.iloc[-1]), 3)
    wei_4w_avg  = round(float(wei.iloc[-4:].mean()), 3)

    # --- 2yr and 10yr yield changes (for bear-steepener disambiguation) ---
    # A "steepening" yield curve can be:
    #   - Bull steepener: 2y falls more than 10y (Fed-cut fears → recession signal)
    #   - Bear steepener: 10y rises more than 2y (growth/inflation repricing higher)
    # Only the bear steepener is a real growth signal, so we track the 2y and
    # 10y moves separately and classify in the classifier.
    y10 = fetch_fred_series("DGS10", periods=30).dropna()
    y2  = fetch_fred_series("DGS2",  periods=30).dropna()
    yield_10y_change = round(float(y10.iloc[-1]) - float(y10.iloc[-20]), 3)
    yield_2y_change  = round(float(y2.iloc[-1])  - float(y2.iloc[-20]),  3)
    # Derived spread change, kept for backward compat with existing UI
    spread_10y2y_change = round(yield_10y_change - yield_2y_change, 3)

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

    # --- Core CPI YoY (current and lag) ---
    # Core CPI strips food and energy — the components headline CPI is most
    # volatile on. This is closer to what the Fed actually reacts to, and
    # a much cleaner gauge of underlying inflation pressure.
    core_cpi = fetch_fred_series("CPILFESL", periods=18)
    core_cpi_yoy_series = core_cpi.pct_change(12) * 100
    core_cpi_yoy     = round(float(core_cpi_yoy_series.dropna().iloc[-1]), 3)
    core_cpi_yoy_lag = round(float(core_cpi_yoy_series.dropna().iloc[-4]), 3)

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

    # --- CPI disinflation override signals ---
    # The framework scores +1 when each momentum signal fires "rising". But in
    # disinflation phases (e.g. late 2022 onward), CPI YoY level may still be
    # elevated even as the direction has clearly turned. These two signals detect
    # "CPI has rolled over from peak" and, when both fire, the classifier zeroes
    # the inflation score to force a disinflation read. Purely directional —
    # stays true to the framework's "rate of change is everything" principle.
    cpi_14 = fetch_fred_series("CPIAUCSL", periods=24)    # need 24 so YoY has 12m of history
    cpi_yoy_14 = (cpi_14.pct_change(12) * 100).dropna()
    cpi_12m_peak     = round(float(cpi_yoy_14.tail(12).max()), 3)
    cpi_peak_gap     = round(cpi_12m_peak - cpi_yoy, 3)    # >=0; positive = off peak
    cpi_rolled_over  = cpi_peak_gap > 0.5                  # at least 0.5pp below 12m high
    cpi_3m_decel     = cpi_yoy < cpi_yoy_lag               # 3m momentum negative

    # --- Sahm-rule recession trigger ---
    # Claudia Sahm's 2019 rule: a recession has started when the 3m moving
    # average of U3 unemployment rises 0.5pp above its trailing 12-month low.
    # Empirically triggered at the onset of every US recession since 1950 with
    # zero false positives. Used as a hard override — if triggered, the regime
    # is forced to Deflation/Bust regardless of the classifier's quadrant math.
    unrate = fetch_fred_series("UNRATE", periods=18)
    unrate_3m = unrate.rolling(3).mean().dropna()
    # Compare latest 3m avg to the minimum 3m avg over the past 12 months
    sahm_current = float(unrate_3m.iloc[-1])
    sahm_12m_min = float(unrate_3m.tail(12).min())
    sahm_trigger = (sahm_current - sahm_12m_min) >= 0.5

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

    # --- Net liquidity (WALCL − TGA − RRP) ---
    # The quantity-of-money read on Fed policy. Fed Funds tells you the *price*
    # of money; net liquidity tells you how much cash is actually circulating
    # in the financial system bidding up assets. Rose sharply in early 2023 as
    # the TGA drained — explained the equity rally despite Fed still hiking.
    #
    # Units: WALCL and TGA are in $M, RRP is in $B. Convert all to $T.
    # Resample to weekly Friday so all three series align (WALCL is Wed-stamped,
    # RRP is daily, TGA is weekly).
    try:
        walcl = fetch_fred_series("WALCL", periods=20).resample("W-FRI").last().ffill() / 1_000_000
        tga   = fetch_fred_series("WTREGEN", periods=20).resample("W-FRI").last().ffill() / 1_000_000
        rrp   = fetch_fred_series("RRPONTSYD", periods=100).resample("W-FRI").last().ffill() / 1_000
        net_liq = (walcl - tga - rrp).dropna()
        net_liq_current  = round(float(net_liq.iloc[-1]), 3)
        net_liq_3m_ago   = round(float(net_liq.iloc[-13]), 3)   # ~13 weeks = 3 months
        net_liq_3m_change_pct = round((net_liq_current / net_liq_3m_ago - 1) * 100, 2)
    except Exception:
        net_liq_current = float("nan")
        net_liq_3m_ago = float("nan")
        net_liq_3m_change_pct = float("nan")

    # --- SOFR − IORB spread (funding stress canary) ---
    # SOFR should trade *below* IORB in normal times — IORB is the risk-free
    # ceiling. When SOFR spikes above IORB, it means dealers are paying a
    # premium for cash because big banks have stopped lending out their
    # reserves (balance sheets full, regulatory ratios constrained).
    # Sept 2019 repo crisis was telegraphed here. Gating override on Layer 2.
    try:
        sofr = fetch_fred_series("SOFR", periods=10).dropna()
        iorb = fetch_fred_series("IORB", periods=10).dropna()
        sofr_iorb_spread_bp = round((float(sofr.iloc[-1]) - float(iorb.iloc[-1])) * 100, 2)
    except Exception:
        sofr_iorb_spread_bp = float("nan")

    # --- MOVE Index (Treasury vol) ---
    # Bond-market equivalent of VIX. When MOVE spikes, *everything* reprices —
    # duration, equity multiples, FX carry. Fed-cycle stress shows up here
    # before VIX. Vote: MOVE > 12m average = tight/uncertain conditions.
    # Yahoo coverage of ^MOVE is sometimes spotty; fall back to NaN if missing.
    try:
        move_raw = yf.download("^MOVE", period="1y", auto_adjust=True, progress=False)
        move_close = move_raw["Close"].squeeze().dropna() if not move_raw.empty else pd.Series(dtype=float)
        if len(move_close) >= 60:
            move_current = round(float(move_close.iloc[-1]), 2)
            move_12m_avg = round(float(move_close.mean()), 2)
        else:
            move_current = float("nan")
            move_12m_avg = float("nan")
    except Exception:
        move_current = float("nan")
        move_12m_avg = float("nan")

    return {
        # Growth
        "pmi_proxy":           pmi_proxy,
        "pmi_mom_change":      pmi_mom_change,         # kept for reference, not used in classifier
        "lei_mom":             lei_mom,
        "claims_trend_change": claims_trend_change,
        "continuing_claims_trend_change": continuing_claims_trend_change,
        "new_orders_3m_chg_pct": new_orders_3m_chg_pct,
        "wei_current":         wei_current,
        "wei_4w_avg":          wei_4w_avg,
        "spread_10y2y_change": spread_10y2y_change,
        "yield_10y_change":    yield_10y_change,
        "yield_2y_change":     yield_2y_change,
        # Inflation
        "cpi_yoy":             cpi_yoy,
        "cpi_yoy_lag":         cpi_yoy_lag,
        "core_cpi_yoy":        core_cpi_yoy,
        "core_cpi_yoy_lag":    core_cpi_yoy_lag,
        "ppi_mom":             ppi_mom,
        "breakeven_5y5y":      breakeven_5y5y,
        "breakeven_5y5y_lag":  breakeven_5y5y_lag,
        "michigan_exp":        michigan_exp,
        # Disinflation + recession override signals (v3 classifier)
        "cpi_rolled_over":     cpi_rolled_over,
        "cpi_3m_decel":        cpi_3m_decel,
        "cpi_peak_gap":        cpi_peak_gap,
        "sahm_trigger":        sahm_trigger,
        # Monetary cycle (Layer 2)
        "fed_funds_current":   fed_funds_current,
        "fed_funds_1m_change": fed_funds_1m_change,
        "fed_funds_6m_change": fed_funds_6m_change,
        "fed_funds_12m_high":  fed_funds_12m_high,
        "nfci":                nfci,
        "real_yield_current":  real_yield_current,
        "real_yield_3m_ago":   real_yield_3m_ago,
        # Net liquidity + funding stress + rates vol (Tier 1 Layer 2 adds)
        "net_liq_current":      net_liq_current,
        "net_liq_3m_change_pct": net_liq_3m_change_pct,
        "sofr_iorb_spread_bp":  sofr_iorb_spread_bp,
        "move_current":         move_current,
        "move_12m_avg":         move_12m_avg,
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

    # HYG price level (for display in market snapshot table; no longer used as
    # a RORO vote — replaced by HY OAS spread below, which is the institutional
    # standard and reads in basis points instead of dollar prices).
    if "hyg" in prices:
        signals["hyg_price"] = round(float(prices["hyg"].iloc[-1]), 2)
        if len(prices["hyg"]) >= 6:
            signals["hyg_5d_change"] = round(
                float((prices["hyg"].iloc[-1] / prices["hyg"].iloc[-6] - 1) * 100), 3
            )

    # HY OAS — ICE BofA US High Yield Option-Adjusted Spread (FRED daily).
    # Direct credit-spread read in bp. Widening = risk-off (credit market
    # pricing in higher default risk). Replaces HYG as the Layer 3 credit vote.
    try:
        hy_oas_series = fetch_fred_series("BAMLH0A0HYM2", periods=15).dropna()
        if len(hy_oas_series) >= 6:
            signals["hy_oas_current"]   = round(float(hy_oas_series.iloc[-1]), 2)
            signals["hy_oas_5d_change"] = round(
                float(hy_oas_series.iloc[-1] - hy_oas_series.iloc[-6]) * 100, 1
            )   # in bp (FRED reports HY OAS in %, so multiply delta by 100)
    except Exception:
        pass

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
