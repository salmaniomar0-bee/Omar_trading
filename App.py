import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SMC Live Dashboard", layout="wide")
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "XAUUSD=X"]

# ----------------------------------------------------------------------------
# DATA FETCHING
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_daily(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
    df = _clean_ohlc(df)
    return df


@st.cache_data(ttl=300)
def fetch_4h(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="60d", interval="1h", progress=False, auto_adjust=True)
    df = _clean_ohlc(df)
    if df.empty:
        return df
    resampled = df.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    return resampled


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


# ----------------------------------------------------------------------------
# STEP 1: MARKET STRUCTURE & DAILY BIAS
# ----------------------------------------------------------------------------
def find_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    swing_highs, swing_lows = [], []
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index
    for i in range(left, len(df) - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]
        if highs[i] == window_h.max():
            swing_highs.append((idx[i], highs[i]))
        if lows[i] == window_l.min():
            swing_lows.append((idx[i], lows[i]))
    return swing_highs, swing_lows


def classify_structure(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Insufficient Data", "NEUTRAL"
    last_high, prev_high = swing_highs[-1][1], swing_highs[-2][1]
    last_low, prev_low = swing_lows[-1][1], swing_lows[-2][1]
    higher_high = last_high > prev_high
    higher_low = last_low > prev_low
    lower_high = last_high < prev_high
    lower_low = last_low < prev_low
    if higher_high and higher_low:
        return "Bullish Structure (HH + HL)", "BUY"
    elif lower_high and lower_low:
        return "Bearish Structure (LH + LL)", "SELL"
    elif higher_high and lower_low:
        return "Expansion / Broad Range", "NEUTRAL"
    elif lower_high and higher_low:
        return "Contraction / Consolidation", "NEUTRAL"
    else:
        return "Mixed Structure", "NEUTRAL"


def build_bias(daily_bias: str, h4_bias: str):
    if daily_bias == h4_bias and daily_bias != "NEUTRAL":
        return daily_bias, "High (Daily & 4H aligned)"
    elif daily_bias != "NEUTRAL":
        return daily_bias, "Medium (Daily bias, 4H not confirming)"
    else:
        return "NEUTRAL", "Low (No clear higher-timeframe trend)"


# ----------------------------------------------------------------------------
# STEP 2: POI DETECTION — ORDER BLOCKS (OB) & FAIR VALUE GAPS (FVG)
# ----------------------------------------------------------------------------
def find_fvgs(df: pd.DataFrame):
    """
    3-candle Fair Value Gap detection.
    Bullish FVG: Low[i] > High[i-2]   -> gap between candle i-2's high and candle i's low
    Bearish FVG: High[i] < Low[i-2]   -> gap between candle i-2's low and candle i's high
    """
    fvgs = []
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index
    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            fvgs.append({
                "type": "bullish", "top": lows[i], "bottom": highs[i - 2],
                "start": idx[i - 2], "end": idx[i],
            })
        if highs[i] < lows[i - 2]:
            fvgs.append({
                "type": "bearish", "top": lows[i - 2], "bottom": highs[i],
                "start": idx[i - 2], "end": idx[i],
            })
    return fvgs


def find_order_blocks(df: pd.DataFrame, swing_highs, swing_lows):
    """
    Simplified Order Block detection anchored to a Break of Structure (BOS):
    - Bullish OB: the last down-close candle before a candle that closes
      above the most recent prior swing high.
    - Bearish OB: the last up-close candle before a candle that closes
      below the most recent prior swing low.
    """
    obs = []
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index

    sh_times = np.array([t for t, _ in swing_highs])
    sh_prices = np.array([p for _, p in swing_highs])
    sl_times = np.array([t for t, _ in swing_lows])
    sl_prices = np.array([p for _, p in swing_lows])

    for i in range(1, len(df)):
        t = idx[i]

        prior_sh = sh_prices[sh_times < t]
        if len(prior_sh) and closes[i] > prior_sh[-1] >= closes[i - 1]:
            j = i - 1
            while j >= 0 and closes[j] >= opens[j]:
                j -= 1
            if j >= 0:
                obs.append({
                    "type": "bullish", "top": highs[j], "bottom": lows[j],
                    "time": idx[j], "bos_time": t,
                })

        prior_sl = sl_prices[sl_times < t]
        if len(prior_sl) and closes[i] < prior_sl[-1] <= closes[i - 1]:
            j = i - 1
            while j >= 0 and closes[j] <= opens[j]:
                j -= 1
            if j >= 0:
                obs.append({
                    "type": "bearish", "top": highs[j], "bottom": lows[j],
                    "time": idx[j], "bos_time": t,
                })
    return obs


# ----------------------------------------------------------------------------
# STEP 3: ENTRY, STOP LOSS & TAKE PROFIT
# ----------------------------------------------------------------------------
def generate_trade_setup(h4_df, h4_highs, h4_lows, h4_obs, h4_fvgs, overall_bias, symbol):
    """
    Builds an entry setup on the 4H timeframe in the direction of the overall bias,
    using the most recent matching Order Block (preferred) or Fair Value Gap as the
    Point of Interest (POI) / entry zone.
    """
    if overall_bias not in ("BUY", "SELL") or h4_df.empty:
        return None

    current_price = float(h4_df["Close"].iloc[-1])
    direction = "bullish" if overall_bias == "BUY" else "bearish"

    candidates = [ob for ob in h4_obs if ob["type"] == direction]
    zone_source = "Order Block"
    if not candidates:
        candidates = [f for f in h4_fvgs if f["type"] == direction]
        zone_source = "Fair Value Gap"
    if not candidates:
        return None

    zone = candidates[-1]
    entry_top, entry_bottom = zone["top"], zone["bottom"]
    entry = (entry_top + entry_bottom) / 2
    zone_height = entry_top - entry_bottom
    buffer = zone_height * 0.15 if zone_height > 0 else current_price * 0.001

    if direction == "bullish":
        sl = entry_bottom - buffer
        risk = entry - sl
        targets = [p for _, p in h4_highs if p > entry]
        tp = min(targets) if targets else entry + risk * 2
        rr = (tp - entry) / risk if risk > 0 else None
        status = "Price inside zone (setup active)" if entry_bottom <= current_price <= entry_top \
            else "Waiting for pullback into zone"
    else:
        sl = entry_top + buffer
        risk = sl - entry
        targets = [p for _, p in h4_lows if p < entry]
        tp = max(targets) if targets else entry - risk * 2
        rr = (entry - tp) / risk if risk > 0 else None
        status = "Price inside zone (setup active)" if entry_bottom <= current_price <= entry_top \
            else "Waiting for pullback into zone"

    return {
        "Symbol": symbol,
        "Direction": "BUY" if direction == "bullish" else "SELL",
        "POI Type": zone_source,
        "Zone Low": round(entry_bottom, 5),
        "Zone High": round(entry_top, 5),
        "Entry": round(entry, 5),
        "Stop Loss": round(sl, 5),
        "Take Profit": round(tp, 5),
        "R:R": round(rr, 2) if rr else "N/A",
        "Current Price": round(current_price, 5),
        "Status": status,
    }


# ----------------------------------------------------------------------------
# CHARTING
# ----------------------------------------------------------------------------
def make_chart(df: pd.DataFrame, swing_highs, swing_lows, title: str,
                obs=None, fvgs=None, trade_setup=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=title
    ))

    if swing_highs:
        sh_x, sh_y = zip(*swing_highs)
        fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode="markers", name="Swing High",
                                  marker=dict(color="red", size=8, symbol="triangle-down")))
    if swing_lows:
        sl_x, sl_y = zip(*swing_lows)
        fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode="markers", name="Swing Low",
                                  marker=dict(color="green", size=8, symbol="triangle-up")))

    x_end = df.index[-1]

    # Order Blocks
    if obs:
        for ob in obs:
            if ob["time"] < df.index[0]:
                continue
            color = "rgba(46,204,113,0.20)" if ob["type"] == "bullish" else "rgba(231,76,60,0.20)"
            fig.add_shape(type="rect", x0=ob["time"], x1=x_end, y0=ob["bottom"], y1=ob["top"],
                          fillcolor=color, line=dict(width=0), layer="below")

    # Fair Value Gaps
    if fvgs:
        for fvg in fvgs:
            if fvg["start"] < df.index[0]:
                continue
            color = "rgba(52,152,219,0.18)" if fvg["type"] == "bullish" else "rgba(230,126,34,0.18)"
            fig.add_shape(type="rect", x0=fvg["start"], x1=x_end, y0=fvg["bottom"], y1=fvg["top"],
                          fillcolor=color, line=dict(width=0), layer="below")

    # Entry / SL / TP lines
    if trade_setup:
        fig.add_hline(y=trade_setup["Entry"], line=dict(color="yellow", width=1.5, dash="dash"),
                      annotation_text="Entry", annotation_position="right")
        fig.add_hline(y=trade_setup["Stop Loss"], line=dict(color="red", width=1.5, dash="dot"),
                      annotation_text="SL", annotation_position="right")
        fig.add_hline(y=trade_setup["Take Profit"], line=dict(color="lime", width=1.5, dash="dot"),
                      annotation_text="TP", annotation_position="right")

    fig.update_layout(title=title, height=460, xaxis_rangeslider_visible=False,
                       margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📊 Smart Money Concepts — Live Structure, POI & Entry Dashboard")

with st.sidebar:
    st.header("Settings")
    symbols = st.multiselect("Symbols to analyze", options=DEFAULT_SYMBOLS + ["Custom..."],
                              default=["BTC-USD", "EURUSD=X"])
    custom_symbol = ""
    if "Custom..." in symbols:
        custom_symbol = st.text_input("Enter custom ticker")
        symbols = [s for s in symbols if s != "Custom..."]
        if custom_symbol:
            symbols.append(custom_symbol.strip())
    left_len = st.slider("Swing sensitivity", 2, 6, 3)
    show_ob = st.checkbox("Show Order Blocks", value=True)
    show_fvg = st.checkbox("Show Fair Value Gaps", value=True)

if not symbols:
    st.info("Select at least one symbol from the sidebar to begin.")
    st.stop()

summary_rows = []
trade_setups = []

for symbol in symbols:
    st.markdown(f"## {symbol}")
    daily_df = fetch_daily(symbol)
    h4_df = fetch_4h(symbol)
    if daily_df.empty or h4_df.empty:
        st.warning(f"No data returned for {symbol}.")
        continue

    d_highs, d_lows = find_swing_points(daily_df, left=left_len, right=left_len)
    h_highs, h_lows = find_swing_points(h4_df, left=left_len, right=left_len)

    daily_structure, daily_bias = classify_structure(d_highs, d_lows)
    h4_structure, h4_bias = classify_structure(h_highs, h_lows)
    overall_bias, confidence = build_bias(daily_bias, h4_bias)

    h4_obs = find_order_blocks(h4_df, h_highs, h_lows)
    h4_fvgs = find_fvgs(h4_df)

    trade_setup = generate_trade_setup(h4_df, h_highs, h_lows, h4_obs, h4_fvgs, overall_bias, symbol)
    if trade_setup:
        trade_setups.append(trade_setup)

    summary_rows.append({
        "Symbol": symbol, "Daily Bias": daily_bias, "4H Bias": h4_bias,
        "Overall Bias": overall_bias, "Confidence": confidence,
        "Active Setup": "Yes" if trade_setup else "No"
    })

    col1, col2, col3 = st.columns(3)
    col1.metric("Daily Bias", daily_bias)
    col2.metric("4H Bias", h4_bias)
    col3.metric("Overall Bias", overall_bias)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        window = daily_df.tail(120)
        st.plotly_chart(
            make_chart(window, [p for p in d_highs if p[0] in window.index],
                       [p for p in d_lows if p[0] in window.index], f"{symbol} - Daily"),
            use_container_width=True
        )
    with chart_col2:
        window = h4_df.tail(120)
        st.plotly_chart(
            make_chart(
                window,
                [p for p in h_highs if p[0] in window.index],
                [p for p in h_lows if p[0] in window.index],
                f"{symbol} - 4H (POI + Entry)",
                obs=h4_obs if show_ob else None,
                fvgs=h4_fvgs if show_fvg else None,
                trade_setup=trade_setup,
            ),
            use_container_width=True
        )

    if trade_setup:
        st.markdown("#### 🎯 Trade Setup")
        st.dataframe(pd.DataFrame([trade_setup]), use_container_width=True, hide_index=True)
    else:
        st.caption("No valid POI-based entry setup found for the current bias.")

if summary_rows:
    st.markdown("## 🧭 Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

if trade_setups:
    st.markdown("## 📋 All Active Trade Setups")
    st.dataframe(pd.DataFrame(trade_setups), use_container_width=True, hide_index=True)
