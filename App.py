import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SMC Dashboard", layout="wide", page_icon="📊")
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "XAUUSD=X"]

TF_CONFIG = {
    "Daily": dict(interval="1d", period="2y", resample=None),
    "4H":    dict(interval="1h", period="180d", resample="4h"),
    "1H":    dict(interval="1h", period="180d", resample=None),
    "30M":   dict(interval="30m", period="60d", resample=None),
}
TF_ORDER = ["Daily", "4H", "1H", "30M"]

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"]  {font-family: 'Inter', sans-serif;}

    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1300px;}

    #MainMenu, footer {visibility: hidden;}

    /* ---- Header ---- */
    .app-header {
        display:flex; align-items:center; justify-content:space-between;
        padding: 4px 0 18px 0; border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 22px;
    }
    .app-title {font-size: 26px; font-weight: 800; letter-spacing: -0.5px; color: #f2f2f2;}
    .app-sub {font-size: 13px; color: #8a8f98; margin-top: 2px; font-weight: 500;}
    .live-dot {
        display:inline-block; width:8px; height:8px; border-radius:50%;
        background:#2ecc71; margin-right:6px; box-shadow: 0 0 8px #2ecc71;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.08);}
    .stTabs [data-baseweb="tab"] {
        height: 44px; background-color: transparent; border-radius: 8px 8px 0 0;
        padding: 0 20px; font-weight: 600; font-size: 14.5px; color: #9aa0a8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.06); color: #ffffff !important;
        border-bottom: 2px solid #4fc3f7;
    }

    /* ---- Cards ---- */
    .panel {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 18px 20px; margin-bottom: 16px;
    }
    .kpi-row {display:flex; gap:12px; flex-wrap:wrap; margin-bottom: 16px;}
    .kpi {
        flex:1; min-width:140px; background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px 16px;
    }
    .kpi-label {font-size: 11.5px; color: #8a8f98; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;}
    .kpi-value {font-size: 19px; font-weight: 700; color:#f2f2f2; font-family:'JetBrains Mono',monospace; margin-top:2px;}

    .bias-card {border-radius: 14px; padding: 18px 20px; margin-bottom: 14px; font-size: 14.5px; line-height: 1.65;}
    .bias-bull {background: linear-gradient(135deg, rgba(46,204,113,0.10), rgba(46,204,113,0.03)); border: 1px solid rgba(46,204,113,0.35);}
    .bias-bear {background: linear-gradient(135deg, rgba(231,76,60,0.10), rgba(231,76,60,0.03)); border: 1px solid rgba(231,76,60,0.35);}
    .bias-neutral {background: linear-gradient(135deg, rgba(149,165,166,0.10), rgba(149,165,166,0.03)); border: 1px solid rgba(149,165,166,0.35);}

    .symbol-row {display:flex; align-items:center; gap:10px; margin-bottom: 14px;}
    .symbol-name {font-size: 22px; font-weight: 800; color:#f2f2f2;}
    .price-tag {font-size: 14px; color:#9aa0a8; font-family:'JetBrains Mono',monospace;}

    .badge {display:inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 12.5px; letter-spacing: 0.5px;}
    .badge-buy {background:#1e8449; color:white;}
    .badge-sell {background:#c0392b; color:white;}
    .badge-neutral {background:#5d6d7e; color:white;}

    .section-label {font-size: 12px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#4fc3f7; margin-bottom:8px;}

    div[data-testid="stMetric"] {background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding: 10px 14px;}

    .zone-tag {
        display:inline-block; padding:2px 10px; border-radius:6px; font-size:12px; font-weight:600;
        font-family:'JetBrains Mono',monospace; margin: 2px 4px 2px 0;
    }
    .zone-bull {background: rgba(46,204,113,0.15); color:#2ecc71; border:1px solid rgba(46,204,113,0.3);}
    .zone-bear {background: rgba(231,76,60,0.15); color:#e74c3c; border:1px solid rgba(231,76,60,0.3);}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA FETCHING
# ----------------------------------------------------------------------------
def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


@st.cache_data(ttl=300)
def fetch_ohlc(symbol: str, tf_label: str) -> pd.DataFrame:
    cfg = TF_CONFIG[tf_label]
    df = yf.download(symbol, period=cfg["period"], interval=cfg["interval"], progress=False, auto_adjust=True)
    df = _clean_ohlc(df)
    if df.empty:
        return df
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()
    return df


# ----------------------------------------------------------------------------
# MARKET STRUCTURE & BIAS
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


def build_bias(htf_bias: str, ltf_bias: str):
    if htf_bias == ltf_bias and htf_bias != "NEUTRAL":
        return htf_bias, "High — both timeframes aligned"
    elif htf_bias != "NEUTRAL":
        return htf_bias, "Medium — higher timeframe only"
    else:
        return "NEUTRAL", "Low — no clear higher-timeframe trend"


def explain_bias(symbol, htf_label, ltf_label, htf_bias, ltf_bias, overall_bias, confidence,
                  htf_highs, htf_lows, ltf_highs, ltf_lows):
    def swing_line(highs, lows, label):
        if len(highs) < 2 or len(lows) < 2:
            return f"Not enough swing points yet on **{label}** to confirm structure."
        lh_date, lh_price = highs[-1]
        ph_date, ph_price = highs[-2]
        ll_date, ll_price = lows[-1]
        pl_date, pl_price = lows[-2]
        high_move = "higher high" if lh_price > ph_price else "lower high"
        low_move = "higher low" if ll_price > pl_price else "lower low"
        return (f"**{label}:** last swing high {lh_price:.4f} ({lh_date.strftime('%b %d, %H:%M')}) vs "
                f"previous {ph_price:.4f} → **{high_move}**. "
                f"Last swing low {ll_price:.4f} ({ll_date.strftime('%b %d, %H:%M')}) vs "
                f"previous {pl_price:.4f} → **{low_move}**.")

    htf_line = swing_line(htf_highs, htf_lows, htf_label)
    ltf_line = swing_line(ltf_highs, ltf_lows, ltf_label)

    if overall_bias == "BUY":
        verdict = (f"**{symbol} is BULLISH.** Structure is printing higher highs and higher lows — "
                   f"buyers are in control and the path of least resistance is up.")
    elif overall_bias == "SELL":
        verdict = (f"**{symbol} is BEARISH.** Structure is printing lower highs and lower lows — "
                   f"sellers are in control and the path of least resistance is down.")
    else:
        verdict = (f"**{symbol} has NO CLEAR BIAS right now.** The two timeframes disagree or the "
                   f"market is ranging — better to wait for a clean break of structure.")

    return f"{verdict}\n\n{htf_line}\n\n{ltf_line}\n\n**Confidence:** {confidence}"


# ----------------------------------------------------------------------------
# POI DETECTION — ORDER BLOCKS & FAIR VALUE GAPS
# ----------------------------------------------------------------------------
def find_fvgs(df: pd.DataFrame):
    fvgs = []
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index
    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            fvgs.append({"type": "bullish", "top": lows[i], "bottom": highs[i - 2], "start": idx[i - 2], "end": idx[i]})
        if highs[i] < lows[i - 2]:
            fvgs.append({"type": "bearish", "top": lows[i - 2], "bottom": highs[i], "start": idx[i - 2], "end": idx[i]})
    return fvgs


def find_order_blocks(df: pd.DataFrame, swing_highs, swing_lows):
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
                obs.append({"type": "bullish", "top": highs[j], "bottom": lows[j], "time": idx[j], "bos_time": t})

        prior_sl = sl_prices[sl_times < t]
        if len(prior_sl) and closes[i] < prior_sl[-1] <= closes[i - 1]:
            j = i - 1
            while j >= 0 and closes[j] <= opens[j]:
                j -= 1
            if j >= 0:
                obs.append({"type": "bearish", "top": highs[j], "bottom": lows[j], "time": idx[j], "bos_time": t})
    return obs


# ----------------------------------------------------------------------------
# ENTRY, STOP LOSS & TAKE PROFIT
# ----------------------------------------------------------------------------
def generate_trade_setup(df, highs, lows, obs, fvgs, overall_bias, symbol):
    if overall_bias not in ("BUY", "SELL") or df.empty:
        return None

    current_price = float(df["Close"].iloc[-1])
    direction = "bullish" if overall_bias == "BUY" else "bearish"

    candidates = [ob for ob in obs if ob["type"] == direction]
    zone_source = "Order Block"
    if not candidates:
        candidates = [f for f in fvgs if f["type"] == direction]
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
        targets = [p for _, p in highs if p > entry]
        tp = min(targets) if targets else entry + risk * 2
        rr = (tp - entry) / risk if risk > 0 else None
        status = "Price inside zone (setup active)" if entry_bottom <= current_price <= entry_top else "Waiting for pullback into zone"
    else:
        sl = entry_top + buffer
        risk = sl - entry
        targets = [p for _, p in lows if p < entry]
        tp = max(targets) if targets else entry - risk * 2
        rr = (entry - tp) / risk if risk > 0 else None
        status = "Price inside zone (setup active)" if entry_bottom <= current_price <= entry_top else "Waiting for pullback into zone"

    return {
        "Symbol": symbol, "Direction": "BUY" if direction == "bullish" else "SELL", "POI Type": zone_source,
        "Zone Low": round(entry_bottom, 5), "Zone High": round(entry_top, 5), "Entry": round(entry, 5),
        "Stop Loss": round(sl, 5), "Take Profit": round(tp, 5), "R:R": round(rr, 2) if rr else "N/A",
        "Current Price": round(current_price, 5), "Status": status,
    }


# ----------------------------------------------------------------------------
# CHARTING
# ----------------------------------------------------------------------------
def make_chart(df, swing_highs=None, swing_lows=None, title="", obs=None, fvgs=None, trade_setup=None, height=520):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=title,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))

    if swing_highs:
        sh_x, sh_y = zip(*swing_highs)
        fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode="markers", name="Swing High",
                                  marker=dict(color="#ef5350", size=8, symbol="triangle-down")))
    if swing_lows:
        sl_x, sl_y = zip(*swing_lows)
        fig.add_trace(go.Scatter(x=sl_x, y=sl_y, mode="markers", name="Swing Low",
                                  marker=dict(color="#26a69a", size=8, symbol="triangle-up")))

    x_end = df.index[-1]

    if obs:
        for ob in obs:
            if ob["time"] < df.index[0]:
                continue
            color = "rgba(46,204,113,0.22)" if ob["type"] == "bullish" else "rgba(231,76,60,0.22)"
            fig.add_shape(type="rect", x0=ob["time"], x1=x_end, y0=ob["bottom"], y1=ob["top"],
                          fillcolor=color, line=dict(width=0), layer="below")

    if fvgs:
        for fvg in fvgs:
            if fvg["start"] < df.index[0]:
                continue
            color = "rgba(79,195,247,0.20)" if fvg["type"] == "bullish" else "rgba(255,167,38,0.20)"
            fig.add_shape(type="rect", x0=fvg["start"], x1=x_end, y0=fvg["bottom"], y1=fvg["top"],
                          fillcolor=color, line=dict(width=0), layer="below")

    if trade_setup:
        fig.add_hline(y=trade_setup["Entry"], line=dict(color="#fdd835", width=1.5, dash="dash"),
                      annotation_text="Entry", annotation_position="right")
        fig.add_hline(y=trade_setup["Stop Loss"], line=dict(color="#ef5350", width=1.5, dash="dot"),
                      annotation_text="SL", annotation_position="right")
        fig.add_hline(y=trade_setup["Take Profit"], line=dict(color="#26a69a", width=1.5, dash="dot"),
                      annotation_text="TP", annotation_position="right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#c8ccd0")),
        height=height, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def badge_html(bias):
    cls = {"BUY": "badge-buy", "SELL": "badge-sell", "NEUTRAL": "badge-neutral"}[bias]
    return f'<span class="badge {cls}">{bias}</span>'


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">📊 SMC Dashboard</div>
        <div class="app-sub"><span class="live-dot"></span>Smart Money Concepts — Structure · POI · Entries</div>
    </div>
    <div class="app-sub">{datetime.now().strftime('%b %d, %Y · %H:%M')}</div>
</div>
""", unsafe_allow_html=True)

tab_bias, tab_poi, tab_entry = st.tabs(["📈  Bias", "🎯  POI (Order Blocks & FVG)", "🚀  Entry Setup"])

# ============================================================================
# TAB 1 — BIAS
# ============================================================================
with tab_bias:
    c1, c2, c3, c4 = st.columns([2, 1.4, 1.4, 1.2])
    with c1:
        symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS, key="bias_symbol")
    with c2:
        htf_label = st.selectbox("Higher timeframe", TF_ORDER, index=0, key="bias_htf")
    with c3:
        ltf_options = [tf for tf in TF_ORDER if tf != htf_label]
        ltf_label = st.selectbox("Confirmation timeframe", ltf_options,
                                  index=min(1, len(ltf_options) - 1), key="bias_ltf")
    with c4:
        sensitivity = st.slider("Swing sensitivity", 2, 6, 3, key="bias_sens")

    htf_df = fetch_ohlc(symbol, htf_label)
    ltf_df = fetch_ohlc(symbol, ltf_label)

    if htf_df.empty or ltf_df.empty:
        st.warning(f"No data returned for {symbol}.")
    else:
        htf_highs, htf_lows = find_swing_points(htf_df, sensitivity, sensitivity)
        ltf_highs, ltf_lows = find_swing_points(ltf_df, sensitivity, sensitivity)
        htf_structure, htf_bias = classify_structure(htf_highs, htf_lows)
        ltf_structure, ltf_bias = classify_structure(ltf_highs, ltf_lows)
        overall_bias, confidence = build_bias(htf_bias, ltf_bias)
        current_price = float(ltf_df["Close"].iloc[-1])

        st.markdown(
            f'<div class="symbol-row"><span class="symbol-name">{symbol}</span> {badge_html(overall_bias)} '
            f'<span class="price-tag">{current_price:.4f}</span></div>', unsafe_allow_html=True
        )

        card_class = {"BUY": "bias-bull", "SELL": "bias-bear", "NEUTRAL": "bias-neutral"}[overall_bias]
        explanation = explain_bias(symbol, htf_label, ltf_label, htf_bias, ltf_bias, overall_bias, confidence,
                                    htf_highs, htf_lows, ltf_highs, ltf_lows)
        st.markdown(f'<div class="bias-card {card_class}">{explanation}</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f'<div class="section-label">{htf_label} Structure</div>', unsafe_allow_html=True)
            window = htf_df.tail(150)
            st.plotly_chart(make_chart(
                window, [p for p in htf_highs if p[0] in window.index],
                [p for p in htf_lows if p[0] in window.index], f"{symbol} · {htf_label}", height=440
            ), use_container_width=True)
        with cc2:
            st.markdown(f'<div class="section-label">{ltf_label} Structure</div>', unsafe_allow_html=True)
            window = ltf_df.tail(150)
            st.plotly_chart(make_chart(
                window, [p for p in ltf_highs if p[0] in window.index],
                [p for p in ltf_lows if p[0] in window.index], f"{symbol} · {ltf_label}", height=440
            ), use_container_width=True)

# ============================================================================
# TAB 2 — POI (Order Blocks & FVG)
# ============================================================================
with tab_poi:
    c1, c2, c3, c4 = st.columns([2, 1.4, 1, 1])
    with c1:
        poi_symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS, key="poi_symbol")
    with c2:
        poi_tf = st.selectbox("Timeframe", TF_ORDER, index=1, key="poi_tf")
    with c3:
        show_ob = st.checkbox("Order Blocks", value=True, key="poi_ob")
    with c4:
        show_fvg = st.checkbox("Fair Value Gaps", value=True, key="poi_fvg")

    poi_sens = st.slider("Swing sensitivity", 2, 6, 3, key="poi_sens")

    poi_df = fetch_ohlc(poi_symbol, poi_tf)
    if poi_df.empty:
        st.warning(f"No data returned for {poi_symbol}.")
    else:
        poi_highs, poi_lows = find_swing_points(poi_df, poi_sens, poi_sens)
        poi_obs = find_order_blocks(poi_df, poi_highs, poi_lows)
        poi_fvgs = find_fvgs(poi_df)
        current_price = float(poi_df["Close"].iloc[-1])

        st.markdown(
            f'<div class="symbol-row"><span class="symbol-name">{poi_symbol}</span> '
            f'<span class="price-tag">· {poi_tf} · {current_price:.4f}</span></div>', unsafe_allow_html=True
        )

        n_bull_ob = sum(1 for o in poi_obs if o["type"] == "bullish")
        n_bear_ob = sum(1 for o in poi_obs if o["type"] == "bearish")
        n_bull_fvg = sum(1 for f in poi_fvgs if f["type"] == "bullish")
        n_bear_fvg = sum(1 for f in poi_fvgs if f["type"] == "bearish")

        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi"><div class="kpi-label">Bullish OB</div><div class="kpi-value">{n_bull_ob}</div></div>
            <div class="kpi"><div class="kpi-label">Bearish OB</div><div class="kpi-value">{n_bear_ob}</div></div>
            <div class="kpi"><div class="kpi-label">Bullish FVG</div><div class="kpi-value">{n_bull_fvg}</div></div>
            <div class="kpi"><div class="kpi-label">Bearish FVG</div><div class="kpi-value">{n_bear_fvg}</div></div>
        </div>
        """, unsafe_allow_html=True)

        window = poi_df.tail(150)
        st.plotly_chart(make_chart(
            window, [p for p in poi_highs if p[0] in window.index], [p for p in poi_lows if p[0] in window.index],
            f"{poi_symbol} · {poi_tf} — POI Zones",
            obs=poi_obs if show_ob else None, fvgs=poi_fvgs if show_fvg else None, height=520
        ), use_container_width=True)

        recent_obs = [o for o in poi_obs if o["time"] >= window.index[0]][-8:]
        if recent_obs:
            st.markdown('<div class="section-label">Recent Order Blocks</div>', unsafe_allow_html=True)
            tags = "".join(
                f'<span class="zone-tag zone-{"bull" if o["type"]=="bullish" else "bear"}">'
                f'{o["type"].upper()} {o["bottom"]:.4f}–{o["top"]:.4f} ({o["time"].strftime("%b %d %H:%M")})</span>'
                for o in recent_obs
            )
            st.markdown(tags, unsafe_allow_html=True)

# ============================================================================
# TAB 3 — ENTRY SETUP
# ============================================================================
with tab_entry:
    c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1.2])
    with c1:
        e_symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS, key="entry_symbol")
    with c2:
        e_htf = st.selectbox("Bias timeframe", TF_ORDER, index=0, key="entry_htf")
    with c3:
        e_ltf_options = [tf for tf in TF_ORDER if tf != e_htf]
        e_ltf = st.selectbox("Confirmation timeframe", e_ltf_options,
                              index=min(1, len(e_ltf_options) - 1), key="entry_ltf")
    with c4:
        e_entry_tf = st.selectbox("Entry timeframe", TF_ORDER, index=1, key="entry_tf")

    e_sens = st.slider("Swing sensitivity", 2, 6, 3, key="entry_sens")

    e_htf_df = fetch_ohlc(e_symbol, e_htf)
    e_ltf_df = fetch_ohlc(e_symbol, e_ltf)
    e_entry_df = fetch_ohlc(e_symbol, e_entry_tf)

    if e_htf_df.empty or e_ltf_df.empty or e_entry_df.empty:
        st.warning(f"No data returned for {e_symbol}.")
    else:
        e_htf_highs, e_htf_lows = find_swing_points(e_htf_df, e_sens, e_sens)
        e_ltf_highs, e_ltf_lows = find_swing_points(e_ltf_df, e_sens, e_sens)
        _, e_htf_bias = classify_structure(e_htf_highs, e_htf_lows)
        _, e_ltf_bias = classify_structure(e_ltf_highs, e_ltf_lows)
        e_overall_bias, e_confidence = build_bias(e_htf_bias, e_ltf_bias)

        e_entry_highs, e_entry_lows = find_swing_points(e_entry_df, e_sens, e_sens)
        e_obs = find_order_blocks(e_entry_df, e_entry_highs, e_entry_lows)
        e_fvgs = find_fvgs(e_entry_df)
        trade_setup = generate_trade_setup(e_entry_df, e_entry_highs, e_entry_lows, e_obs, e_fvgs,
                                            e_overall_bias, e_symbol)

        st.markdown(
            f'<div class="symbol-row"><span class="symbol-name">{e_symbol}</span> {badge_html(e_overall_bias)} '
            f'<span class="price-tag">bias from {e_htf} + {e_ltf} · {e_confidence}</span></div>',
            unsafe_allow_html=True
        )

        if trade_setup:
            k = trade_setup
            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi"><div class="kpi-label">POI Type</div><div class="kpi-value">{k['POI Type']}</div></div>
                <div class="kpi"><div class="kpi-label">Entry</div><div class="kpi-value">{k['Entry']}</div></div>
                <div class="kpi"><div class="kpi-label">Stop Loss</div><div class="kpi-value">{k['Stop Loss']}</div></div>
                <div class="kpi"><div class="kpi-label">Take Profit</div><div class="kpi-value">{k['Take Profit']}</div></div>
                <div class="kpi"><div class="kpi-label">R:R</div><div class="kpi-value">{k['R:R']}</div></div>
            </div>
            <div class="panel"><b>Status:</b> {k['Status']} &nbsp;·&nbsp; <b>Zone:</b> {k['Zone Low']} – {k['Zone High']}</div>
            """, unsafe_allow_html=True)

            window = e_entry_df.tail(150)
            st.plotly_chart(make_chart(
                window, [p for p in e_entry_highs if p[0] in window.index],
                [p for p in e_entry_lows if p[0] in window.index],
                f"{e_symbol} · {e_entry_tf} — Entry Setup",
                obs=e_obs, fvgs=e_fvgs, trade_setup=trade_setup, height=540
            ), use_container_width=True)
        else:
            st.info("No valid POI-based entry setup found for the current bias on this timeframe.")
            window = e_entry_df.tail(150)
            st.plotly_chart(make_chart(
                window, [p for p in e_entry_highs if p[0] in window.index],
                [p for p in e_entry_lows if p[0] in window.index],
                f"{e_symbol} · {e_entry_tf}", obs=e_obs, fvgs=e_fvgs, height=460
            ), use_container_width=True)
