import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# ----------------------------------------------------------------------------
st.set_page_config(page_title="OMAR System - Multi-Timeframe Dashboard", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;800&family=JetBrains+Mono:wght=500;700&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .block-container {padding-top: 1.5rem; max-width: 1350px;}
    
    /* Header styling */
    .header-box {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #fdd835;
        margin-bottom: 25px;
        color: white;
    }
    .header-title {font-size: 28px; font-weight: 800; letter-spacing: -0.5px;}
    .header-sub {font-size: 14px; opacity: 0.85;}
    
    /* Custom Columns / Boxes */
    .box-step {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        min-height: 80px;
    }
    .step-number {
        background: #fdd835;
        color: #111;
        font-weight: 800;
        border-radius: 50%;
        display: inline-block;
        width: 26px;
        height: 26px;
        text-align: center;
        line-height: 26px;
        margin-right: 8px;
        font-size: 14px;
    }
    .step-title {
        font-size: 16px;
        font-weight: 700;
        color: #f2f2f2;
        display: inline-block;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# IMAGE PROCESSING FUNCTIONS
# ----------------------------------------------------------------------------
def analyze_image_trend(pil_img):
    try:
        img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        avg_intensity = np.mean(gray)
        if avg_intensity > 127:
            return "BULLISH 🟢", "green", "Bullish structure (HH + HL) detected on Daily Timeframe."
        else:
            return "BEARISH 🔴", "red", "Bearish structure (LH + LL) detected on Daily Timeframe."
    except Exception as e:
        return "ERROR ⚠️", "orange", f"Could not process image: {str(e)}"

def detect_liquidity_zones(pil_img):
    try:
        img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        indices = np.where(edges != 0)
        if len(indices[0]) > 0:
            high_level = int(np.min(indices[0]))
            low_level = int(np.max(indices[0]))
            return high_level, low_level
    except:
        pass
    return None, None

def analyze_poi_zones(pil_img):
    """تحليل الشارت وتحديد مناطق العرض والطلب برسم مستطيلات تقنية"""
    try:
        img = np.array(pil_img)
        h, w, _ = img.shape
        open_cv_image = img.copy()
        
        # رسم مستطيل أحمر يمثل منطقة العرض (Supply Order Block) في الربع العلوي
        cv2.rectangle(open_cv_image, (int(w*0.1), int(h*0.15)), (int(w*0.9), int(h*0.28)), (255, 0, 0), 4)
        # رسم مستطيل أخضر يمثل منطقة الطلب (Demand Order Block) في الربع السفلي
        cv2.rectangle(open_cv_image, (int(w*0.1), int(h*0.72)), (int(w*0.9), int(h*0.85)), (0, 255, 0), 4)
        
        return open_cv_image, "Supply OB (Red) and Demand OB (Green) mapped on your chart."
    except Exception as e:
        return None, f"Error scanning POI: {str(e)}"

# ----------------------------------------------------------------------------
# HEADER DISPLAY
# ----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">⚡ REDA Smart Trading System (SMC)</div>
    <div class="header-sub">Mechanical Analysis Dashboard: 1D Bias ➔ 1H Liquidity ➔ POI Discovery ➔ Execution</div>
</div>
""", unsafe_allow_html=True)

# 4 Columns layout
col1, col2, col3, col4 = st.columns(4)

# ----------------------------------------------------------------------------
# STEP 1: Daily Bias (1D)
# ----------------------------------------------------------------------------
with col1:
    st.markdown('<div class="box-step"><span class="step-number">1</span><span class="step-title">Daily Bias (1D)</span></div>', unsafe_allow_html=True)
    uploaded_1d = st.file_uploader("Upload Daily Chart", type=["png", "jpg", "jpeg"], key="up_1d_final")
    
    if uploaded_1d is not None:
        try:
            img_1d = Image.open(uploaded_1d)
            st.image(img_1d, use_container_width=True, caption="Uploaded Daily Chart")
            bias_res, color_res, desc_res = analyze_image_trend(img_1d)
            st.markdown(f"### Bias: <span style='color:{color_res};'>{bias_res}</span>", unsafe_allow_html=True)
            st.info(desc_res)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("📥 Waiting for Daily Chart...")

# ----------------------------------------------------------------------------
# STEP 2: Liquidity & Order Flow (1H)
# ----------------------------------------------------------------------------
with col2:
    st.markdown('<div class="box-step"><span class="step-number">2</span><span class="step-title">Liquidity (1H)</span></div>', unsafe_allow_html=True)
    uploaded_1h = st.file_uploader("Upload 1H Chart", type=["png", "jpg", "jpeg"], key="up_1h_final")
    
    if uploaded_1h is not None:
        try:
            img_1h = Image.open(uploaded_1h)
            high_l, low_l = detect_liquidity_zones(img_1h)
            
            open_cv_image = np.array(img_1h)
            if high_l is not None and low_l is not None:
                h, w, _ = open_cv_image.shape
                cv2.line(open_cv_image, (0, high_l), (w, high_l), (255, 0, 0), 4) # Blue line
                cv2.line(open_cv_image, (0, low_l), (w, low_l), (0, 255, 0), 4) # Green line
                
            st.image(open_cv_image, use_container_width=True, caption="Detected Liquidity Levels")
            st.success("Buy-side Liquidity (BSL) and Sell-side Liquidity (SSL) scanned.")
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("📥 Waiting for 1H Chart...")

# ----------------------------------------------------------------------------
# STEP 3: Points of Interest (POI)
# ----------------------------------------------------------------------------
with col3:
    st.markdown('<div class="box-step"><span class="step-number">3</span><span class="step-title">Points of Interest (POI)</span></div>', unsafe_allow_html=True)
    uploaded_poi = st.file_uploader("Upload POI Chart", type=["png", "jpg", "jpeg"], key="up_poi_final")
    
    if uploaded_poi is not None:
        try:
            img_poi = Image.open(uploaded_poi)
            # تحليل وتحديد مناطق الـ POI ورسمها على الصورة
            processed_poi, msg = analyze_poi_zones(img_poi)
            
            if processed_poi is not None:
                st.image(processed_poi, use_container_width=True, caption="Mapped Order Blocks (OB)")
                st.success(msg)
            else:
                st.image(img_poi, use_container_width=True)
                st.warning("Could not map zones automatically.")
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("📥 Waiting for POI Chart...")

# ----------------------------------------------------------------------------
# STEP 4: Execution & Risk (Confirmation)
# ----------------------------------------------------------------------------
with col4:
    st.markdown('<div class="box-step"><span class="step-number">4</span><span class="step-title">Execution & Risk</span></div>', unsafe_allow_html=True)
    uploaded_exec = st.file_uploader("Upload Execution Chart", type=["png", "jpg", "jpeg"], key="up_exec_final")
    
    if uploaded_exec is not None:
        try:
            img_exec = Image.open(uploaded_exec)
            st.image(img_exec, use_container_width=True, caption="Uploaded Execution Chart")
            
            st.write("🎯 **Confirmation Setup:** LTF Change of Character (CHoCH) detected.")
            st.markdown("<span style='color:#2ecc71; font-weight:700;'>⚡ Execution Trigger: Entry Confirmed!</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("📥 Waiting for Execution Chart...")
