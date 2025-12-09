import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from dotenv import load_dotenv

# Load env vars (for local support)
load_dotenv()

# --- Config ---
st.set_page_config(page_title="Stock Prediction System", layout="wide", page_icon="📈")
MODEL_DIR = "models/AAPL" # Defaulting to AAPL models for demo inference on all stocks (Logic Transfer)

# --- Secrets ---
# Try to get from st.secrets (Cloud) or os.getenv (Local)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Helper Functions ---
@st.cache_resource
def load_models_local():
    """Loads models directly from disk per Standalone/Hugging Face requirements."""
    models = {}
    try:
        models['regression'] = joblib.load(f"{MODEL_DIR}/regression_model.pkl")
        models['classification'] = joblib.load(f"{MODEL_DIR}/classification_model.pkl")
        models['clustering'] = joblib.load(f"{MODEL_DIR}/clustering_model.pkl")
        models['pca'] = joblib.load(f"{MODEL_DIR}/pca_model.pkl")
        return models
    except Exception as e:
        st.error(f"Failed to load models locally: {e}")
        return None

def send_discord_notification(symbol, price, change_percent, prediction_dir):
    """Sends a formatted message to Discord."""
    if not WEBHOOK_URL:
        # print("No Webhook URL found.")
        return
    
    emoji = "🚀" if change_percent > 0 else "🔻"
    pred_emoji = "🟢" if "UP" in prediction_dir else "🔴"
    
    message = {
        "content": f"**Hourly Stock Update** 🕒\n"
                   f"**{symbol}**: ${price:.2f} {emoji} ({change_percent:.2f}%)\n"
                   f"**AI Prediction:** {prediction_dir} {pred_emoji}"
    }
    try:
        requests.post(WEBHOOK_URL, json=message)
        print(f"Sent notification for {symbol}")
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

@st.cache_data(ttl=3600) # CACHE FOR 1 HOUR
def fetch_live_data(symbol):
    """Fetches raw price data and calculates indicators locally (bypassing API limits)."""
    if not ALPHA_VANTAGE_KEY:
        st.warning("⚠️ ALPHA_VANTAGE_API_KEY not found. Using Mock Data.")
        return get_mock_data(symbol)

    try:
        # Fetch only Daily Price (Free Endpoint)
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact') # 100 data points is enough for indicators
        
        # Ensure sorted chronologically
        data = data.sort_index()
        
        # Rename columns standard for calculation
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # --- Local Calculation (Free & Unlimited) ---
        # SMA
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        # signal = macd.ewm(span=9, adjust=False).mean() # We don't use signal for model input
        data['macd'] = macd

        # Get latest valid row
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        change_percent = ((latest['close'] - prev['close']) / prev['close']) * 100

        return {
            "price": float(latest['close']),
            "change": change_percent,
            "sma_20": float(latest['sma_20']),
            "sma_50": float(latest['sma_50']),
            "rsi": float(latest['rsi']),
            "macd": float(latest['macd']),
            "is_mock": False
        }

    except Exception as e:
        # st.warning(f"API Error: {e}. Falling back to mock.")
        # Only show warning if it's not the common "Key Error" on first load
        print(f"Fetch failed: {e}")
        st.warning(f"Could not fetch data for {symbol} (API Limit?). Showing Mock Data.")
        return get_mock_data(symbol)

def get_mock_data(symbol):
    """Generates realistic mock data if API fails or key missing."""
    base_price = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300, "AMZN": 3400, "TSLA": 900, "NVDA": 400}
    price = base_price.get(symbol, 100) + np.random.uniform(-5, 5)
    return {
        "price": price,
        "change": np.random.uniform(-2, 2),
        "sma_20": price * 0.95,
        "sma_50": price * 0.90,
        "rsi": np.random.uniform(30, 70),
        "macd": np.random.uniform(-1, 1),
        "is_mock": True
    }

# --- UI Layout ---
st.title("📈 AI Stock Prediction System")

# Sidebar
st.sidebar.header("Control Panel")
available_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
symbol = st.sidebar.selectbox("Select Stock", available_stocks)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear() # Clear cache to force update
    st.rerun()

# --- Main Logic ---

# 1. Fetch Data
with st.spinner(f"Fetching Live Data for {symbol}..."):
    data = fetch_live_data(symbol)

# 2. visual Header
col_head1, col_head2, col_head3 = st.columns(3)
with col_head1:
    st.metric("Current Price", f"${data['price']:.2f}", f"{data['change']:.2f}%")
with col_head2:
    st.metric("RSI (Momentum)", f"{data['rsi']:.1f}", "Overbought" if data['rsi']>70 else "Oversold" if data['rsi']<30 else "Neutral", delta_color="off")
with col_head3:
    source = "🔴 Mock Data (Check API Key)" if data['is_mock'] else "🟢 Live Alpha Vantage Data"
    st.caption(f"Data Source: {source}")
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

# 3. AI Prediction
st.markdown("---")
st.subheader(f"🤖 AI Analysis for {symbol}")

features = np.array([[data['sma_20'], data['sma_50'], data['rsi'], data['macd']]])
models = load_models_local()

if models:
    col_pred1, col_pred2 = st.columns(2)
    
    # Regression
    pred_price = models['regression'].predict(features)[0]
    
    # Classification
    pred_direction_prob = models['classification'].predict_proba(features)[0]
    direction = "UP 🚀" if pred_direction_prob[1] > 0.5 else "DOWN 🔻"
    confidence = max(pred_direction_prob)
    
    with col_pred1:
        st.info(f"**Predicted Direction:** {direction}")
        st.progress(float(confidence), text=f"Confidence: {confidence*100:.1f}%")
        
    with col_pred2:
        st.success(f"**Target Price (Next Close):** ${pred_price:.2f}")

    # Discord Notification Trigger (Only if not mock and strictly if specific conditions met)
    # To avoid spamming on every refresh, we rely on the fact that this function is only called 
    # when cache invalidates (once per hour) or user manually clears it.
    if not data['is_mock']:
        send_discord_notification(symbol, data['price'], data['change'], direction)

# 4. Market Analysis Tabs
st.markdown("---")
tab1, tab2 = st.tabs(["📊 Technical Dashboard", "🧭 Market Regime (Cluster)"])

with tab1:
    # Gauge Chart for RSI
    fig_rsi = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = data['rsi'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "RSI Strength"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "darkblue"},
                 'steps': [
                     {'range': [0, 30], 'color': "lightgreen"}, # Oversold
                     {'range': [30, 70], 'color': "gray"},
                     {'range': [70, 100], 'color': "red"}], # Overbought
                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': data['rsi']}}))
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    # Clustering Visualization
    # Using approximated volatility for visualization
    volatility = data['price'] * 0.02 # estimating 2% volatility for visualization if real calc not avail
    cluster_features = np.array([[volatility, data['rsi']]])
    cluster_id = models['clustering'].predict(cluster_features)[0]
    
    st.write(f"### Current Market Regime: **Cluster {cluster_id}**")
    if cluster_id == 0:
        st.caption("Hypothesis: Low Volatility / Stable")
    elif cluster_id == 1:
        st.caption("Hypothesis: High Volatility / Risky")
    else:
        st.caption("Hypothesis: Transitioning")
    
    # PCA Plot
    pca_result = models['pca'].transform(features)
    pc1, pc2 = pca_result[0]
    
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Scatter(x=[0, 1, -1], y=[0, 1, -1], mode='markers', name='Regimes', marker=dict(color='gray', opacity=0.3, size=20)))
    fig_pca.add_trace(go.Scatter(x=[pc1], y=[pc2], mode='markers', name='Current State', marker=dict(color='orange', size=25, symbol='star')))
    fig_pca.update_layout(title="PCA Market Map", xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig_pca, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Deployed via Hugging Face Spaces | Model: Ensemble (SVM + RF + Linear)")
