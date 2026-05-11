import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings('ignore')

# --- Apple-Style Custom CSS ---
st.set_page_config(page_title="Quant Forecast Studio", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-title {
        font-size: 48px;
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #1d1d1f;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 20px;
        font-weight: 300;
        color: #86868b;
        margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #0071e3;
        color: white;
        border-radius: 980px;
        padding: 12px 24px;
        border: none;
        font-weight: 400;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0077ED;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Fractional Differentiation Math ---
def get_weights_floored(d, num_k, floor=1e-4):
    w_k = np.array([1.0])
    for k in range(1, num_k):
        next_w = -w_k[-1] * (d - k + 1) / k
        w_k = np.append(w_k, next_w)
    w_k = w_k[np.abs(w_k) >= floor]
    return w_k

def frac_diff(series, d, floor=1e-4):
    weights = get_weights_floored(d, len(series), floor)
    weights = weights[::-1] 
    
    res = []
    window = len(weights)
    
    for i in range(window, len(series)):
        res.append(np.dot(weights, series.iloc[i - window:i]).item())
        
    padded_res = [np.nan] * window + res
    return pd.Series(padded_res, index=series.index)

def inverse_frac_diff(original_series, frac_diff_preds, d, floor=1e-4):
    """Inverts the fractional differentiation to recover actual predicted prices."""
    total_len = len(original_series) + len(frac_diff_preds)
    weights = get_weights_floored(d, total_len, floor)
    
    preds_actual = []
    history = original_series.values.tolist()
    
    for pred_fd in frac_diff_preds:
        max_k = min(len(history), len(weights)-1)
        w_past = weights[1:max_k+1]
        x_past = history[-max_k:][::-1] # Reverse to align X_{t-1} with w_1, etc.
        
        # X_t = Y_t - (w_1*X_{t-1} + w_2*X_{t-2} + ...)
        dot_product = np.dot(w_past, x_past)
        x_t = pred_fd - dot_product
        
        preds_actual.append(x_t)
        history.append(x_t)
        
    return preds_actual

# --- Data Engineering ---
@st.cache_data(show_spinner=False)
def fetch_and_prepare_data(ticker, d_value):
    df = yf.download(ticker, start='2020-01-01', end='2026-05-11', progress=False)
    if df.empty:
        return None
        
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Frac_Diff_Close'] = frac_diff(df['Close'], d=d_value)
    df.dropna(inplace=True)
    return df

# --- Upgraded Models ---
def forecast_sarimax(df, steps=5):
    y = df['Frac_Diff_Close']
    X = df[['ATR', 'RSI']]
    
    model = SARIMAX(y, exog=X, order=(1, 0, 1), enforce_stationarity=False)
    results = model.fit(disp=False)
    
    # Still holding exogenous variables constant for this baseline
    last_exog = X.iloc[-1].values
    future_exog = np.tile(last_exog, (steps, 1))
    
    forecast = results.get_forecast(steps=steps, exog=future_exog)
    return forecast.predicted_mean.values

def forecast_lstm_direct(df, steps=5, lookback=30):
    """Direct Multi-Step LSTM - Outputs 'steps' predictions at once."""
    features = df[['Frac_Diff_Close', 'ATR', 'RSI']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_data) - steps + 1):
        X_train.append(scaled_data[i-lookback:i])
        y_train.append(scaled_data[i:i+steps, 0]) # Target is a vector of the next n steps
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(32, return_sequences=False),
        Dense(steps) # Vector output matching the forecast horizon
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Predict directly using the final window
    current_batch = scaled_data[-lookback:].reshape(1, lookback, features.shape[1])
    scaled_preds = model.predict(current_batch, verbose=0)[0]
    
    dummy_matrix = np.zeros((steps, features.shape[1]))
    dummy_matrix[:, 0] = scaled_preds
    rescaled_returns = scaler.inverse_transform(dummy_matrix)[:, 0]
    
    return rescaled_returns

# --- Streamlit UI ---
st.markdown('<p class="main-title">Quant Forecast Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Time Series Prediction (Direct Multi-Step & Price Inversion)</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Configuration")
    ticker = st.text_input("Asset Ticker", value="AAPL")
    d_value = st.slider("Fractional Differencing (d)", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    steps = st.slider("Forecast Horizon (Days)", min_value=1, max_value=14, value=5)
    run_btn = st.button("Generate Forecast")

if run_btn:
    with st.spinner('Preparing data...'):
        df = fetch_and_prepare_data(ticker, d_value)
        
    if df is None:
        st.error(f"Failed to fetch data for {ticker}.")
    else:
        with col2:
            st.markdown("### Historical Price")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#0071e3')))
            fig1.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=250)
            st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.markdown("### Projected Price Forecasts (Inverse Transformed)")
        
        # Limit history plotted on forecast charts for clarity
        recent_history = df['Close'].tail(30)
        future_dates = pd.date_range(start=df.index[-1], periods=steps+1, freq='B')[1:]
        
        col3, col4 = st.columns(2)
        
        with col3:
            with st.spinner('Running SARIMAX...'):
                fd_preds_sarimax = forecast_sarimax(df, steps=steps)
                price_preds_sarimax = inverse_frac_diff(df['Close'], fd_preds_sarimax, d_value)
                
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=recent_history.index, y=recent_history.values, mode='lines', name='History', line=dict(color='#86868b')))
            fig2.add_trace(go.Scatter(x=future_dates, y=price_preds_sarimax, mode='lines+markers', name='SARIMAX', line=dict(color='#34c759', dash='dot')))
            fig2.update_layout(title="SARIMAX (Price Scale)", template='plotly_white', margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with col4:
            with st.spinner('Training Direct LSTM...'):
                fd_preds_lstm = forecast_lstm_direct(df, steps=steps)
                price_preds_lstm = inverse_frac_diff(df['Close'], fd_preds_lstm, d_value)
                
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=recent_history.index, y=recent_history.values, mode='lines', name='History', line=dict(color='#86868b')))
            fig3.add_trace(go.Scatter(x=future_dates, y=price_preds_lstm, mode='lines+markers', name='LSTM', line=dict(color='#ff9500', dash='dot')))
            fig3.update_layout(title="LSTM (Price Scale)", template='plotly_white', margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig3, use_container_width=True)
