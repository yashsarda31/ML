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

# --- Fractional Differentiation (Lopez de Prado) ---
def get_weights_floored(d, num_k, floor=1e-4):
    """Calculate fractional differentiation weights and floor them to drop insignificant lags."""
    w_k = np.array([1.0])
    for k in range(1, num_k):
        next_w = -w_k[-1] * (d - k + 1) / k
        w_k = np.append(w_k, next_w)
    # Filter weights below floor
    w_k = w_k[np.abs(w_k) >= floor]
    return w_k.reshape(-1, 1)

def frac_diff(df, d, floor=1e-4):
    """Apply fractional differentiation to a pandas Series."""
    weights = get_weights_floored(d, len(df), floor)
    weights = weights[::-1] # Reverse weights for dot product
    
    res = []
    window = len(weights)
    
    for i in range(window, len(df)):
        res.append(np.dot(weights.T, df.iloc[i - window:i]).item())
        
    # Pad beginning with NaNs to align lengths
    padded_res = [np.nan] * window + res
    return pd.Series(padded_res, index=df.index)

# --- Data Fetching & Feature Engineering ---
@st.cache_data(show_spinner=False)
def fetch_and_prepare_data(ticker, d_value):
    df = yf.download(ticker, start='2020-01-01', end='2026-05-11', progress=False)
    
    if df.empty:
        return None
        
    # Standard Features
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
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
    
    # Fractional Differentiation
    df['Frac_Diff_Close'] = frac_diff(df['Close'], d=d_value)
    
    df.dropna(inplace=True)
    return df

# --- Models ---
def forecast_sarimax(df, steps=100):
    # Using fractionally differenced series as target instead of log returns
    y = df['Frac_Diff_Close']
    X = df[['ATR', 'RSI']]
    
    model = SARIMAX(y, exog=X, order=(1, 0, 1), enforce_stationarity=False)
    results = model.fit(disp=False)
    
    last_exog = X.iloc[-1].values
    future_exog = np.tile(last_exog, (steps, 1))
    
    forecast = results.get_forecast(steps=steps, exog=future_exog)
    return forecast.predicted_mean.values

def forecast_lstm(df, steps=100, lookback=60):
    features = df[['Frac_Diff_Close', 'ATR', 'RSI']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        X_train.append(scaled_data[i-lookback:i])
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    future_predictions = []
    current_batch = scaled_data[-lookback:].reshape(1, lookback, features.shape[1])
    
    for _ in range(steps):
        pred_return = model.predict(current_batch, verbose=0)[0, 0]
        future_predictions.append(pred_return)
        
        last_exog = current_batch[0, -1, 1:]
        new_step = np.insert(last_exog, 0, pred_return).reshape(1, 1, features.shape[1])
        current_batch = np.append(current_batch[:, 1:, :], new_step, axis=1)
        
    dummy_matrix = np.zeros((steps, features.shape[1]))
    dummy_matrix[:, 0] = future_predictions
    rescaled_returns = scaler.inverse_transform(dummy_matrix)[:, 0]
    
    return rescaled_returns

# --- Streamlit UI ---
st.markdown('<p class="main-title">Quant Forecast Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Time Series Prediction with Fractional Differentiation</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Configuration")
    ticker = st.text_input("Asset Ticker", value="AAPL")
    d_value = st.slider("Fractional Differencing (d)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, 
                        help="d=1 is a standard log return. 0 < d < 1 retains memory of the price series.")
    steps = st.number_input("Forecast Horizon (Days)", min_value=10, max_value=200, value=100)
    run_btn = st.button("Generate Forecast")

if run_btn:
    with st.spinner('Fetching data and calculating fractional differentiation...'):
        df = fetch_and_prepare_data(ticker, d_value)
        
    if df is None:
        st.error(f"Failed to fetch data for {ticker}. Please check the ticker symbol.")
    else:
        with col2:
            st.markdown("### Historical Fractional Differentiation")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df.index, y=df['Frac_Diff_Close'], mode='lines', name=f'Frac Diff (d={d_value})', line=dict(color='#0071e3')))
            fig1.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=30, b=0), height=300)
            st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.markdown("### Model Predictions")
        
        col3, col4 = st.columns(2)
        
        with col3:
            with st.spinner('Training SARIMAX Model...'):
                sarimax_preds = forecast_sarimax(df, steps=steps)
            st.success("SARIMAX Training Complete")
            
            fig2 = go.Figure()
            future_dates = pd.date_range(start=df.index[-1], periods=steps+1, freq='B')[1:]
            fig2.add_trace(go.Scatter(x=future_dates, y=sarimax_preds, mode='lines', name='SARIMAX Forecast', line=dict(color='#34c759')))
            fig2.update_layout(title="SARIMAX Frac-Diff Forecast", template='plotly_white', margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with col4:
            with st.spinner('Training LSTM Architecture...'):
                lstm_preds = forecast_lstm(df, steps=steps)
            st.success("LSTM Training Complete")
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=future_dates, y=lstm_preds, mode='lines', name='LSTM Forecast', line=dict(color='#ff9500')))
            fig3.update_layout(title="LSTM Frac-Diff Forecast", template='plotly_white', margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig3, use_container_width=True)
