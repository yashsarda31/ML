# app.py
# pip install streamlit yfinance pandas numpy statsmodels scikit-learn tensorflow plotly
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta

st.set_page_config(page_title="Forecast Lab", page_icon="◐", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", Inter, sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.card { background: white; border-radius: 18px; padding: 1.2rem 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #eee; }
.metric { font-size: 13px; color: #6e6e73; margin-bottom: 4px; }
.big { font-size: 28px; font-weight: 600; letter-spacing: -0.02em; }
.stButton>button { border-radius: 12px; padding: 0.6rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big'>Forecast Lab</div>", unsafe_allow_html=True)
st.caption("SARIMAX + LSTM with López de Prado safeguards. Educational use only.")

def frac_diff(series, d=0.4, thres=1e-5):
    w = [1.]
    for k in range(1, len(series)):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres: break
        w.append(w_)
    w = np.array(w[::-1])
    return pd.Series(np.convolve(series, w, mode='valid'), index=series.iloc[len(w)-1:].index)

@st.cache_data(show_spinner=False)
def get_data(ticker, period):
    df = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data returned")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_features(df):
    df = df.copy()
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df = df.dropna()
    for c in ['atr','rsi','log_ret']:
        df[c] = df[c].shift(1)
    return df.dropna()

def purged_split(df, test_size=0.2, embargo=10):
    n = len(df)
    start = int(n * (1-test_size))
    return df.iloc[:start-embargo], df.iloc[start:]

def run_sarimax(df, horizon):
    y = df['log_ret']; X = df[['atr','rsi']]
    train,_ = purged_split(pd.concat([y,X],1))
    model = SARIMAX(train['log_ret'], exog=train[['atr','rsi']], order=(1,0,1),
                    seasonal_order=(1,0,1,5), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    last_exog = X.iloc[-1:].values
    fut_exog = np.repeat(last_exog, horizon, axis=0)
    fc = res.get_forecast(steps=horizon, exog=fut_exog)
    return fc.predicted_mean.values

def run_lstm(df, horizon, lookback):
    feat = df[['log_ret','atr','rsi']].values
    target = df['log_ret'].values
    train,_ = purged_split(df)
    split = len(train)
    scaler = StandardScaler().fit(feat[:split])
    fs = scaler.transform(feat)
    Xs, ys = [], []
    for i in range(lookback, split):
        Xs.append(fs[i-lookback:i]); ys.append(target[i])
    Xs, ys = np.array(Xs), np.array(ys)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback,3)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile('adam','mse')
    model.fit(Xs, ys, epochs=25, batch_size=32, verbose=0,
              validation_split=0.1, callbacks=[EarlyStopping(patience=4, restore_best_weights=True)])
    seq = fs[-lookback:].copy()
    preds = []
    for _ in range(horizon):
        p = model.predict(seq[np.newaxis,:,:], verbose=0)[0,0]
        preds.append(p)
        new = seq[-1].copy(); new[0]=p
        seq = np.vstack([seq[1:], new])
    return np.array(preds)

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "AAPL")
    period = st.selectbox("History", ["2y","5y","10y"], index=1)
    horizon = st.slider("Forecast horizon", 20, 200, 100, 10)
    lookback = st.slider("LSTM lookback", 30, 120, 60, 10)
    models = st.multiselect("Models", ["SARIMAX","LSTM"], default=["SARIMAX","LSTM"])
    run = st.button("Run Forecast", type="primary")

if not run:
    st.info("Enter a ticker and press Run Forecast.")
    st.stop()

try:
    with st.spinner("Downloading data"):
        raw = get_data(ticker, period)
        df = compute_features(raw)
        if len(df) < lookback + 50:
            st.error("Not enough history for this lookback"); st.stop()
except Exception as e:
    st.error(f"Data error: {e}"); st.stop()

last_row = df.iloc[-1]
last_close = float(raw['Close'].iloc[-1])
last_logret = float(last_row['log_ret'])
last_atr = float(last_row['atr'])
last_rsi = float(last_row['rsi'])

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f"<div class='card'><div class='metric'>Last Close</div><div class='big'>{last_close:.2f}</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='card'><div class='metric'>Log Return</div><div class='big'>{last_logret*100:.2f}%</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='card'><div class='metric'>ATR(14)</div><div class='big'>{last_atr:.2f}</div></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='card'><div class='metric'>RSI(14)</div><div class='big'>{last_rsi:.1f}</div></div>", unsafe_allow_html=True)

results = {}
if "SARIMAX" in models:
    with st.spinner("Fitting SARIMAX"):
        results['SARIMAX'] = run_sarimax(df, horizon)
if "LSTM" in models:
    with st.spinner("Training LSTM"):
        results['LSTM'] = run_lstm(df, horizon, lookback)

future_idx = [raw.index[-1] + timedelta(days=i+1) for i in range(horizon)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=raw.index, y=raw['Close'], name='History', line=dict(width=2)))
for name, preds in results.items():
    price_path = last_close * np.exp(np.cumsum(preds))
    fig.add_trace(go.Scatter(x=future_idx, y=price_path, name=name, line=dict(width=2, dash='dot')))
fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=420, legend=dict(orientation='h'), template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

tab1, tab2 = st.tabs(["Forecasts","Features"])
with tab1:
    out = pd.DataFrame({'date': future_idx})
    for k,v in results.items():
        out[k+'_logret'] = v
        out[k+'_price'] = last_close * np.exp(np.cumsum(v))
    st.dataframe(out, use_container_width=True, height=300)
    st.download_button("Download CSV", out.to_csv(index=False), "forecast.csv")
with tab2:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR'))
    f2.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', yaxis='y2'))
    f2.update_layout(yaxis2=dict(overlaying='y', side='right'), height=300, margin=dict(l=0,r=0,t=20,b=0), template='plotly_white')
    st.plotly_chart(f2, use_container_width=True)

st.caption("Built with purged walk-forward splits and embargo. No look-ahead. Not investment advice.")
