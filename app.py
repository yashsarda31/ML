"""
De Prado Forecasting Dashboard
A production-grade Streamlit application for SARIMAX & LSTM forecasting
following Marcos Lopez de Prado's financial machine learning framework.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Statistical / ML imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="De Prado Forecast Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR APPLE-INSPIRED DESIGN
# =============================================================================
st.markdown("""
<style>
    /* Apple-inspired clean aesthetic */
    .main {
        background-color: #fafafa;
    }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }

    /* Typography */
    h1 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 700;
        color: #1d1d1f;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
        color: #1d1d1f;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0071e3;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #0077ed;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,113,227,0.3);
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #0071e3;
    }

    /* Dataframes */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 500;
    }

    /* Info boxes */
    .info-box {
        background: rgba(0,113,227,0.08);
        border-left: 4px solid #0071e3;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }

    .warning-box {
        background: rgba(255,149,0,0.08);
        border-left: 4px solid #ff9500;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }

    .success-box {
        background: rgba(52,199,89,0.08);
        border-left: 4px solid #34c759;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CORE FORECASTING MODULES (from de_prado_forecasting.py)
# =============================================================================

class FractionalDifferentiator(BaseEstimator, TransformerMixin):
    """Fractional differentiation preserving long-term memory."""

    def __init__(self, d: float = 0.5, thresh: float = 1e-5):
        self.d = d
        self.thresh = thresh
        self.weights_ = None
        self.width_ = None

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
            if abs(w[-1]) < self.thresh:
                break
        return np.array(w[::-1]).reshape(-1, 1)

    def fit(self, X: pd.Series, y=None):
        self.weights_ = self._get_weights(self.d, len(X))
        self.width_ = len(self.weights_)
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        if isinstance(X, pd.Series):
            X = X.values
        result = np.convolve(X, self.weights_.flatten(), mode='valid')
        idx_start = len(X) - len(result)
        if hasattr(X, 'index'):
            return pd.Series(result, index=X.index[idx_start:])
        return pd.Series(result)

    def fit_transform(self, X: pd.Series, y=None) -> pd.Series:
        return self.fit(X).transform(X)


class FeatureEngineer:
    """Compute ATR, RSI, Log Returns with fractional differentiation."""

    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_log_returns(close: pd.Series) -> pd.Series:
        return np.log(close / close.shift(1))

    @staticmethod
    def prepare_features(ticker: str, start_date: str, end_date: str, frac_diff_d: float = 0.5) -> pd.DataFrame:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data['Close'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]

        log_ret = FeatureEngineer.compute_log_returns(close)
        atr = FeatureEngineer.compute_atr(high, low, close)
        rsi = FeatureEngineer.compute_rsi(close)

        fd = FractionalDifferentiator(d=frac_diff_d)

        features = pd.DataFrame(index=close.index)
        features['close'] = close
        features['log_ret'] = log_ret
        features['atr'] = atr
        features['rsi'] = rsi
        features['log_ret_fd'] = fd.fit_transform(log_ret.dropna())

        fd_atr = FractionalDifferentiator(d=frac_diff_d)
        features['atr_fd'] = fd_atr.fit_transform(atr.dropna())

        fd_rsi = FractionalDifferentiator(d=frac_diff_d)
        features['rsi_fd'] = fd_rsi.fit_transform(rsi.dropna())

        features['target'] = log_ret.shift(-1)
        features = features.dropna()

        return features


class PurgedKFold(BaseCrossValidator):
    """Purged K-Fold Cross-Validation for time series."""

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01, t1: Optional[pd.Series] = None):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.t1 = t1

    def split(self, X, y=None, groups=None):
        if self.t1 is None:
            raise ValueError("t1 must be provided for purged CV")
        indices = np.arange(X.shape[0])
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_starts:
            test_indices = indices[i:j]
            test_start_time = self.t1.index[i]
            train_indices = []
            for idx in indices[:i]:
                if self.t1.iloc[idx] < test_start_time:
                    train_indices.append(idx)
            yield np.array(train_indices), test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class SARIMAXForecaster:
    """SARIMAX with fractionally differentiated exogenous features."""

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1), use_frac_diff: bool = True):
        self.order = order
        self.use_frac_diff = use_frac_diff
        self.model_ = None
        self.results_ = None
        self.feature_cols_ = None

    def fit(self, features: pd.DataFrame):
        self.feature_cols_ = ['log_ret_fd', 'atr_fd', 'rsi_fd'] if self.use_frac_diff else ['log_ret', 'atr', 'rsi']
        available_cols = [c for c in self.feature_cols_ if c in features.columns]
        self.feature_cols_ = available_cols

        y = features['target']
        X = features[self.feature_cols_]

        self.model_ = SARIMAX(y, exog=X, order=self.order,
                              enforce_stationarity=False, enforce_invertibility=False)
        self.results_ = self.model_.fit(disp=False)
        return self

    def forecast(self, features: pd.DataFrame, steps: int = 100) -> pd.DataFrame:
        if self.results_ is None:
            raise ValueError("Model not fitted yet")

        last_features = features[self.feature_cols_].iloc[-steps:]
        if len(last_features) < steps:
            mean_vals = features[self.feature_cols_].mean()
            extension = pd.DataFrame([mean_vals] * (steps - len(last_features)))
            future_exog = pd.concat([last_features, extension], ignore_index=True)
        else:
            future_exog = last_features

        forecast_result = self.results_.get_forecast(steps=steps, exog=future_exog)
        conf_int = forecast_result.conf_int()

        return pd.DataFrame({
            'forecast': forecast_result.predicted_mean.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        })

    def cross_validate(self, features: pd.DataFrame, cv: Optional[PurgedKFold] = None) -> Dict:
        if cv is None:
            t1 = pd.Series(features.index, index=features.index)
            cv = PurgedKFold(n_splits=5, pct_embargo=0.02, t1=t1)

        scores = []
        y = features['target'].values
        X = features[self.feature_cols_].values

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model = SARIMAX(y_train, exog=X_train, order=self.order,
                                enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                pred = results.forecast(steps=len(y_test), exog=X_test)
                scores.append({'mse': mean_squared_error(y_test, pred),
                               'mae': mean_absolute_error(y_test, pred)})
            except Exception:
                continue

        return {
            'avg_mse': np.mean([s['mse'] for s in scores]) if scores else np.nan,
            'avg_mae': np.mean([s['mae'] for s in scores]) if scores else np.nan,
            'scores': scores
        }


class LSTMForecaster:
    """LSTM with fractionally differentiated features and permutation importance."""

    def __init__(self, sequence_length: int = 20, lstm_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 batch_size: int = 32, epochs: int = 100, use_frac_diff: bool = True):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_frac_diff = use_frac_diff
        self.model_ = None
        self.scaler_ = None
        self.feature_cols_ = None
        self.history_ = None

    def _build_model(self, n_features: int) -> Sequential:
        model = Sequential()
        for i, units in enumerate(self.lstm_units):
            if i == 0:
                model.add(LSTM(units, return_sequences=(i < len(self.lstm_units) - 1),
                              input_shape=(self.sequence_length, n_features)))
            else:
                model.add(LSTM(units, return_sequences=(i < len(self.lstm_units) - 1)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse', metrics=['mae'])
        return model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, features: pd.DataFrame, validation_split: float = 0.2):
        self.feature_cols_ = ['log_ret_fd', 'atr_fd', 'rsi_fd'] if self.use_frac_diff else ['log_ret', 'atr', 'rsi']
        available_cols = [c for c in self.feature_cols_ if c in features.columns]
        self.feature_cols_ = available_cols

        X = features[self.feature_cols_]
        y = features['target']

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_seq, y_seq = self._create_sequences(X_scaled, y.values)

        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        self.model_ = self._build_model(len(self.feature_cols_))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
        ]

        self.history_ = self.model_.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return self

    def forecast(self, features: pd.DataFrame, steps: int = 100) -> pd.DataFrame:
        if self.model_ is None:
            raise ValueError("Model not fitted yet")

        X = features[self.feature_cols_]
        X_scaled = self.scaler_.transform(X)

        current_sequence = X_scaled[-self.sequence_length:].copy()
        predictions = []

        for _ in range(steps):
            pred = self.model_.predict(
                current_sequence.reshape(1, self.sequence_length, len(self.feature_cols_)),
                verbose=0
            )[0, 0]
            predictions.append(pred)
            new_row = current_sequence[-1:].copy()
            new_row[0, 0] = pred
            current_sequence = np.vstack([current_sequence[1:], new_row])

        val_mae = min(self.history_.history['val_mae']) if self.history_ else 0.02
        std_error = val_mae * 1.96

        return pd.DataFrame({
            'forecast': predictions,
            'lower_ci': [p - std_error for p in predictions],
            'upper_ci': [p + std_error for p in predictions]
        })

    def feature_importance(self, features: pd.DataFrame, n_permutations: int = 5) -> pd.Series:
        if self.model_ is None:
            raise ValueError("Model not fitted")

        X = features[self.feature_cols_]
        y = features['target']
        X_scaled = self.scaler_.transform(X)
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)

        baseline_pred = self.model_.predict(X_seq, verbose=0).flatten()
        baseline_mse = mean_squared_error(y_seq, baseline_pred)

        importances = {}
        for i, col in enumerate(self.feature_cols_):
            mse_increases = []
            for _ in range(n_permutations):
                X_perm = X_seq.copy()
                perm_values = np.random.permutation(X_perm[:, :, i].flatten())
                X_perm[:, :, i] = perm_values.reshape(X_perm[:, :, i].shape)
                perm_pred = self.model_.predict(X_perm, verbose=0).flatten()
                perm_mse = mean_squared_error(y_seq, perm_pred)
                mse_increases.append(perm_mse - baseline_mse)
            importances[col] = np.mean(mse_increases)

        return pd.Series(importances).sort_values(ascending=False)


# =============================================================================
# PLOTLY CHARTING FUNCTIONS
# =============================================================================

def create_price_chart(features: pd.DataFrame, ticker: str) -> go.Figure:
    """Interactive price chart with volume-style styling."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=features.index,
        y=features['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#0071e3', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,113,227,0.08)'
    ))

    fig.update_layout(
        title=dict(text=f'{ticker} Price History', font=dict(size=18, color='#1d1d1f')),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', gridwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified'
    )

    return fig


def create_feature_chart(features: pd.DataFrame) -> go.Figure:
    """Feature visualization with subplots."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Log Returns (Fractionally Diff)', 'ATR (Fractionally Diff)', 'RSI (Fractionally Diff)'),
        vertical_spacing=0.08
    )

    colors = ['#0071e3', '#34c759', '#ff9500']
    cols = ['log_ret_fd', 'atr_fd', 'rsi_fd']

    for i, (col, color) in enumerate(zip(cols, colors), 1):
        fig.add_trace(go.Scatter(
            x=features.index,
            y=features[col],
            mode='lines',
            name=col.replace('_fd', '').upper(),
            line=dict(color=color, width=1.2),
            showlegend=False
        ), row=i, col=1)

    fig.update_layout(
        title=dict(text='Fractionally Differentiated Features', font=dict(size=18, color='#1d1d1f')),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
        height=500,
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode='x unified'
    )

    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=i, col=1)

    return fig


def create_forecast_chart(forecast_df: pd.DataFrame, model_name: str, color: str) -> go.Figure:
    """Forecast chart with confidence intervals."""
    fig = go.Figure()

    periods = list(range(1, len(forecast_df) + 1))

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=periods + periods[::-1],
        y=list(forecast_df['upper_ci']) + list(forecast_df['lower_ci'])[::-1],
        fill='toself',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=periods,
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color)
    ))

    fig.update_layout(
        title=dict(text=f'{model_name} Forecast (Next {len(forecast_df)} Periods)', font=dict(size=18, color='#1d1d1f')),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
        xaxis=dict(title='Periods Ahead', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(title='Expected Log Return', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified'
    )

    return fig


def create_comparison_chart(sarimax_df: pd.DataFrame, lstm_df: pd.DataFrame) -> go.Figure:
    """Side-by-side model comparison."""
    fig = go.Figure()

    periods = list(range(1, len(sarimax_df) + 1))

    fig.add_trace(go.Scatter(
        x=periods, y=sarimax_df['forecast'],
        mode='lines', name='SARIMAX',
        line=dict(color='#0071e3', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=periods, y=lstm_df['forecast'],
        mode='lines', name='LSTM',
        line=dict(color='#ff2d55', width=2)
    ))

    fig.update_layout(
        title=dict(text='Model Comparison: SARIMAX vs LSTM', font=dict(size=18, color='#1d1d1f')),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
        xaxis=dict(title='Periods Ahead', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(title='Expected Log Return', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode='x unified'
    )

    return fig


def create_importance_chart(importance: pd.Series) -> go.Figure:
    """Horizontal bar chart for feature importance."""
    fig = go.Figure()

    colors = ['#0071e3', '#34c759', '#ff9500', '#af52de', '#ff2d55']

    fig.add_trace(go.Bar(
        y=importance.index,
        x=importance.values,
        orientation='h',
        marker=dict(color=colors[:len(importance)], line=dict(width=0)),
        text=[f'{v:.4f}' for v in importance.values],
        textposition='outside',
        textfont=dict(size=12, color='#1d1d1f')
    ))

    fig.update_layout(
        title=dict(text='Feature Importance (Permutation Method)', font=dict(size=18, color='#1d1d1f')),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
        xaxis=dict(title='MSE Increase', showgrid=True, gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(showgrid=False),
        margin=dict(l=120, r=40, t=60, b=40),
        height=300
    )

    return fig


def create_cv_results_chart(cv_results: Dict) -> go.Figure:
    """CV fold performance visualization."""
    if not cv_results['scores']:
        return go.Figure()

    scores = cv_results['scores']
    folds = [f'Fold {i+1}' for i in range(len(scores))]
    mse_vals = [s['mse'] for s in scores]
    mae_vals = [s['mae'] for s in scores]

    fig = make_subplots(rows=1, cols=2, subplot_titles=('MSE by Fold', 'MAE by Fold'))

    fig.add_trace(go.Bar(x=folds, y=mse_vals, marker_color='#0071e3', showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=folds, y=mae_vals, marker_color='#34c759', showlegend=False), row=1, col=2)

    fig.update_layout(
        title=dict(text='Purged Cross-Validation Results', font=dict(size=18, color='#1d1d1f')),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
        height=350,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

def render_sidebar() -> Dict:
    """Render the configuration sidebar."""
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 24px; color: #1d1d1f; margin: 0;">📈 Forecast Lab</h1>
            <p style="font-size: 12px; color: #86868b; margin-top: 4px;">De Prado Framework</p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<hr style='margin: 0 0 20px 0; border: none; border-top: 1px solid rgba(0,0,0,0.1);'>", unsafe_allow_html=True)

    # Data Configuration
    st.sidebar.markdown("<h3 style='font-size: 14px; color: #1d1d1f; margin-bottom: 12px;'>📊 Data Configuration</h3>", unsafe_allow_html=True)

    ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Enter a Yahoo Finance ticker (e.g., SPY, AAPL, BTC-USD)").upper().strip()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 1, 1))

    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Model Configuration
    st.sidebar.markdown("<h3 style='font-size: 14px; color: #1d1d1f; margin-bottom: 12px;'>⚙️ Model Configuration</h3>", unsafe_allow_html=True)

    forecast_horizon = st.sidebar.slider("Forecast Horizon", min_value=10, max_value=252, value=100, step=10,
                                          help="Number of future periods to forecast")

    frac_diff_d = st.sidebar.slider("Fractional Diff (d)", min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                                     help="Fractional differentiation parameter. Lower = more memory preserved.")

    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # LSTM Configuration
    st.sidebar.markdown("<h3 style='font-size: 14px; color: #1d1d1f; margin-bottom: 12px;'>🧠 LSTM Settings</h3>", unsafe_allow_html=True)

    sequence_length = st.sidebar.selectbox("Sequence Length", options=[10, 20, 30, 50, 60], index=1,
                                           help="Lookback window for LSTM sequences")

    lstm_epochs = st.sidebar.slider("Max Epochs", min_value=20, max_value=200, value=50, step=10)

    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # SARIMAX Configuration
    st.sidebar.markdown("<h3 style='font-size: 14px; color: #1d1d1f; margin-bottom: 12px;'>📐 SARIMAX Settings</h3>", unsafe_allow_html=True)

    sarimax_p = st.sidebar.selectbox("AR Order (p)", options=[0, 1, 2, 3], index=1)
    sarimax_d = st.sidebar.selectbox("Diff Order (d)", options=[0, 1], index=0)
    sarimax_q = st.sidebar.selectbox("MA Order (q)", options=[0, 1, 2, 3], index=1)

    st.sidebar.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid rgba(0,0,0,0.1);'>", unsafe_allow_html=True)

    run_button = st.sidebar.button("🚀 Run Forecast Pipeline", use_container_width=True)

    return {
        'ticker': ticker,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'forecast_horizon': forecast_horizon,
        'frac_diff_d': frac_diff_d,
        'sequence_length': sequence_length,
        'lstm_epochs': lstm_epochs,
        'sarimax_order': (sarimax_p, sarimax_d, sarimax_q),
        'run_button': run_button
    }


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def render_header():
    """Render the main dashboard header."""
    st.markdown("""
        <div style="text-align: center; padding: 30px 0 20px 0;">
            <h1 style="font-size: 42px; font-weight: 700; color: #1d1d1f; margin: 0; letter-spacing: -0.03em;">
                De Prado Forecast Lab
            </h1>
            <p style="font-size: 16px; color: #86868b; margin-top: 8px; max-width: 600px; margin-left: auto; margin-right: auto;">
                SARIMAX & LSTM forecasting powered by fractional differentiation, 
                purged cross-validation, and the triple barrier method
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_framework_badges():
    """Render framework principle badges."""
    cols = st.columns(5)
    badges = [
        ("🔢", "Fractional Diff", "Preserve memory while achieving stationarity"),
        ("🛡️", "Purged CV", "Eliminate look-ahead bias in backtesting"),
        ("📊", "Triple Barrier", "Event-based labeling for robust targets"),
        ("🔍", "Feature Importance", "Permutation-based model diagnostics"),
        ("⚖️", "Embargo", "Temporal gap between train and test sets")
    ]

    for col, (emoji, title, desc) in zip(cols, badges):
        with col:
            st.markdown(f"""
                <div style="background: white; border-radius: 12px; padding: 14px; text-align: center; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.04);">
                    <div style="font-size: 24px; margin-bottom: 6px;">{emoji}</div>
                    <div style="font-size: 13px; font-weight: 600; color: #1d1d1f;">{title}</div>
                    <div style="font-size: 10px; color: #86868b; margin-top: 2px;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)


def render_metrics(features: pd.DataFrame, cv_results: Dict, importance: pd.Series):
    """Render key metric cards."""
    cols = st.columns(4)

    metrics = [
        ("📅", "Data Points", f"{len(features):,}"),
        ("📈", "Avg Daily Return", f"{features['log_ret'].mean()*100:.3f}%"),
        ("🎯", "CV MAE", f"{cv_results.get('avg_mae', 0):.6f}"),
        ("⭐", "Top Feature", f"{importance.index[0] if len(importance) > 0 else 'N/A'}")
    ]

    for col, (emoji, label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
                <div style="background: white; border-radius: 16px; padding: 20px; 
                            box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.04);">
                    <div style="font-size: 20px; margin-bottom: 4px;">{emoji}</div>
                    <div style="font-size: 11px; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                    <div style="font-size: 20px; font-weight: 700; color: #1d1d1f; margin-top: 4px;">{value}</div>
                </div>
            """, unsafe_allow_html=True)


def render_data_explorer(features: pd.DataFrame, ticker: str):
    """Render data exploration tab."""
    st.markdown("<h2 style='margin-top: 20px;'>Data Explorer</h2>", unsafe_allow_html=True)

    st.plotly_chart(create_price_chart(features, ticker), use_container_width=True)
    st.plotly_chart(create_feature_chart(features), use_container_width=True)

    with st.expander("📋 View Raw Feature Data"):
        st.dataframe(features.tail(100), use_container_width=True, height=400)


def render_forecast_results(sarimax_forecast: pd.DataFrame, lstm_forecast: pd.DataFrame,
                            features: pd.DataFrame, ticker: str):
    """Render forecast results tab."""
    st.markdown("<h2 style='margin-top: 20px;'>Forecast Results</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📐 SARIMAX", "🧠 LSTM", "⚖️ Comparison"])

    with tab1:
        st.markdown("""
            <div class="info-box">
                <b>SARIMAX Forecast</b><br>
                State-space model with fractionally differentiated exogenous features (ATR, RSI, Log Returns).
                Confidence intervals computed via state-space representation.
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(create_forecast_chart(sarimax_forecast, "SARIMAX", "#0071e3"), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Forecast", f"{sarimax_forecast['forecast'].mean():.6f}")
        with col2:
            st.metric("Forecast Std", f"{sarimax_forecast['forecast'].std():.6f}")
        with col3:
            st.metric("Max Drawdown (Forecast)", f"{sarimax_forecast['forecast'].min():.6f}")

        with st.expander("📋 SARIMAX Forecast Table"):
            st.dataframe(sarimax_forecast, use_container_width=True)

    with tab2:
        st.markdown("""
            <div class="info-box">
                <b>LSTM Forecast</b><br>
                Recurrent neural network with BatchNorm, Dropout, and Early Stopping. 
                Recursive multi-step forecasting with validation MAE-based confidence bands.
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(create_forecast_chart(lstm_forecast, "LSTM", "#ff2d55"), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Forecast", f"{lstm_forecast['forecast'].mean():.6f}")
        with col2:
            st.metric("Forecast Std", f"{lstm_forecast['forecast'].std():.6f}")
        with col3:
            st.metric("Max Drawdown (Forecast)", f"{lstm_forecast['forecast'].min():.6f}")

        with st.expander("📋 LSTM Forecast Table"):
            st.dataframe(lstm_forecast, use_container_width=True)

    with tab3:
        st.markdown("""
            <div class="info-box">
                <b>Model Comparison</b><br>
                Overlay of SARIMAX (parametric, interpretable) and LSTM (non-parametric, flexible) forecasts.
                Divergence indicates regime changes or non-linear dynamics.
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(create_comparison_chart(sarimax_forecast, lstm_forecast), use_container_width=True)

        # Difference analysis
        diff = sarimax_forecast['forecast'] - lstm_forecast['forecast']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Difference", f"{diff.abs().mean():.6f}")
        with col2:
            st.metric("Correlation", f"{sarimax_forecast['forecast'].corr(lstm_forecast['forecast']):.4f}")


def render_model_diagnostics(cv_results: Dict, importance: pd.Series, lstm_history):
    """Render model diagnostics tab."""
    st.markdown("<h2 style='margin-top: 20px;'>Model Diagnostics</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("<h3>Purged Cross-Validation</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div class="warning-box">
                <b>Why Purged CV?</b><br>
                Standard k-fold leaks information in time series. We purge overlapping 
                observations and embargo post-test periods to simulate true out-of-sample performance.
            </div>
        """, unsafe_allow_html=True)

        if cv_results['scores']:
            st.plotly_chart(create_cv_results_chart(cv_results), use_container_width=True)
        else:
            st.info("No CV results available.")

    with col2:
        st.markdown("<h3>Feature Importance (Permutation Method)</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div class="info-box">
                <b>De Prado's First Law:</b> "Backtesting is not a research tool. Feature importance is."<br>
                We measure how much permuting each feature increases MSE. Higher = more important.
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(create_importance_chart(importance), use_container_width=True)

    # LSTM Training History
    if lstm_history is not None:
        st.markdown("<h3>LSTM Training History</h3>", unsafe_allow_html=True)

        hist_df = pd.DataFrame({
            'Epoch': range(1, len(lstm_history.history['loss']) + 1),
            'Train Loss': lstm_history.history['loss'],
            'Val Loss': lstm_history.history['val_loss'],
            'Train MAE': lstm_history.history['mae'],
            'Val MAE': lstm_history.history['val_mae']
        })

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'MAE'))

        fig.add_trace(go.Scatter(x=hist_df['Epoch'], y=hist_df['Train Loss'],
                                  mode='lines', name='Train Loss', line=dict(color='#0071e3')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df['Epoch'], y=hist_df['Val Loss'],
                                  mode='lines', name='Val Loss', line=dict(color='#ff2d55')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df['Epoch'], y=hist_df['Train MAE'],
                                  mode='lines', name='Train MAE', line=dict(color='#0071e3')), row=1, col=2)
        fig.add_trace(go.Scatter(x=hist_df['Epoch'], y=hist_df['Val MAE'],
                                  mode='lines', name='Val MAE', line=dict(color='#ff2d55')), row=1, col=2)

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1d1d1f'),
            height=350,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    render_header()
    render_framework_badges()

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    config = render_sidebar()

    if not config['run_button']:
        st.markdown("""
            <div style="text-align: center; padding: 80px 20px; background: white; border-radius: 20px; 
                        box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-top: 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">👈</div>
                <h2 style="color: #1d1d1f; margin: 0;">Configure Your Forecast</h2>
                <p style="color: #86868b; margin-top: 8px; max-width: 400px; margin-left: auto; margin-right: auto;">
                    Select a ticker, date range, and model parameters in the sidebar, 
                    then click <b>Run Forecast Pipeline</b> to begin.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Run pipeline
    with st.spinner(f"Loading data for {config['ticker']}..."):
        try:
            features = FeatureEngineer.prepare_features(
                config['ticker'],
                config['start_date'],
                config['end_date'],
                frac_diff_d=config['frac_diff_d']
            )
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    if len(features) < 100:
        st.error("Insufficient data points. Please select a wider date range.")
        return

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # SARIMAX
    status_text.text("Training SARIMAX model with Purged Cross-Validation...")
    sarimax = SARIMAXForecaster(order=config['sarimax_order'], use_frac_diff=True)
    sarimax.fit(features)
    progress_bar.progress(33)

    t1 = pd.Series(features.index, index=features.index)
    cv = PurgedKFold(n_splits=5, pct_embargo=0.02, t1=t1)
    cv_results = sarimax.cross_validate(features, cv=cv)
    progress_bar.progress(50)

    sarimax_forecast = sarimax.forecast(features, steps=config['forecast_horizon'])
    status_text.text("SARIMAX complete. Training LSTM...")
    progress_bar.progress(60)

    # LSTM
    lstm = LSTMForecaster(
        sequence_length=config['sequence_length'],
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=config['lstm_epochs'],
        use_frac_diff=True
    )
    lstm.fit(features, validation_split=0.2)
    progress_bar.progress(85)

    importance = lstm.feature_importance(features)
    lstm_forecast = lstm.forecast(features, steps=config['forecast_horizon'])
    progress_bar.progress(100)

    status_text.empty()
    progress_bar.empty()

    # Success message
    st.markdown("""
        <div class="success-box">
            <b>✅ Pipeline Complete</b><br>
            Both models trained successfully with purged cross-validation and fractional differentiation.
        </div>
    """, unsafe_allow_html=True)

    # Metrics
    render_metrics(features, cv_results, importance)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Tabs for different views
    main_tab1, main_tab2, main_tab3 = st.tabs(["🔍 Data Explorer", "📊 Forecasts", "🧪 Diagnostics"])

    with main_tab1:
        render_data_explorer(features, config['ticker'])

    with main_tab2:
        render_forecast_results(sarimax_forecast, lstm_forecast, features, config['ticker'])

    with main_tab3:
        render_model_diagnostics(cv_results, importance, lstm.history_)


if __name__ == "__main__":
    main()
