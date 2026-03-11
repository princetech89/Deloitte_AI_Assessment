"""
AI Model module - Core machine learning pipeline for stock price prediction.
Includes data preprocessing, LSTM model training, and visualization generation.
"""

import json
import warnings
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

from config import (
    MIN_DATA_ROWS, SEQUENCE_LENGTH, TRAIN_TEST_SPLIT,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_UNITS_LAYER_1, LSTM_UNITS_LAYER_2,
    LSTM_DROPOUT, LSTM_OPTIMIZER, LSTM_LOSS, MOVING_AVERAGE_SHORT,
    MOVING_AVERAGE_LONG, ROLLING_MEAN_WINDOW, LAG_FEATURES, PLOTLY_TEMPLATE
)
from logger import setup_logger

np.bool = np.bool_
warnings.filterwarnings('ignore')
log = setup_logger(__name__)

def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert series to numeric, handling currency symbols and other non-numeric characters.
    
    Args:
        series: Pandas Series to convert
        
    Returns:
        Numeric series with non-numeric values converted to NaN
    """
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    return pd.to_numeric(series, errors='coerce')

def _find_date_column(df: pd.DataFrame) -> str:
    """Find date column in DataFrame columns or index."""
    # Check columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            return col
    
    # Check index
    if df.index.name and ('date' in str(df.index.name).lower() or 'time' in str(df.index.name).lower()):
        return None  # Signal to use index
    
    return None


def _set_date_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse date column and set as index."""
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    df.sort_values(date_col, inplace=True)
    df.set_index(date_col, inplace=True)
    return df


def _normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone information from DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
    return df


def _find_target_column(df: pd.DataFrame) -> str:
    """Find target column (Close price) in DataFrame."""
    target_candidates = ['Close', 'Adj Close', 'Price', 'Last']
    
    for candidate in target_candidates:
        if candidate in df.columns:
            log.info(f"Found target column: {candidate}")
            return candidate
    
    # Fallback: find first numeric column with enough data
    for col in df.columns:
        numeric_col = clean_numeric(df[col])
        if numeric_col.notna().sum() > 10:
            log.info(f"Auto-detected target column: {col}")
            return col
    
    raise ValueError("No Close/Price column found in data")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and prepare data for LSTM model.
    
    Includes:
    - MultiIndex column flattening
    - Date parsing and indexing
    - Target column detection
    - Feature engineering (MAs, returns, lags)
    - Missing value imputation
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        Processed DataFrame ready for LSTM
        
    Raises:
        ValueError: If data processing fails
    """
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Standardize column names
    df.columns = [str(c).title() for c in df.columns]
    
    # Handle date column
    date_col = _find_date_column(df)
    if date_col:
        df = _set_date_index(df, date_col)
    elif df.index.name and ('date' in str(df.index.name).lower() or 'time' in str(df.index.name).lower()):
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        df = df[df.index.notnull()]
        df.sort_index(inplace=True)
    
    df = _normalize_timezone(df)
    
    # Find and normalize target column
    target_col = _find_target_column(df)
    df['Close'] = clean_numeric(df[target_col])
    df.dropna(subset=['Close'], inplace=True)
    
    # Feature Engineering - Phase 2
    # Lag features
    for lag in LAG_FEATURES:
        df[f'Close_t-{lag}'] = df['Close'].shift(lag)
    
    # Rolling statistics
    df['Rolling_Mean_7'] = df['Close'].rolling(window=ROLLING_MEAN_WINDOW, min_periods=1).mean()
    
    # Moving averages
    df['MA_50'] = df['Close'].rolling(window=MOVING_AVERAGE_SHORT, min_periods=1).mean()
    df['MA_200'] = df['Close'].rolling(window=MOVING_AVERAGE_LONG, min_periods=1).mean()
    
    # Daily returns (volatility)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    
    log.info(f"Data processing complete: {len(df)} rows, {len(df.columns)} columns")
    return df

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Input data array (2D)
        seq_length: Sequence length in time steps
        
    Returns:
        Tuple of (X sequences, y targets)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_dashboard_data(raw_df: pd.DataFrame, name: str = "") -> Dict[str, Any]:
    """
    Build complete dashboard with visualizations and LSTM predictions.
    
    Args:
        raw_df: Raw input DataFrame
        name: Dataset name for display
        
    Returns:
        Dictionary with charts (JSON) and metrics
    """
    df = process_data(raw_df.copy())
    
    if len(df) < MIN_DATA_ROWS:
        raise ValueError(f"Insufficient data (found {len(df)} rows). Need at least {MIN_DATA_ROWS} days for LSTM.")
        
    layout_template = PLOTLY_TEMPLATE
    common_layout = dict(
        template=layout_template,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=12),
        margin=dict(l=40, r=20, t=50, b=40)
    )

    # 1. Trend Analysis (Date vs Close)
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#3b82f6', width=2)))
    fig_trend.update_layout(**common_layout, title="Trend Analysis & Historical Prices", yaxis_title="Price")

    # 2. Volatility Analysis (Daily Returns) Line Plot + Histogram
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Daily_Return'], mode='lines', name='Daily Return', line=dict(color='#f59e0b', width=1)))
    fig_vol.update_layout(**common_layout, title="Volatility Analysis (Daily Returns)", yaxis_title="Percentage Change")
    
    fig_vol_hist = go.Figure()
    fig_vol_hist.add_trace(go.Histogram(x=df['Daily_Return'], nbinsx=50, marker_color='#10b981'))
    fig_vol_hist.update_layout(**common_layout, title="Return Distribution (Risk Profile)", xaxis_title="Daily Return", yaxis_title="Frequency")

    # 3. Moving Averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='#3b82f6', width=1.5)))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA_50'], mode='lines', name='50-Day SMA', line=dict(color='#f59e0b', width=2)))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA_200'], mode='lines', name='200-Day SMA', line=dict(color='#ef4444', width=2)))
    fig_ma.update_layout(**common_layout, title="Moving Averages (Golden/Death Crosses)", yaxis_title="Price")

    # 4. Correlation Heatmap
    corr_cols = ['Close', 'MA_50', 'MA_200', 'Rolling_Mean_7', 'Close_t-1', 'Close_t-2']
    # Add open/high/low/vol if present
    for c in ['Open', 'High', 'Low', 'Volume']:
        if c in df.columns: corr_cols.append(c)
    
    corr_matrix = df[corr_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    fig_corr.update_layout(**common_layout, title="Feature Correlation Heatmap")

    # Phase 2: Data Preprocessing for LSTM
    data_values = df.filter(['Close']).values
    
    # Scaling (Normalization)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    # Sequence Generation
    dataset_len = len(scaled_data)
    
    # Phase 4: Data Splitting (Train/Test split)
    train_data_len = int(np.ceil(dataset_len * TRAIN_TEST_SPLIT))
    train_data = scaled_data[0:train_data_len, :]
    
    x_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Phase 3: Train LSTM
    log.info("Training LSTM model...")
    model = Sequential()
    model.add(LSTM(LSTM_UNITS_LAYER_1, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(LSTM_UNITS_LAYER_2, return_sequences=False))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(1))

    model.compile(optimizer=LSTM_OPTIMIZER, loss=LSTM_LOSS)
    model.fit(x_train, y_train, batch_size=LSTM_BATCH_SIZE, epochs=LSTM_EPOCHS, verbose=0)
    log.info("LSTM training complete")

    # Test Data creation
    test_data = scaled_data[train_data_len - SEQUENCE_LENGTH:, :]
    x_test = []
    y_test_real = data_values[train_data_len:, :]
    for i in range(SEQUENCE_LENGTH, len(test_data)):
        x_test.append(test_data[i-SEQUENCE_LENGTH:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # Inverse Scale to Dollars

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_real, predictions))

    # Validation Set for Chart
    train = df[:train_data_len]
    valid = df[train_data_len:].copy()
    valid['Predictions'] = predictions

    # Final Plot: Predictions vs Actuals
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Price', line=dict(color='#10b981', width=2)))
    fig_pred.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Price', line=dict(color='#ef4444', width=2, dash='dash')))
    fig_pred.update_layout(**common_layout, title="LSTM Forecasting (Test set)", yaxis_title="Price (USD)")

    start_date = df.index[0].strftime('%b %d, %Y')
    end_date = df.index[-1].strftime('%b %d, %Y')
    
    return {
        "dataset_name": name,
        "date_range": f"{start_date} - {end_date}",
        "trading_days": len(df),
        "rmse": f"${rmse:.2f}",
        
        # Charts
        "trend_chartJSON": json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder),
        "vol_chartJSON": json.dumps(fig_vol, cls=plotly.utils.PlotlyJSONEncoder),
        "volhist_chartJSON": json.dumps(fig_vol_hist, cls=plotly.utils.PlotlyJSONEncoder),
        "ma_chartJSON": json.dumps(fig_ma, cls=plotly.utils.PlotlyJSONEncoder),
        "corr_chartJSON": json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder),
        "pred_chartJSON": json.dumps(fig_pred, cls=plotly.utils.PlotlyJSONEncoder),
    }

