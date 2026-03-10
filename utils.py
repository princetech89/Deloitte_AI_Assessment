import pandas as pd
import numpy as np
np.bool = np.bool_

def clean_numeric(series):
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    return pd.to_numeric(series, errors='coerce')

def preprocess_data(df):
    """
    Phase 1 & 2: Data Preprocessing
    Features: Lag, Rolling Mean, SMA
    Target: Next day's Close price
    """
    df = df.copy()
    
    # 1. Parse Date and sort
    has_date = False
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            df.dropna(subset=[col], inplace=True)
            df.sort_values(col, inplace=True)
            df.set_index(col, inplace=True)
            has_date = True
            break
            
    if not has_date:
        if df.index.name and ('date' in str(df.index.name).lower() or 'time' in str(df.index.name).lower()):
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            df = df[df.index.notnull()]
            df.sort_index(inplace=True)
        else:
            raise ValueError("Dataset must contain a Date/Time column.")

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)

    # 2. Identify Target Column (Close or Price)
    target_col = None
    for col in ['Close', 'close', 'Adj Close', 'Price', 'price']:
        if col in df.columns:
            target_col = col
            break
            
    if not target_col:
        for col in df.columns:
            temp = clean_numeric(df[col])
            if temp.notna().sum() > 10:
                df['Close'] = temp
                target_col = 'Close'
                break
    else:
        df[target_col] = clean_numeric(df[target_col])

    if not target_col:
        raise ValueError("Dataset must contain a 'Close' or 'Price' column to predict.")

    # Engineering features for EDA
    df['SMA_50'] = df[target_col].rolling(window=50, min_periods=1).mean()
    df['SMA_200'] = df[target_col].rolling(window=200, min_periods=1).mean()
    df['Daily_Return'] = df[target_col].pct_change()
    
    # For LSTM, we can use simple univariate prediction
    data = df.filter([target_col]).values
    
    return data, target_col, df

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def train_and_evaluate_lstm(data, seq_length=60, epochs=10):
    """
    Phase 3 & 4: Train LSTM Model and Evaluate
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense

    if len(data) < seq_length + 10:
        raise ValueError("Dataset is too small for LSTM. Please provide at least 100 days of data.")
        
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    train_data_len = int(np.ceil(len(data) * 0.8))
    train_data = scaled_data[0:train_data_len, :]
    
    X_train, y_train = create_sequences(train_data, seq_length)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    # LSTM Layers (x2)
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    
    # Testing
    test_data = scaled_data[train_data_len - seq_length:, :]
    X_test = []
    y_test = data[train_data_len:, :]
    
    for i in range(seq_length, len(test_data)):
        X_test.append(test_data[i-seq_length:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return model, scaler, predictions, y_test, rmse, mae, train_data_len
