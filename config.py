"""Configuration and constants for StockViz application."""

import os
from typing import Final

# Server Configuration
HOST: Final = os.getenv('FLASK_HOST', '0.0.0.0')
PORT: Final = int(os.getenv('FLASK_PORT', '5000'))
DEBUG: Final = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Security
SECRET_KEY: Final = os.getenv('SECRET_KEY', 'dev-secret-change-in-production')

# File Upload Configuration
UPLOAD_FOLDER: Final = 'uploads'
MAX_FILE_SIZE: Final = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS: Final = {'csv', 'txt'}

# Data Processing Constants
MIN_DATA_ROWS: Final = 100  # Minimum rows needed for LSTM
SEQUENCE_LENGTH: Final = 60  # LSTM sequence length in days
TRAIN_TEST_SPLIT: Final = 0.8

# LSTM Model Configuration
LSTM_EPOCHS: Final = 10
LSTM_BATCH_SIZE: Final = 32
LSTM_UNITS_LAYER_1: Final = 50
LSTM_UNITS_LAYER_2: Final = 50
LSTM_DROPOUT: Final = 0.2
LSTM_OPTIMIZER: Final = 'adam'
LSTM_LOSS: Final = 'mean_squared_error'

# Yahoo Finance Configuration
YFINANCE_PERIOD: Final = '5y'
YFINANCE_TIMEOUT: Final = 30

# Feature Engineering
MOVING_AVERAGE_SHORT: Final = 50
MOVING_AVERAGE_LONG: Final = 200
ROLLING_MEAN_WINDOW: Final = 7
LAG_FEATURES: Final = [1, 2]

# Plotly Configuration
PLOTLY_TEMPLATE: Final = 'plotly_dark'
