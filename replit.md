# StockViz - Stock Market Analysis & Prediction Dashboard

## Overview
A production-quality Flask-based stock market analysis and prediction dashboard using LSTM deep learning. Analyzes historical stock trends and predicts future prices with interactive Plotly visualizations. Supports CSV uploads and Yahoo Finance ticker data.

## Code Quality Features

- **Type Hints**: Full type annotations for all functions
- **Logging**: Comprehensive logging for debugging and monitoring
- **Configuration Management**: Centralized config.py for all constants
- **Input Validation**: File extension validation, ticker validation, data validation
- **Error Handling**: Specific error messages, proper exception handling
- **Code Organization**: Modular design with helper functions and separation of concerns
- **Documentation**: Detailed docstrings for all functions
- **Security**: Secure filename handling, file size limits, environment variable support

## Architecture

- **Backend**: Flask (Python), runs on configurable host/port (default `0.0.0.0:5000`)
- **ML Model**: TensorFlow/Keras LSTM (2 layers, 50 units, Dropout 0.2)
- **Data Source**: Yahoo Finance API or CSV upload
- **Visualizations**: Plotly.js (interactive charts, JSON serialized)
- **Frontend**: Jinja2 templates with glassmorphism CSS theme

## Project Structure

```
├── app.py                    # Main Flask application
├── ai_model.py              # ML pipeline (data processing, LSTM)
├── config.py                # Configuration & constants
├── logger.py                # Logging setup
├── templates/
│   ├── index.html           # Input form
│   └── dashboard.html       # Results dashboard
├── static/css/style.css     # Glassmorphism theme
├── uploads/                 # User CSV storage
└── requirements.txt         # Dependencies
```

## Key Improvements Made

### app.py
- Environment variable configuration (SECRET_KEY, HOST, PORT, DEBUG)
- Separated functions: `fetch_ticker_data()`, `load_csv_data()`
- Secure filename handling with `werkzeug.security`
- File size validation and extension whitelist
- Error handlers for 413 (too large) and 500 (server error)
- Structured logging throughout
- Type hints on all functions
- Detailed docstrings

### ai_model.py
- Imports moved to top (no lazy imports)
- Helper functions: `_find_date_column()`, `_set_date_index()`, `_find_target_column()`
- Centralized configuration constants from config.py
- Type hints and docstrings on all functions
- Logging for debugging data processing
- Reduced code duplication in column detection
- Better error messages with context

### New Files
- **config.py**: Single source of truth for all configuration constants
- **logger.py**: Consistent logging setup across modules

## ML Pipeline

**Phase 1: EDA** - Trend analysis, moving averages, volatility
**Phase 2: Preprocessing** - Feature engineering (MA, returns, lags), missing value handling
**Phase 3: Model Training** - LSTM with configurable parameters
**Phase 4: Evaluation** - RMSE metrics, predictions vs actual comparison

## Configuration

All parameters are centralized in `config.py`:
- LSTM epochs, batch size, units, dropout
- Moving average windows (50/200 day)
- Sequence length (60 days)
- Train/test split (80/20)
- Yahoo Finance period (5 years)
- File upload limits (10 MB)

Override with environment variables:
```bash
SECRET_KEY=prod-key FLASK_PORT=8000 FLASK_DEBUG=False python app.py
```

## Running the App

Development:
```bash
python app.py
```

Production:
```bash
gunicorn --bind=0.0.0.0:5000 --reuse-port --timeout=300 app:app
```

## Testing Features

- Integrated error handlers for graceful degradation
- Input validation on all user inputs
- Comprehensive logging for debugging
- Type hints prevent runtime errors
- Configuration-based flexibility
