# StockViz - Stock Market Analysis & Prediction Dashboard

## Overview
A Flask-based stock market analysis and prediction dashboard that uses LSTM deep learning to analyze historical stock trends and predict future price movements. Features interactive Plotly charts and supports both CSV uploads and Yahoo Finance ticker data fetching.

## Architecture

- **Backend**: Flask (Python), runs on `0.0.0.0:5000`
- **ML Model**: TensorFlow/Keras LSTM neural network (2-layer, 50 units each, Dropout 0.2)
- **Data Source**: Yahoo Finance API (yfinance) or CSV upload
- **Visualizations**: Plotly.js (interactive charts rendered via JSON)
- **Frontend**: Jinja2 templates with custom CSS (glassmorphism dark theme)

## Key Files

- `app.py` - Flask application entry point, routes, file upload handling
- `ai_model.py` - Core ML pipeline: data preprocessing, LSTM model training, chart generation
- `utils.py` - Utility functions for Streamlit version (unused by Flask app)
- `main.py` - Streamlit version of the dashboard (not used in current workflow)
- `templates/index.html` - Main input form (ticker symbol or CSV upload)
- `templates/dashboard.html` - Results dashboard with Plotly charts
- `static/css/style.css` - Dark theme glassmorphism styles
- `uploads/` - Uploaded CSV files stored here

## ML Pipeline (ai_model.py)

1. **Data Processing**: Handles MultiIndex columns, date parsing, target column detection
2. **Feature Engineering**: MA_50, MA_200, Daily Returns, Lag features, Rolling Mean
3. **LSTM Training**: 80/20 train/test split, 60-day sequences, 10 epochs, batch size 32
4. **Evaluation**: RMSE metric, actual vs predicted comparison chart

## Running the App

```bash
python app.py
```
Runs on `http://0.0.0.0:5000`

## Deployment

Configured for Autoscale deployment using Gunicorn:
```
gunicorn --bind=0.0.0.0:5000 --reuse-port --timeout=300 app:app
```

## Dependencies

- flask, pandas, numpy, scikit-learn, tensorflow, keras
- plotly, matplotlib, yfinance, gunicorn
