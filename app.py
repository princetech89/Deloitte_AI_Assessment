"""
StockViz - Stock Market Analysis & Prediction Dashboard
Main Flask application entry point.
"""

import os
import logging
from typing import Optional, Tuple
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash

from config import (
    HOST, PORT, DEBUG, SECRET_KEY, UPLOAD_FOLDER, 
    MAX_FILE_SIZE, ALLOWED_EXTENSIONS, YFINANCE_PERIOD
)
from logger import setup_logger

# Setup
np.bool = np.bool_
log = setup_logger(__name__)

# Flask app initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fetch_ticker_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        DataFrame with stock data or None on error
        
    Raises:
        ValueError: If ticker data cannot be fetched
    """
    try:
        import yfinance as yf
        log.info(f"Fetching data for ticker: {ticker}")
        df = yf.download(ticker, period=YFINANCE_PERIOD, progress=False)
        df.reset_index(inplace=True)
        log.info(f"Successfully fetched {len(df)} rows for {ticker}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch ticker data for {ticker}: {str(e)}")
        raise ValueError(f"Unable to fetch data for ticker '{ticker}'. Please verify the symbol is valid.")


def load_csv_data(file) -> Optional[pd.DataFrame]:
    """
    Load CSV file from upload.
    
    Args:
        file: File object from request.files
        
    Returns:
        DataFrame with CSV data or None on error
        
    Raises:
        ValueError: If CSV is invalid
    """
    try:
        if not allowed_file(file.filename):
            raise ValueError(f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        log.info(f"Loaded CSV file: {filename}")
        df = pd.read_csv(filepath)
        log.info(f"CSV loaded successfully with {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        log.error(f"Failed to load CSV: {str(e)}")
        raise ValueError(f"Failed to process CSV file: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok", "app": "StockViz"}, 200


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main index route - handles data input and dashboard generation."""
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip()
        file = request.files.get('file')
        
        df = None
        dataset_name = ""
        
        try:
            # Process file upload
            if file and file.filename:
                df = load_csv_data(file)
                dataset_name = f"File: {file.filename}"
                log.info(f"Processing CSV upload: {dataset_name}")
            
            # Process ticker symbol
            elif ticker:
                df = fetch_ticker_data(ticker)
                dataset_name = f"Ticker: {ticker.upper()}"
                log.info(f"Processing ticker: {dataset_name}")
            
            else:
                flash("Please enter a ticker symbol or upload a CSV file.")
                return render_template('index.html')
            
            # Generate dashboard
            if df is not None and not df.empty:
                log.info(f"Generating dashboard for {dataset_name}")
                # Lazy import: only load TensorFlow when processing data
                from ai_model import build_dashboard_data
                data = build_dashboard_data(df, dataset_name)
                return render_template('dashboard.html', data=data)
            
        except ValueError as e:
            log.warning(f"Validation error: {str(e)}")
            flash(str(e))
        except Exception as e:
            log.error(f"Unexpected error in dashboard generation: {str(e)}")
            flash(f"Error processing data: {str(e)}")
        
        return render_template('index.html')
    
    return render_template('index.html')


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded."""
    log.warning("File upload exceeded size limit")
    flash(f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB")
    return render_template('index.html'), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    log.error(f"Internal server error: {str(error)}")
    flash("An unexpected error occurred. Please try again.")
    return render_template('index.html'), 500


if __name__ == '__main__':
    log.info(f"Starting StockViz on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
