import os
from flask import Flask, render_template, request, redirect, flash
import pandas as pd
import numpy as np
np.bool = np.bool_
from ai_model import build_dashboard_data

app = Flask(__name__)
app.secret_key = 'super_secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        file = request.files.get('file')
        
        df = None
        dataset_name = ""
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            dataset_name = f"File: {file.filename}"
        elif ticker:
            try:
                import yfinance as yf
                df = yf.download(ticker, period="5y")
                df.reset_index(inplace=True)
                dataset_name = f"Ticker: {ticker.upper()}"
            except Exception as e:
                flash(f"Error fetching data for {ticker}: {str(e)}")
                return render_template('index.html')
        
        if df is not None and not df.empty:
            try:
                data = build_dashboard_data(df, dataset_name)
                return render_template('dashboard.html', data=data)
            except Exception as e:
                flash(f"Error processing data or training AI model: {str(e)}")
                return render_template('index.html')
            
    return render_template('index.html')

if __name__ == '__main__':
    # Launch on port 5001 as shown in user screenshot
    app.run(port=5001, debug=True)
