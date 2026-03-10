<<<<<<< HEAD
# Deloitte Graduate Hiring Assessment

Name: Prince Chourasiya  
Email: chourasiya8919@gmail.com  
College: Malla Reddy College of Engineering and Technology  
Track: AI & Machine Learning  

## Project Overview
This project is a high-performance **Stock Market Analysis & Prediction Dashboard** developed for the Deloitte Graduate Hiring Assessment. It leverages **Random Forest Machine Learning** to analyze historical stock trends and predict future price movements. The application features a premium, interactive web interface built with Flask and Plotly, providing real-time technical insights.

## Features & capabilities
- **Intelligent Data Acquisition**: 
  - Integrated with **Yahoo Finance API** for real-time ticker data scraping.
  - Robust **CSV Upload System** with deep-cleaning logic (strips currencies, sorts dates automatically).
- **Advanced Feature Engineering**: 
  - Automated calculation of **Technical Indicators**: Simple Moving Averages (SMA-20, SMA-50), Relative Strength Index (RSI), and MACD.
  - Multi-lag feature set (Lag_1, Lag_2) for robust time-series forecasting.
- **Machine Learning Model**: 
  - Uses an optimized **Random Forest Regressor** for stable and efficient price prediction.
  - Includes **Binary Trend Classification** (Up/Down) to evaluate market momentum.
- **Interactive Visualizations**: 
  - **Dynamic Price Charts**: Actual vs. Predicted trends with synced moving averages.
  - **Performance Evaluation**: Live Confusion Matrix, Precision/Recall metrics, and ROC Curve analysis.
- **Premium UX/UI**: 
  - Modern **Glass morphism** design with dark-mode optimized aesthetics.
  - Fully responsive layout with high-visibility **Outfit** typography.

## Technology Stack
- **Languages**: Python
- **ML/DS**: Scikit-learn, Pandas, NumPy
- **Web Backend**: Flask
- **Data Viz**: Plotly.js (Interactive), Matplotlib
- **External APIs**: finance (Yahoo Finance)
- **UI Architecture**: Premium CSS (Glass morphism, CSS Variables, Flex/Grid)

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to: `http://localhost:5001`

*(Note: The interface is optimized for modern browsers like Chrome or Edge)*
=======
