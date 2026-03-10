import sys
import streamlit as st
import pandas as pd
import numpy as np
np.bool = np.bool_
from streamlit.web import cli as stcli

try:
    from utils import preprocess_data, train_and_evaluate_lstm
except ImportError:
    pass

def run_dashboard():
    st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
    
    st.title("📈 Stock Price Predictor & Analysis Dashboard")
    st.markdown("""
    Welcome to the **End-to-End Stock Price Prediction Dashboard**.
    Follow this interactive pipeline for **Data Analysis (EDA)**, **Visualization**, **Preprocessing**, and **Prediction using LSTM**.
    """)
    
    st.sidebar.header("📂 1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain Date and Close columns)", type=["csv", "txt"])
    st.sidebar.info("💡 **Tip**: You can download historical APPL, MSFT, or GOOGL stock data from Yahoo Finance and upload the CSV here.")
    
    if uploaded_file is not None:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Load Data
            df = pd.read_csv(uploaded_file)
            st.write("### 🔍 Raw Historical Data (Last 5 Rows)")
            st.dataframe(df.tail())
            
            with st.spinner("Processing data, running EDA, and training LSTM model..."):
                data, target_col, processed_df = preprocess_data(df)
                
                # Phase 1: EDA
                st.write("---")
                st.header("Phase 1: Exploratory Data Analysis (EDA) & Visualization")
                
                col1, col2 = st.columns(2)
                
                # 1. Trend Analysis
                with col1:
                    st.subheader("1. Trend Analysis")
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=processed_df.index, y=processed_df[target_col], mode='lines', name='Close Price'))
                    fig_trend.update_layout(title="Historical Prices (Trend)", template="plotly_dark")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                # 2. Moving Averages
                with col2:
                    st.subheader("2. Moving Averages")
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=processed_df.index, y=processed_df[target_col], name='Close Price'))
                    fig_ma.add_trace(go.Scatter(x=processed_df.index, y=processed_df['SMA_50'], name='50-Day SMA'))
                    fig_ma.add_trace(go.Scatter(x=processed_df.index, y=processed_df['SMA_200'], name='200-Day SMA'))
                    fig_ma.update_layout(title="50 vs 200 Day Moving Averages", template="plotly_dark")
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                col3, col4 = st.columns(2)
                
                # 3. Volatility Analysis
                with col3:
                    st.subheader("3. Volatility Analysis (Daily Returns)")
                    fig_vol_hist = go.Figure()
                    fig_vol_hist.add_trace(go.Histogram(x=processed_df['Daily_Return'], nbinsx=50, marker_color='#10b981'))
                    fig_vol_hist.update_layout(title="Distribution of Returns", template="plotly_dark")
                    st.plotly_chart(fig_vol_hist, use_container_width=True)
                    
                # 4. Correlation Heatmap
                with col4:
                    st.subheader("4. Correlation Heatmap")
                    corr_cols = [target_col, 'SMA_50', 'SMA_200']
                    for extra in ['Open', 'High', 'Low', 'Volume']:
                        if extra in processed_df.columns:
                            corr_cols.append(extra)
                            
                    corr_matrix = processed_df[corr_cols].corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                    fig_corr.update_layout(title="Feature Interaction", template="plotly_dark")
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                st.write("---")
                st.header("Phase 2-4: Preprocessing & LSTM Deep Learning Model")
                
                # Train Model
                seq_length = 60
                model, scaler, predictions, y_test, rmse, mae, train_len = train_and_evaluate_lstm(data, seq_length=seq_length, epochs=10)
                
            st.success("LSTM Model trained successfully!")
            
            # Key performance metrics
            st.write("### 🤖 Artificial Intelligence Model Performance (LSTM)")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}", help="Average distance between prediction and actual in dollars")
            mcol2.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
            mcol3.metric("Dataset Split", f"{train_len} Train / {len(data)-train_len} Test")
            
            # Predict Next Day
            latest_data = data[-seq_length:]
            latest_scaled = scaler.transform(latest_data)
            latest_scaled = np.reshape(latest_scaled, (1, seq_length, 1))
            next_pred = scaler.inverse_transform(model.predict(latest_scaled))[0][0]
            
            st.info(f"🔮 **LSTM Prediction**: Based on the last 60 days, the predicted closing price for the next trading day is **${next_pred:.2f}**")
            
            # Final Plot: Predictions vs Actual
            st.write("#### Final Evaluation: Actual vs Predicted Prices (Test Period)")
            
            test_dates = processed_df.index[train_len:]
            
            fig_final = go.Figure()
            fig_final.add_trace(go.Scatter(x=test_dates, y=y_test.flatten(), name="Actual Price", line=dict(color='#10b981', width=2)))
            fig_final.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), name="Predicted LSTM", line=dict(color='#ef4444', width=2, dash='dash')))
            fig_final.update_layout(
                title="LSTM Model Validation (Test Data)",
                xaxis_title="Date",
                yaxis_title="Stock Price (USD)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig_final, use_container_width=True)
            
        except ValueError as ve:
            st.error(f"Data Error: {str(ve)}")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            st.markdown("Please ensure your CSV is properly formatted with dates and stock prices and has sufficient rows.")
            
    else:
        st.info("👈 Please use the sidebar to upload a CSV file and run the complete LSTM end-to-end pipeline.")

if __name__ == '__main__':
    if st.runtime.exists():
        run_dashboard()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
