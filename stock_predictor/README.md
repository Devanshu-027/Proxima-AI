Stock Price Predictor: Workflow and Overview
===========================================

Project Overview:
-----------------
This Stock Price Predictor uses an LSTM (Long Short-Term Memory) neural network to predict 
future stock closing prices. Historical stock data is fetched from Yahoo Finance, preprocessed, 
and used to train the model. The model predicts the next day’s closing price and visualizes 
actual vs. predicted prices.

Purpose:
--------
- Demonstrate time-series forecasting with deep learning.
- Provide an educational tool for stock price analysis.
- Showcase professional Python practices: modularity, error handling, documentation, visualization.

Key Components & Workflow:
--------------------------
1. Input Parameters:
   - Ticker symbol (e.g., 'AAPL', 'NHPC.NS').
   - Historical data period (default 2y).
   - Window size for input sequences (default 60 days).

2. Data Fetching:
   - Uses yfinance to download historical stock prices.
   - Keeps only 'Close' prices.
   - Handles errors if data is unavailable or invalid.

3. Preprocessing:
   - Scales data using MinMaxScaler (0-1) for LSTM compatibility.
   - Converts series into sequences (window size) and next-day targets.

4. Train-Test Split:
   - 90% training, 10% testing split.
   - Ensures enough sequences for model training.

5. LSTM Model:
   - Architecture: two LSTM layers with Dropout, Dense layers for output.
   - Loss: Mean Squared Error (MSE).
   - Optimizer: Adam.
   - Early stopping callback to prevent overfitting.

6. Training:
   - Model trained with batch size 32, max epochs 50.
   - Validation performed on test set during training.

7. Evaluation & Visualization:
   - Predictions transformed back to original scale.
   - Plots actual vs. predicted closing prices.
   - Saves plot as {ticker}_pred_vs_actual.png.

8. Next-Day Prediction:
   - Uses the last 'window' days to predict the next closing price.
   - Console output provides numeric prediction.

9. Error Handling:
   - Checks for insufficient data.
   - Catches data fetching errors and exits gracefully.

10. Limitations:
    - Predictions are probabilistic; markets are volatile.
    - Requires sufficient historical data for accurate training.
    - GPU recommended for faster training on large datasets.

Author:
-------
[Your Name/Username]

License:
--------
MIT

Summary:
--------
This Python-based LSTM Stock Predictor demonstrates AI-driven time-series forecasting. 
It fetches real stock data, preprocesses it, trains an LSTM model, evaluates performance, 
plots actual vs. predicted prices, and predicts the next day's closing price.
