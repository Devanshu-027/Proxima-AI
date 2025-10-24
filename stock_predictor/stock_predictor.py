"""
Stock Price Predictor using LSTM Neural Networks
===============================================

Description:
    This script uses AI (LSTM-based neural network) to predict stock closing prices.
    It fetches historical data from Yahoo Finance, preprocesses it, trains a model,
    evaluates performance on test data, and predicts the next day's price.
    Outputs include a plot of actual vs. predicted prices and a console prediction.

Purpose:
    - Demonstrate time-series forecasting with deep learning.
    - Provide a tool for stock analysis (educational; not financial advice).
    - Showcase professional Python practices: modularity, error handling, documentation.

Key Features:
    - Data fetching via yfinance.
    - LSTM model with dropout for regularization.
    - Early stopping to prevent overfitting.
    - Scalable preprocessing (MinMax scaling, sequence creation).
    - Visualization and next-day prediction.

Prerequisites:
    - Python 3.8+
    - Libraries: pip install yfinance scikit-learn tensorflow matplotlib numpy pandas.
    - Internet for data download.

Usage:
    python stock_predictor.py --ticker AAPL --period 2y --window 60
    - --ticker: Stock symbol (default: 'NHPC.NS').
    - --period: Historical data period (default: '2y').
    - --window: Days for input sequences (default: 60).

Outputs:
    - Plot: {ticker}_pred_vs_actual.png (actual vs. predicted on test set).
    - Console: Next-day predicted closing price.

Example:
    python stock_predictor.py --ticker TSLA --period 1y
    # Trains on Tesla data, saves plot, prints prediction.

Limitations:
    - Requires sufficient historical data for training.
    - Predictions are probabilistic; markets are unpredictable.
    - GPU recommended for faster training on large datasets.

Author: [Your Name/Username]
License: MIT (or your choice)
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load_data(ticker, period='2y'):
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL').
        period (str): Time period (e.g., '2y' for 2 years).

    Returns:
        pd.DataFrame: DataFrame with 'Close' prices, indexed by date.

    Raises:
        SystemExit: If data fetching fails.
    """
    try:
        df = yf.download(ticker, period=period, auto_adjust=True)
        df = df[['Close']].dropna()  # Keep only closing prices, drop NaN
        if df.empty:
            raise ValueError("No data available for the given ticker/period.")
        return df
    except Exception as e:
        raise SystemExit(f"Error fetching data for {ticker}: {e}")


def create_sequences(values, window=60):
    """
    Creates input sequences and targets for time-series prediction.

    Args:
        values (np.ndarray): Scaled price values.
        window (int): Number of past days for each sequence.

    Returns:
        tuple: (X, y) where X is sequences, y is next-day targets.
    """
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])  # Past 'window' days
        y.append(values[i])  # Next day's price
    return np.array(X), np.array(y)


def build_model(input_shape):
    """
    Builds and compiles the LSTM model.

    Args:
        input_shape (tuple): Shape of input sequences (window, features).

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),  # First LSTM layer
        Dropout(0.2),  # Dropout for regularization
        LSTM(32),  # Second LSTM layer
        Dropout(0.1),  # More dropout
        Dense(16, activation='relu'),  # Dense layer
        Dense(1)  # Output layer for prediction
    ])
    model.compile(optimizer='adam', loss='mse')  # Adam optimizer, MSE loss
    return model


def main():
    """
    Main function: Parses args, loads data, trains model, predicts, and plots.
    """
    parser = argparse.ArgumentParser(description="AI Stock Price Predictor using LSTM")
    parser.add_argument('--ticker', default='NHPC.NS', help="Stock ticker symbol (e.g., 'AAPL')")
    parser.add_argument('--period', default='2y', help="Historical data period (e.g., '2y')")
    parser.add_argument('--window', type=int, default=60, help="Sequence window size in days")
    args = parser.parse_args()

    # Load and preprocess data
    print(f"Fetching data for {args.ticker} over {args.period}...")
    df = load_data(args.ticker, args.period)
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale to 0-1 for LSTM
    scaled_values = scaler.fit_transform(df.values)

    # Create sequences
    X, y = create_sequences(scaled_values, args.window)
    if len(X) < 10:  # Ensure enough data
        raise SystemExit("Not enough data for the given window size.")

    # Train-test split (90% train, 10% test)
    split_idx = int(0.9 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build and train model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    print("Training model...")
    model.fit(
        X_train, y_train,
        epochs=50,  # Configurable if needed
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate and plot
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)  # Inverse scale
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Actual Prices', color='blue')
    plt.plot(predictions_inv, label='Predicted Prices', color='red')
    plt.title(f'{args.ticker} Stock Price: Actual vs. Predicted (Test Set)')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plot_filename = f'{args.ticker}_pred_vs_actual.png'
    plt.savefig(plot_filename, dpi=150)
    print(f"Plot saved to {plot_filename}")

    # Predict next day
    last_window = scaled_values[-args.window:]  # Last 'window' days
    next_pred_scaled = model.predict(last_window.reshape(1, args.window, 1))
    next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
    print(f"Next-day predicted close for {args.ticker}: {next_pred:.2f}")


if __name__ == '__main__':
    main()
