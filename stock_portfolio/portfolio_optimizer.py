"""
Portfolio Optimizer using Modern Portfolio Theory (MPT)
======================================================

Description:
    This script optimizes a stock portfolio to maximize the Sharpe ratio (risk-adjusted returns)
    using MPT. It fetches historical data, calculates returns/covariance, and uses numerical
    optimization to find optimal allocations. Results include allocations, daily/annualized
    metrics, and optional export to Excel.

Purpose:
    - Demonstrate portfolio optimization with quantitative finance.
    - Provide a tool for investors to balance risk and return.
    - Showcase professional Python: CLI with Click, data handling, optimization.

Key Features:
    - Data fetching via yfinance with error checking.
    - Sharpe ratio maximization using SciPy's minimize (SLSQP).
    - Annualized metrics for better interpretation.
    - Tabulated output with tabulate library.
    - Interactive mode for user-friendly input.
    - Optional Excel export.

Prerequisites:
    - Python 3.8+.
    - Libraries: pip install click numpy pandas yfinance scipy tabulate openpyxl.
    - Internet for stock data.

Usage:
    python portfolio_optimizer.py --tickers "AAPL MSFT" --period 1y --rf 0.02
    - --tickers: Stock symbols as a space-separated string (e.g., "AAPL MSFT GOOGL").
    - --period: Data period (e.g., '1y').
    - --rf: Annual risk-free rate (default 0.0).
    - --save-results: Path to save Excel (optional).
    - --interactive: Guided input mode.

Outputs:
    - Console: Tabulated allocations and metrics.
    - Optional: Excel file with allocations and metrics sheets.

Example:
    python portfolio_optimizer.py --tickers "TSLA NVDA" --period 2y --interactive
    # Fetches data, optimizes, displays results.

Limitations:
    - Assumes past performance predicts future (not always true).
    - Requires sufficient data; short periods may fail.
    - Optimization may not converge for highly correlated assets.
    - Not financial advice; consult professionals.

Author: [Your Name/Username]
License: MIT
"""

import click
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from tabulate import tabulate
import sys

TRADING_DAYS_PER_YEAR = 252  # Standard assumption for annualizing returns/volatility


def fetch_stock_data(tickers: list, period: str) -> pd.DataFrame:
    """
    Fetches adjusted close prices for the given tickers over the specified period.

    Args:
        tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): Time period (e.g., '1y' for 1 year).

    Returns:
        pd.DataFrame: DataFrame with closing prices, indexed by date.

    Raises:
        ValueError: If data cannot be fetched or is incomplete.
    """
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
        if data.empty:
            raise ValueError("No data retrieved. Check internet connection, tickers, or try a different period (e.g., '2y'). Yahoo Finance may be temporarily unavailable.")
        if len(data.columns) != len(tickers):
            missing = [t for t in tickers if t not in data.columns]
            raise ValueError(f"Unable to fetch data for tickers: {missing}. Check ticker symbols or try a different period.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data: {e}. Ensure internet access and valid tickers.")


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily percentage returns from price data.

    Args:
        data (pd.DataFrame): Price data.

    Returns:
        pd.DataFrame: Daily returns.

    Raises:
        ValueError: If insufficient data.
    """
    returns = data.pct_change(fill_method=None).dropna()  # Fixed: Use fill_method=None to avoid deprecation warning
    if returns.empty:
        raise ValueError("Insufficient data to calculate returns. Try a longer period (e.g., '2y'), different tickers, or check data availability.")
    return returns


def compute_portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, rf: float = 0.0) -> tuple:
    """
    Computes portfolio return, volatility, and Sharpe ratio.

    Args:
        weights (np.ndarray): Portfolio weights.
        mean_returns (np.ndarray): Mean daily returns.
        cov_matrix (np.ndarray): Covariance matrix.
        rf (float): Daily risk-free rate.

    Returns:
        tuple: (portfolio_return, portfolio_volatility, sharpe_ratio).
    """
    portfolio_return = np.dot(mean_returns, weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - rf / TRADING_DAYS_PER_YEAR) / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio


def negative_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, rf: float) -> float:
    """
    Objective function for minimization: negative Sharpe ratio (to maximize Sharpe).

    Args:
        weights (np.ndarray): Portfolio weights.
        mean_returns (np.ndarray): Mean returns.
        cov_matrix (np.ndarray): Covariance matrix.
        rf (float): Risk-free rate.

    Returns:
        float: Negative Sharpe ratio.
    """
    _, _, sharpe = compute_portfolio_stats(weights, mean_returns, cov_matrix, rf)
    return -sharpe


def optimize_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray, rf: float) -> np.ndarray:
    """
    Optimizes portfolio weights to maximize Sharpe ratio.

    Args:
        mean_returns (np.ndarray): Mean returns.
        cov_matrix (np.ndarray): Covariance matrix.
        rf (float): Risk-free rate.

    Returns:
        np.ndarray: Optimal weights.

    Raises:
        ValueError: If optimization fails.
    """
    n = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n))  # No short selling
    initial_weights = np.array([1.0 / n] * n)  # Equal weights start
    
    # Regularize covariance for numerical stability
    cov_matrix += np.eye(n) * 1e-8
    
    result = minimize(
        negative_sharpe, initial_weights,
        args=(mean_returns, cov_matrix, rf),
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}. Try different tickers or period.")
    return result.x


def display_results(tickers: list, weights: np.ndarray, daily_return: float, daily_volatility: float, sharpe: float, rf: float, save_path: str = None):
    """
    Displays optimized portfolio results in tabulated format.
    Optionally saves to Excel.

    Args:
        tickers (list): List of tickers.
        weights (np.ndarray): Optimal weights.
        daily_return (float): Daily portfolio return.
        daily_volatility (float): Daily volatility.
        sharpe (float): Daily Sharpe ratio.
        rf (float): Annual risk-free rate.
        save_path (str, optional): Path to save Excel file.
    """
    # Annualize metrics
    annual_return = (1 + daily_return) ** TRADING_DAYS_PER_YEAR - 1
    annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    annual_sharpe = sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Prepare data for tables
    allocations = [[ticker, f"{weight * 100:.2f}%"] for ticker, weight in zip(tickers, weights)]
    summary = [
        ["Risk-Free Rate", f"{rf * 100:.2f}%"],
        ["Daily Return", f"{daily_return:.4f}"],
        ["Daily Volatility", f"{daily_volatility:.4f}"],
        ["Daily Sharpe Ratio", f"{sharpe:.4f}"],
        ["Annual Return", f"{annual_return:.4f}"],
        ["Annual Volatility", f"{annual_volatility:.4f}"],
        ["Annual Sharpe Ratio", f"{annual_sharpe:.4f}"],
    ]
    
    # Display tables
    click.echo("\n" + "="*50)
    click.echo("OPTIMAL PORTFOLIO ALLOCATIONS")
    click.echo("="*50)
    click.echo(tabulate(allocations, headers=["Ticker", "Allocation"], tablefmt="grid"))
    
    click.echo("\n" + "="*50)
    click.echo("PORTFOLIO METRICS")
    click.echo("="*50)
    click.echo(tabulate(summary, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Save to Excel if requested
    if save_path:
        df_alloc = pd.DataFrame(allocations, columns=["Ticker", "Allocation"])
        df_summary = pd.DataFrame(summary, columns=["Metric", "Value"])
        try:
            with pd.ExcelWriter(save_path) as writer:
                df_alloc.to_excel(writer, sheet_name="Allocations", index=False)
                df_summary.to_excel(writer, sheet_name="Metrics", index=False)
            click.echo(f"\nResults saved to {save_path}")
        except Exception as e:
            click.echo(f"Error saving to {save_path}: {e}", err=True)


@click.command()
@click.option('--tickers', help='List of stock tickers separated by spaces (e.g., "AAPL MSFT").')
@click.option('--period', default='1y', help='Time period for data (e.g., 1y, 2y, 5y).')
@click.option('--rf', default=0.0, type=float, help='Risk-free rate (annual, e.g., 0.02 for 2%).')
@click.option('--save-results', type=click.Path(), help='Path to save results as Excel (optional).')
@click.option('--interactive', is_flag=True, help='Run in interactive mode for guided input.')
def main(tickers, period, rf, save_results, interactive):
    """
    Portfolio Optimizer: Maximizes Sharpe ratio for given stocks.
    Example: python portfolio_optimizer.py --tickers "AAPL MSFT GOOGL" --period 2y
    """
    if interactive:
        click.echo("Welcome to the Portfolio Optimizer (Interactive Mode)!")
        tickers = click.prompt("Enter tickers separated by spaces", type=str)
        period = click.prompt("Enter period (e.g., 1y, 2y)", default='1y')
        rf = click.prompt("Enter risk-free rate (annual, e.g., 0.02)", default=0.0, type=float)
        save_results = click.prompt("Save results to file? Enter path or leave blank", default="", type=str) or None
    
    # Split tickers into list if provided
    if tickers:
        tickers = tickers.split()  # Split space-separated string into list
    else:
        tickers = []
    
    if not tickers:
        click.echo("Error: No tickers provided. Use --tickers or --interactive.")
        sys.exit(1)
    
    try:
        click.echo(f"Fetching data for {tickers} over {period}...")
        data = fetch_stock_data(tickers, period)
        returns = calculate_returns(data)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        click.echo("Optimizing portfolio...")
        optimal_weights = optimize_portfolio(mean_returns, cov_matrix, rf)
        
        daily_return, daily_volatility, sharpe = compute_portfolio_stats(optimal_weights, mean_returns, cov_matrix, rf)
        display_results(tickers, optimal_weights, daily_return, daily_volatility, sharpe, rf, save_results)
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
