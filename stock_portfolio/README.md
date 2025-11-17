Portfolio Optimizer: Detailed Workflow
======================================

Project Overview:
-----------------
This Portfolio Optimizer leverages Modern Portfolio Theory (MPT) to find optimal stock allocations 
that maximize the Sharpe ratio. It fetches historical stock data, calculates returns and covariance, 
and uses numerical optimization to suggest allocations. Results can be displayed on console 
or exported to Excel.

Purpose:
--------
- Demonstrate quantitative finance with Python.
- Provide a practical tool for investors to balance risk and return.
- Showcase professional Python: CLI, data handling, optimization, tabulation.

Key Components & Workflow:
--------------------------
1. Input:
   - Stock tickers (space-separated string).
   - Time period (e.g., 1y, 2y, 5y).
   - Optional risk-free rate.
   - Optional save path for Excel export.

2. Data Fetching:
   - Uses yfinance to download adjusted closing prices.
   - Error handling for missing tickers or insufficient data.

3. Returns & Covariance:
   - Daily percentage returns are calculated.
   - Covariance matrix computed for optimization.
   - Regularization added to covariance matrix for numerical stability.

4. Portfolio Optimization:
   - Objective: maximize Sharpe ratio (risk-adjusted return).
   - Uses SciPy's SLSQP method with constraints (weights sum to 1, no short selling).
   - Fallback and error handling if optimization fails.

5. Metrics Calculation:
   - Daily portfolio return, volatility, and Sharpe ratio.
   - Annualized metrics for intuitive interpretation.

6. Output:
   - Console: tabulated allocations and portfolio metrics.
   - Optional Excel file: sheets for allocations and metrics.
   - Interactive mode: guided CLI input.

7. Error Handling:
   - Handles missing tickers, failed downloads, insufficient data.
   - Provides clear messages for invalid inputs or optimization failures.

8. Customization:
   - Risk-free rate can be adjusted.
   - Users can select interactive mode for guided inputs.
   - Excel export path customizable.

9. Limitations:
   - Historical data may not predict future performance.
   - Optimization may fail for highly correlated assets.
   - Not financial advice; for educational and analytical purposes only.

Author:
-------
[Your Name/Username]

License:
--------
MIT

Summary:
--------
This Python-based Portfolio Optimizer combines modern quantitative finance with practical coding. 
It automates the process of fetching stock data, calculating metrics, optimizing allocations, 
and presenting results clearly. Suitable for investors, students, and educators to understand 
risk-return optimization with real stock data.
