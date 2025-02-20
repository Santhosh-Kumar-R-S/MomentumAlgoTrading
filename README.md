# MomentumAlgoTrading
# Advanced Momentum Trading Strategy

This project implements an advanced momentum-based trading strategy using Python. It incorporates technical indicators, risk management, machine learning for signal enhancement, and backtesting.

## Features
- **Historical Data Fetching**: Uses `yFinance` to fetch stock data.
- **Momentum and Technical Indicators**: Calculates Momentum, RSI, MACD, and ATR.
- **Trading Signals**: Generates buy/sell signals based on momentum.
- **Risk Management**: Implements stop-loss and take-profit levels.
- **Backtesting**: Simulates trading performance over historical data.
- **Momentum Optimization**: Finds the best momentum period for trading.
- **Machine Learning Model**: Uses a `RandomForestClassifier` to refine trading signals.
- **Performance Visualization**: Plots portfolio value over time.

## Installation
Ensure you have Python installed and install dependencies using:

```bash
pip install pandas numpy matplotlib yfinance talib scipy scikit-learn
```

<b>The script:</b>
<ul>
<li>Fetches stock data for AAPL (Apple) from 2020 to 2025.<br>
<li>Optimizes the momentum period.<br>
<li>Calculates indicators and trading signals.<br>
<li>Applies risk management.<br>
<li>Backtests the strategy.<br>
<li>Trains a machine learning model for better predictions.<br>
<li>Plots portfolio performance.<br>
</ul>


## Output
<ul>
<li>Optimal momentum period and cumulative return.
<li>Machine learning model accuracy.
<li>Portfolio performance visualization.
</ul>
<br>

## Customization
<ul>
<li>Modify ticker, start_date, and end_date in main().
<li>Adjust stop-loss and take-profit in apply_risk_management().
<li>Change technical indicators or ML model.
</ul>
