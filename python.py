import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import talib
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fetch historical data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Calculate momentum and technical indicators
def calculate_indicators(data, momentum_period=10):
    data['Momentum'] = data['Close'] - data['Close'].shift(momentum_period)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'])
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data.dropna(inplace=True)
    return data

# Generate trading signals
def generate_signals(data):
    data['Signal'] = 0
    data.loc[data['Momentum'] > 0, 'Signal'] = 1
    data.loc[data['Momentum'] < 0, 'Signal'] = -1
    return data

# Apply risk management (stop-loss, take-profit, position sizing)
def apply_risk_management(data, stop_loss=0.02, take_profit=0.04):
    data['Stop_Loss'] = data['Close'] * (1 - stop_loss)
    data['Take_Profit'] = data['Close'] * (1 + take_profit)
    return data

# Backtest strategy
def backtest(data, initial_capital=10000):
    capital = initial_capital
    position = 0
    returns = []
    for i in range(1, len(data)):
        if data['Signal'].iloc[i-1] == 1:
            position = capital / data['Close'].iloc[i-1]
            capital = 0
        elif data['Signal'].iloc[i-1] == -1 and position > 0:
            capital = position * data['Close'].iloc[i-1]
            position = 0
        returns.append(capital + (position * data['Close'].iloc[i]))
    data['Portfolio Value'] = returns
    return data

# Optimize momentum period
def optimize_momentum(data, start_period=5, end_period=50):
    best_period = None
    best_return = -np.inf
    for period in range(start_period, end_period + 1):
        temp_data = calculate_indicators(data.copy(), period)
        temp_data = backtest(temp_data)
        total_return = temp_data['Portfolio Value'].iloc[-1]
        if total_return > best_return:
            best_return = total_return
            best_period = period
    return best_period, best_return

# Train ML model for signal enhancement
def train_ml_model(data):
    features = ['Momentum', 'RSI', 'MACD', 'ATR']
    X = data[features]
    y = np.where(data['Signal'].shift(-1) > 0, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'Model Accuracy: {accuracy_score(y_test, predictions):.2f}')
    return model

# Plot results
def plot_results(data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Portfolio Value'], label='Portfolio Value', color='green')
    plt.title(f"{ticker} Advanced Momentum Trading Strategy")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

# Main function
def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-01-01"
    
    print(f"Fetching data for {ticker}...")
    data = fetch_data(ticker, start_date, end_date)
    
    print("Optimizing momentum period...")
    best_period, best_return = optimize_momentum(data, 5, 50)
    print(f"Optimal Momentum Period: {best_period} with Cumulative Return: {best_return:.2f}")
    
    print(f"Calculating indicators with optimal period ({best_period})...")
    data = calculate_indicators(data, best_period)
    data = generate_signals(data)
    data = apply_risk_management(data)
    data = backtest(data)
    
    print("Training machine learning model...")
    model = train_ml_model(data)
    
    print("Plotting results...")
    plot_results(data, ticker)

if __name__ == "__main__":
    main()
