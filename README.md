# Stock Price Predictor

An interactive web dashboard that forecasts stock closing prices using machine learning, built with Python, Dash, and XGBoost.

## What It Does

Users enter any valid stock ticker (e.g. `AAPL`, `TSLA`, `OKLO`), a start date, and an end date. The application then:

1. **Fetches up to 5 years of historical price data** from Yahoo Finance, including OHLCV values and derived technical indicators (RSI, MACD, Bollinger Bands, moving averages, volatility).
2. **Pulls macro market context** — the CBOE VIX fear index, S&P 500 level and returns, and the 10-year US Treasury yield — to give the model awareness of the broader market regime.
3. **Retrieves quarterly fundamental data** from the company's income statement, balance sheet, and cash flow statement (revenue, earnings, debt, margins, etc.), where available.
4. **Trains an XGBoost regression model** on the combined dataset using 60-day rolling sequences and produces a price forecast for the requested date range.
5. **Displays results** on a Prediction tab (price forecast chart, RSI gauge, recommended buy/sell/hold action, model accuracy metrics) and a Data Insights tab (top feature importances, feature trend charts).

## Key Features

- Works with any stock listed on Yahoo Finance, including recently-listed and pre-revenue companies
- Recommended action (BUY / SELL / HOLD) derived from RSI and moving average crossover signals
- Model accuracy reported on a held-out 20% chronological test split (RMSE, MAE, R²)
- Feature importance chart showing which indicators most influenced the forecast
- Optional overlays: current fundamentals, current market sentiment

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Plotly Dash |
| Machine Learning | XGBoost (`XGBRegressor`) |
| Data | yfinance, pandas, NumPy |
| Feature Scaling | scikit-learn `MinMaxScaler` |
| Server | Flask (via Dash) / Gunicorn for deployment |

## Model

The model uses **XGBoost** over flattened 60-day price/technical sequences combined with the 2 most recent quarters of fundamental data. Each prediction timestep is built iteratively, updating technical indicators (RSI, MACD, Bollinger Bands, moving averages) from the previous predicted close before forecasting the next day.

An earlier dual-input LSTM architecture (TensorFlow/Keras) was attempted but replaced due to deployment size constraints, per-request training time (~60–120 s vs ~2–4 s for XGBoost), and fragility on small or sparse datasets.

## Running Locally

```bash
pip install -r requirements.txt
python components/dashboard.py
```

Then open `http://127.0.0.1:8050` in your browser.
