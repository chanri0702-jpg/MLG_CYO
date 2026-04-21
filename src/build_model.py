
import pandas as pd
import sys
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


test_period_days = 180
test_start_date = None
test_end_date = None
future_pred_start_date = "2026-07-01"
future_pred_end_date = "2026-07-31"


def ticker_exists(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if info and info.get('regularMarketPrice') is not None:
            return True
        hist = t.history(period="5d")
        return len(hist) > 0
    except Exception:
        return False
    

def get_date_range(ticker, max_years=5):
    t = yf.Ticker(ticker)
    hist = t.history(period="max")
    if len(hist) == 0:
        raise ValueError(f"No data found for {ticker}")
    end_date = datetime.today()
    start_date_5yr = end_date - timedelta(days=max_years * 365)
    earliest_available = hist.index.min().to_pydatetime().replace(tzinfo=None)
    start_date = max(start_date_5yr, earliest_available)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_ticker_data(ticker, start, end):
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end)
    df.index = df.index.tz_localize(None)
    df['Ticker'] = ticker
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df['RSI'] = compute_rsi(df['Close'])
    def compute_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df['BB_Upper'] = df['MA_20'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['MA_20'] - 2 * df['Close'].rolling(20).std()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    return df

price_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Daily_Return', 'Log_Return',
    'MA_20', 'MA_50', 'MA_200',
    'Volatility_20', 'Volume_Change', 'Volume_MA_20',
    'RSI', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Lower', 'BB_Width'
]

def get_market_sentiment(start, end):
    sentiment = pd.DataFrame()
    vix = yf.download("^VIX", start=start, end=end)['Close'].squeeze()
    vix.index = vix.index.tz_localize(None)  # strip timezone
    sentiment['VIX'] = vix
    sentiment['VIX_MA20'] = vix.rolling(20).mean()
    sp500 = yf.download("^GSPC", start=start, end=end)['Close'].squeeze()
    sp500.index = sp500.index.tz_localize(None)  # strip timezone
    sentiment['SP500'] = sp500
    sentiment['SP500_Return'] = sp500.pct_change()
    sentiment['SP500_MA50'] = sp500.rolling(50).mean()
    treasury = yf.download("^TNX", start=start, end=end)['Close'].squeeze()
    treasury.index = treasury.index.tz_localize(None)
    sentiment['Treasury_10Y'] = treasury
    sentiment['Yield_Change'] = treasury.pct_change()
    gold = yf.download("GC=F", start=start, end=end)['Close'].squeeze()
    gold.index = gold.index.tz_localize(None)
    sentiment['Gold_Price'] = gold
    sentiment['Gold_Return'] = gold.pct_change()
    dxy = yf.download("DX-Y.NYB", start=start, end=end)['Close'].squeeze()
    dxy.index = dxy.index.tz_localize(None)
    sentiment['DXY'] = dxy
    sentiment['DXY_Return'] = dxy.pct_change()
    sp500_aligned = sentiment['SP500']
    ma50_aligned = sentiment['SP500_MA50']
    sentiment['Market_Regime'] = np.where(
        sp500_aligned > ma50_aligned, 1, 0
    )
    return sentiment

sentiment_cols = [
    'VIX', 'VIX_MA20',
    'SP500', 'SP500_Return', 'SP500_MA50',
    'Treasury_10Y', 'Yield_Change',
    'Gold_Price', 'Gold_Return',
    'DXY', 'DXY_Return',
    'Market_Regime'
]

def get_full_historical_fundamentals(ticker):
    t = yf.Ticker(ticker)
    
    income = t.quarterly_income_stmt.T
    balance = t.quarterly_balance_sheet.T
    cashflow = t.quarterly_cashflow.T
    
    fundamentals = pd.concat([income, balance, cashflow], axis=1)
    fundamentals.index = pd.to_datetime(fundamentals.index)
    fundamentals['Ticker'] = ticker
    
    return fundamentals.sort_index()

fundamental_cols = [
    'Total Revenue', 'Gross Profit', 'Operating Income',
    'Net Income', 'EBITDA', 'EBIT',
    'Diluted EPS', 'Basic EPS',
    'Total Assets', 'Total Debt', 'Net Debt',
    'Common Stock Equity', 'Working Capital',
    'Free Cash Flow', 'Operating Cash Flow',
    'Research And Development',
    'Capital Expenditure',
    'Stock Based Compensation',
    'Diluted Average Shares', 'Ordinary Shares Number'
]

def build_full_dataset(ticker, start, end):
    print("Fetching market sentiment...")
    sentiment = get_market_sentiment(start, end)
    sentiment.index = pd.to_datetime(sentiment.index.date)
    print("Fetching price data...")
    price_data = get_ticker_data(ticker, start, end)
    price_data.index = price_data.index.tz_localize(None)
    print("Fetching fundamentals...")
    fundamentals_df = get_full_historical_fundamentals(ticker)
    fundamentals_df.index = pd.to_datetime(fundamentals_df.index.date)  # date only, no time
    fundamentals_df = fundamentals_df.sort_index()

    daily_index = price_data.index
    fundamentals_daily = (
        fundamentals_df
        .reindex(fundamentals_df.index.union(daily_index))
        .sort_index()
        .ffill()
        .bfill()
        .reindex(daily_index)
    )
    dataset = pd.concat([price_data, sentiment, fundamentals_daily], axis=1)
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()

    keep_cols = price_cols + sentiment_cols + fundamental_cols
    
    keep_cols = [col for col in keep_cols if col in dataset.columns]
    
    dataset = dataset[keep_cols]
    
    dataset['Gross_Margin'] = dataset['Gross Profit'] / dataset['Total Revenue']
    dataset['Operating_Margin'] = dataset['Operating Income'] / dataset['Total Revenue']
    dataset['Net_Margin'] = dataset['Net Income'] / dataset['Total Revenue']
    dataset['Debt_To_Equity'] = dataset['Total Debt'] / dataset['Common Stock Equity']
    dataset['ROA'] = dataset['Net Income'] / dataset['Total Assets']
    return dataset, sentiment, fundamentals_daily, price_data

#---------------------------TRAINING MODEL-------------------------------------------
dataset = build_full_dataset("AAPL", "2018-01-01", "2024-01-01")[0]
daily_cols = [col for col in price_cols + sentiment_cols if col in dataset.columns]
fund_cols = [col for col in fundamental_cols if col in dataset.columns] #is quarterly

sequence_length = 60                          # look back 60 days
n_daily_features = len(daily_cols)            
n_fundamental_features = len(fund_cols)  

daily_input = Input(shape=(sequence_length, n_daily_features))
lstm_out = LSTM(128, return_sequences=True)(daily_input)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = LSTM(64)(lstm_out)

fund_input = Input(shape=(n_fundamental_features,))
fund_out = Dense(32, activation='relu')(fund_input)
fund_out = Dense(16, activation='relu')(fund_out)

combined = Concatenate()([lstm_out, fund_out])
combined = Dense(32, activation='relu')(combined)
output = Dense(1)(combined)  # predicted price

model = Model(inputs=[daily_input, fund_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


daily_cols = [col for col in price_cols + sentiment_cols if col in dataset.columns]
fund_cols = [col for col in fundamental_cols if col in dataset.columns]
daily_cols = ['Close'] + [col for col in daily_cols if col != 'Close']
print("Dataset shape before dropna:", dataset.shape)
print("NaN counts per column:")
print(dataset[daily_cols + fund_cols].isnull().sum())
dataset = dataset.dropna(subset=daily_cols + fund_cols)
missing_daily = [col for col in daily_cols if col not in dataset.columns]
missing_fund = [col for col in fund_cols if col not in dataset.columns]
print("Missing daily cols:", missing_daily)
print("Missing fund cols:", missing_fund)

daily_scaler = MinMaxScaler()
daily_scaled = daily_scaler.fit_transform(dataset[daily_cols])
fundamentals_scaler = MinMaxScaler()
fundamentals_scaled = fundamentals_scaler.fit_transform(dataset[fund_cols])

def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # predicting Close price
    return np.array(X), np.array(y)


def split_by_test_period(X_daily, X_fund, y, sequence_dates, test_period_days=None, test_start_date=None, test_end_date=None):
    if len(sequence_dates) == 0:
        raise ValueError("No sequence dates available for train/test split")

    sequence_dates = pd.to_datetime(sequence_dates)

    if test_start_date is not None:
        test_start = pd.to_datetime(test_start_date)
    elif test_period_days is not None:
        test_start = sequence_dates.max() - pd.Timedelta(days=int(test_period_days) - 1)
    else:
        split_idx = int(len(sequence_dates) * 0.8)
        test_start = sequence_dates[split_idx]

    test_end = pd.to_datetime(test_end_date) if test_end_date is not None else sequence_dates.max()

    test_mask = (sequence_dates >= test_start) & (sequence_dates <= test_end)
    train_mask = sequence_dates < test_start

    if not test_mask.any():
        raise ValueError("No samples fall into the selected test period")
    if not train_mask.any():
        raise ValueError("No training samples remain before selected test period")

    return (
        X_daily[train_mask], X_daily[test_mask],
        X_fund[train_mask], X_fund[test_mask],
        y[train_mask], y[test_mask],
        sequence_dates[test_mask]
    )


def train_and_predict_future_period(
    model,
    daily_scaled,
    fundamentals_scaled,
    daily_scaler,
    sequence_length,
    future_start_date,
    future_end_date,
    epochs=50,
    batch_size=32
):
    future_start = pd.to_datetime(future_start_date)
    future_end = pd.to_datetime(future_end_date)
    if future_end < future_start:
        raise ValueError("future_end_date must be on or after future_start_date")

    X_daily_full, y_full = create_sequences(daily_scaled, sequence_length=sequence_length)
    X_fund_full = fundamentals_scaled[sequence_length:]
    if len(X_daily_full) == 0:
        raise ValueError("Not enough historical data to train sequences")

    model.fit(
        [X_daily_full, X_fund_full],
        y_full,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Use business days for stock forecasts.
    future_dates = pd.bdate_range(start=future_start, end=future_end)
    if len(future_dates) == 0:
        raise ValueError("No business days found in the requested future period")

    last_sequence = daily_scaled[-sequence_length:].copy()
    last_fund_vector = fundamentals_scaled[-1].reshape(1, -1)
    n_daily_features = daily_scaled.shape[1]
    predictions = []

    for forecast_date in future_dates:
        pred_scaled = model.predict(
            [last_sequence.reshape(1, sequence_length, n_daily_features), last_fund_vector],
            verbose=0
        )[0, 0]

        row_for_inverse = np.zeros((1, n_daily_features))
        row_for_inverse[0, 0] = pred_scaled  # daily_cols[0] is Close
        pred_close = daily_scaler.inverse_transform(row_for_inverse)[0, 0]
        predictions.append((forecast_date, pred_close))

        next_row = last_sequence[-1].copy()
        next_row[0] = pred_scaled
        last_sequence = np.vstack([last_sequence[1:], next_row])

    return pd.DataFrame(predictions, columns=['Date', 'Predicted_Close']).set_index('Date')

X_daily, y = create_sequences(daily_scaled, sequence_length=60)
X_fund = fundamentals_scaled[60:]  # align with sequences
sequence_dates = dataset.index[60:]

X_daily_train, X_daily_test, X_fund_train, X_fund_test, y_train, y_test, test_dates = split_by_test_period(
    X_daily,
    X_fund,
    y,
    sequence_dates,
    test_period_days=test_period_days,
    test_start_date=test_start_date,
    test_end_date=test_end_date
)

print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
print(f"Test period: {test_dates.min().date()} -> {test_dates.max().date()}")

model.fit(
    [X_daily_train, X_fund_train], y_train,
    epochs=50,
    batch_size=32,
    validation_data=([X_daily_test, X_fund_test], y_test)  # use test as validation
)

y_pred = model.predict([X_daily_test, X_fund_test])

y_pred_actual = daily_scaler.inverse_transform(
    np.concatenate([y_pred, np.zeros((len(y_pred), daily_scaled.shape[1]-1))], axis=1)
)[:, 0]

y_test_actual = daily_scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), daily_scaled.shape[1]-1))], axis=1)
)[:, 0]

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

if future_pred_start_date and future_pred_end_date:
    future_forecast = train_and_predict_future_period(
        model=model,
        daily_scaled=daily_scaled,
        fundamentals_scaled=fundamentals_scaled,
        daily_scaler=daily_scaler,
        sequence_length=sequence_length,
        future_start_date=future_pred_start_date,
        future_end_date=future_pred_end_date,
        epochs=50,
        batch_size=32
    )
    print("Future forecast:")
    print(future_forecast)