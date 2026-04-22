
import pandas as pd
import sys
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate  # type: ignore
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

def get_full_historical_fundamentals(ticker, start=None, end=None):
    t = yf.Ticker(ticker)
    
    income = t.quarterly_income_stmt.T
    balance = t.quarterly_balance_sheet.T
    cashflow = t.quarterly_cashflow.T
    
    fundamentals = pd.concat([income, balance, cashflow], axis=1)
    fundamentals.index = pd.to_datetime(fundamentals.index)
    fundamentals['Ticker'] = ticker

    fundamentals = fundamentals.sort_index()
    if start is not None:
        start_ts = pd.to_datetime(start)
        fundamentals = fundamentals[fundamentals.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        fundamentals = fundamentals[fundamentals.index <= end_ts]

    return fundamentals

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

    sentiment = get_market_sentiment(start, end)
    sentiment.index = pd.to_datetime(sentiment.index.date)  # date only, no time

    price_data = get_ticker_data(ticker, start, end)
    price_data.index = pd.to_datetime(price_data.index.date)  # date only, no time

    fundamentals_df = get_full_historical_fundamentals(ticker, start=start, end=end)
    fundamentals_df.index = pd.to_datetime(fundamentals_df.index.date)  # date only, no time

    dataset = pd.concat([price_data, sentiment], axis=1)
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()

    keep_cols = price_cols + sentiment_cols 

    keep_cols = [col for col in keep_cols if col in dataset.columns]
    
    dataset = dataset[keep_cols]

    avail_fund_cols = [col for col in fundamental_cols if col in fundamentals_df.columns]
    fund = fundamentals_df[avail_fund_cols].copy()
    if 'Gross Profit' in fund.columns and 'Total Revenue' in fund.columns:
        fund['Gross_Margin'] = fund['Gross Profit'] / fund['Total Revenue']
    if 'Operating Income' in fund.columns and 'Total Revenue' in fund.columns:
        fund['Operating_Margin'] = fund['Operating Income'] / fund['Total Revenue']
    if 'Net Income' in fund.columns and 'Total Revenue' in fund.columns:
        fund['Net_Margin'] = fund['Net Income'] / fund['Total Revenue']
    if 'Total Debt' in fund.columns and 'Common Stock Equity' in fund.columns:
        fund['Debt_To_Equity'] = fund['Total Debt'] / fund['Common Stock Equity']
    if 'Net Income' in fund.columns and 'Total Assets' in fund.columns:
        fund['ROA'] = fund['Net Income'] / fund['Total Assets']
    return dataset, sentiment, fund, price_data

def build_sequences(dataset, fundamentals_df, sequence_length=60, n_quarters=2):
    daily_cols = dataset.columns.to_list()
    fund_cols = fundamentals_df.select_dtypes(include=[np.number]).columns.tolist()

    n_fund_features = len(fund_cols)
    fund_data = fundamentals_df[fund_cols].sort_index() #sort ascending by date

    daily_scaler = MinMaxScaler()
    daily_scaled = daily_scaler.fit_transform(dataset[daily_cols])

    X_daily_list, X_fund_raw, y_list, sequence_dates = [], [], [], []

    for i in range(sequence_length, len(dataset)):
        date = dataset.index[i] #get date from stock data
        #get fund records earlier than or equal to this date, sorted ascending
        past_reports = fund_data[fund_data.index <= date] 
        if len(past_reports) == 0:
            continue #skip for loop if no fund data available before this date

        rows = past_reports.iloc[-n_quarters:].values #get prev 2 records
        #pad records if less than 2 is there
        if len(rows) < n_quarters:
            pad = np.tile(rows[0], (n_quarters - len(rows), 1))
            rows = np.vstack([pad, rows])

        X_daily_list.append(daily_scaled[i - sequence_length:i])
        X_fund_raw.append(rows)                         # (n_quarters, n_fund_features)
        y_list.append(daily_scaled[i, 3])           # predict scaled Close price
        sequence_dates.append(date)

    # Scale: flatten to 2D to fit scaler, then reshape back to 3D
    X_fund_arr = np.array(X_fund_raw, dtype=np.float32)  # Each sample has shape (n_quarters, n_fund_features)
    N = X_fund_arr.shape[0] #get number of samples
    X_fund_2d = X_fund_arr.reshape(N * n_quarters, n_fund_features) #turn 2d
    fund_scaler = MinMaxScaler()
    #scale and reshape back to 3d
    X_fund_scaled = fund_scaler.fit_transform(X_fund_2d).reshape(N, n_quarters, n_fund_features)

    return (
        np.array(X_daily_list, dtype=np.float32),#X_daily
        X_fund_scaled.astype(np.float32),#Xfund
        np.array(y_list, dtype=np.float32),#output
        pd.DatetimeIndex(sequence_dates),#sequence_dates
        daily_scaler,
        fund_scaler,
    )


def split_by_test_period(X_daily, X_fund, y, sequence_dates, test_start_date=None, test_end_date=None):
    if len(sequence_dates) == 0:
        raise ValueError("No sequence dates available for train/test split")

    sequence_dates = pd.to_datetime(sequence_dates)

    if test_start_date is not None:
        test_start = pd.to_datetime(test_start_date)
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

def configure_model(
    sequence_length,
    n_daily_features,
    n_quarters,
    n_fundamental_features
):
    daily_input = Input(shape=(sequence_length, n_daily_features))
    lstm_out = LSTM(128, return_sequences=True)(daily_input)
    lstm_out = Dropout(0.2)(lstm_out) #avoid overfitting by randomly dropping 20% of the connections between LSTM layers during training
    lstm_out = LSTM(64)(lstm_out)

    fund_input = Input(shape=(n_quarters, n_fundamental_features))  # 4 quarters x fund features
    fund_out = LSTM(32)(fund_input)

    combined = Concatenate()([lstm_out, fund_out])
    combined = Dense(32, activation='relu')(combined)
    output = Dense(1)(combined)  # predicted price

    model = Model(inputs=[daily_input, fund_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def train_and_predict_future_period(
    model,
    X_daily,
    X_fund,
    y,
    daily_scaler,
    fund_scaler,
    fundamentals_df,
    fund_cols,
    sequence_length,
    n_quarters,
    future_start_date,
    future_end_date,
    epochs=50,
    batch_size=32
):

    future_start = pd.to_datetime(future_start_date)
    future_end = pd.to_datetime(future_end_date)
    
    #train model
    model.fit([X_daily, X_fund], y, epochs=epochs, batch_size=batch_size, verbose=0)

    #get dates in date range
    future_dates = pd.bdate_range(start=future_start, end=future_end)
    if len(future_dates) == 0:
        raise ValueError("No business days found in the requested future period")

    avail_fund_cols = [c for c in fund_cols if c in fundamentals_df.columns]
    n_fund_features = len(avail_fund_cols)
    fund_data = fundamentals_df[avail_fund_cols].sort_index()
    n_daily_features = X_daily.shape[2]

    #get most recent x records
    last_daily_seq = X_daily[-1].copy()   
    last_fund_window = X_fund[-1].copy()  

    predictions = []

    #Iteratively predict each future day, then update the input sequences with the new prediction and any new fund data that becomes available
    for forecast_date in future_dates:
        pred_scaled = model.predict(
            [last_daily_seq.reshape(1, sequence_length, n_daily_features),
             last_fund_window.reshape(1, n_quarters, n_fund_features)],
            verbose=0
        )[0, 0] #sample, output

        row_for_inverse = np.zeros((1, n_daily_features)) #dummy row with all zeros
        row_for_inverse[0, 3] = pred_scaled #add prediction
        pred_close = daily_scaler.inverse_transform(row_for_inverse)[0, 3] #unscale to get actual price
        predictions.append((forecast_date, pred_close))

        next_row = last_daily_seq[-1].copy()
        next_row[3] = pred_scaled #get output column (Close) to be the predicted value, keep other features as last known values
        last_daily_seq = np.vstack([last_daily_seq[1:], next_row]) #add new row and remove oldest row to maintain sequence length

        past_reports = fund_data[fund_data.index <= forecast_date]
        if len(past_reports) > 0:
            rows = past_reports.iloc[-n_quarters:].values #copy fund recs
            if len(rows) < n_quarters:
                #pad if not enough
                pad = np.tile(rows[0], (n_quarters - len(rows), 1))
                rows = np.vstack([pad, rows])
                #transform records for model
            last_fund_window = fund_scaler.transform(
                rows.reshape(n_quarters, n_fund_features)
            )

    return pd.DataFrame(predictions, columns=['Date', 'Predicted_Close']).set_index('Date')














