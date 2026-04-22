
import pandas as pd
import sys
import time
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
# from tensorflow.keras.models import Model  # type: ignore
# from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate  # type: ignore
# from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Download all tickers in parallel to cut 3 sequential network calls down to 1 round-trip
from concurrent.futures import ThreadPoolExecutor

test_period_days = 180
test_start_date = None
test_end_date = None
future_pred_start_date = "2026-07-01"
future_pred_end_date = "2026-07-31"


def _retry_yf_request(fetch_fn, attempts=3, delay_seconds=1.5, retry_on_empty=True):
    last_error = None
    for attempt in range(attempts):
        try:
            result = fetch_fn()
            if retry_on_empty and result is not None and hasattr(result, "__len__") and len(result) == 0:
                raise ValueError("Yahoo Finance returned empty data")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds * (attempt + 1))
    raise last_error


def ticker_exists(ticker):
    t = yf.Ticker(ticker)
    # Yahoo metadata calls are less reliable in hosted environments.
    # Treat them as an optional fast path and fall back to recent price history.
    try:
        fast_info = getattr(t, "fast_info", None)
        if fast_info:
            last_price = fast_info.get("lastPrice")
            if last_price is not None:
                return True
    except Exception:
        pass

    try:
        hist = _retry_yf_request(lambda: t.history(period="5d", auto_adjust=False))
        return hist is not None and len(hist) > 0
    except Exception:
        return False
    

def get_date_range(ticker, max_years=5):
    t = yf.Ticker(ticker)
    hist = _retry_yf_request(lambda: t.history(period="max"))
    if len(hist) == 0:
        raise ValueError(f"No data found for {ticker}")
    end_date = datetime.today()
    start_date_5yr = end_date - timedelta(days=max_years * 365)
    earliest_available = hist.index.min().to_pydatetime().replace(tzinfo=None)
    start_date = max(start_date_5yr, earliest_available)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_ticker_data(ticker, start, end):
    t = yf.Ticker(ticker)
    df = _retry_yf_request(lambda: t.history(start=start, end=end))
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
    

    tickers = {"VIX": "^VIX", "SP500": "^GSPC", "TNX": "^TNX"}

    def _fetch(symbol):
        s = _retry_yf_request(
            lambda: yf.download(symbol, start=start, end=end, progress=False)['Close'].squeeze()
        )
        s.index = s.index.tz_localize(None)
        return s

    with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
        futures = {name: pool.submit(_fetch, sym) for name, sym in tickers.items()}
        results = {name: f.result() for name, f in futures.items()}

    vix      = results["VIX"]
    sp500    = results["SP500"]
    treasury = results["TNX"]

    sentiment = pd.DataFrame()
    sentiment['VIX'] = vix
    sentiment['VIX_MA20'] = vix.rolling(20).mean()
    sentiment['SP500'] = sp500
    sentiment['SP500_Return'] = sp500.pct_change()
    sentiment['SP500_MA50'] = sp500.rolling(50).mean()
    sentiment['Treasury_10Y'] = treasury
    sentiment['Yield_Change'] = treasury.pct_change()
    sentiment['Market_Regime'] = np.where(sentiment['SP500'] > sentiment['SP500_MA50'], 1, 0)
    return sentiment

sentiment_cols = [
    'VIX', 'VIX_MA20',
    'SP500', 'SP500_Return', 'SP500_MA50',
    'Treasury_10Y', 'Yield_Change',
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
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
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

    # Newly-listed or pre-revenue stocks may have no numeric fundamental
    # columns, or all-NaN values that break MinMaxScaler.  In either case we fall back
    # to a single dummy zero feature so the sequence shapes stay consistent downstream.
    if n_fund_features == 0:
        n_fund_features = 1
        fund_cols = ['_no_fundamentals']
        fund_data = pd.DataFrame(
            {'_no_fundamentals': [0.0]},
            index=[dataset.index[0]]
        )
    else:
        # Replace inf and NaN with 0 so MinMaxScaler doesn't fail.
        fund_data = (
            fundamentals_df[fund_cols]
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    daily_scaler = MinMaxScaler()
    daily_scaled = daily_scaler.fit_transform(dataset[daily_cols])

    X_daily_list, X_fund_raw, y_list, sequence_dates = [], [], [], []

    for i in range(sequence_length, len(dataset)):
        date = dataset.index[i] #get date from stock data
        #get fund records earlier than or equal to this date, sorted ascending
        past_reports = fund_data[fund_data.index <= date]
        if len(past_reports) == 0:
            # No fundamental report available yet — use zeros rather than skipping
            rows = np.zeros((n_quarters, n_fund_features), dtype=np.float32)
        else:
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


def flatten_sequence_features(X_daily, X_fund):
    #Flatten 3D sequence inputs into a 2D tabular matrix for XGBoost.
    if len(X_daily) == 0 or len(X_fund) == 0:
        return np.empty((0, 0), dtype=np.float32)
    X_daily_flat = X_daily.reshape(X_daily.shape[0], -1)
    X_fund_flat = X_fund.reshape(X_fund.shape[0], -1)
    return np.hstack([X_daily_flat, X_fund_flat]).astype(np.float32)


def configure_model(
    
):
    # OLD TENSORFLOW LSTM MODEL (kept for reference):
    # daily_input = Input(shape=(sequence_length, n_daily_features))
    # lstm_out = LSTM(128, return_sequences=True)(daily_input)
    # lstm_out = Dropout(0.2)(lstm_out)
    # lstm_out = LSTM(64)(lstm_out)
    #
    # fund_input = Input(shape=(n_quarters, n_fundamental_features))
    # fund_out = LSTM(32)(fund_input)
    #
    # combined = Concatenate()([lstm_out, fund_out])
    # combined = Dense(32, activation='relu')(combined)
    # output = Dense(1)(combined)
    #
    # model = Model(inputs=[daily_input, fund_input], outputs=output)
    # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # return model

    # XGBoost regressor over flattened sequence features
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
    )


def train_test_validate_model(model, X_daily, X_fund, y, test_size=0.2, random_state=42):
    """Train/test validation for the XGBoost model on flattened features."""
    X = flatten_sequence_features(X_daily, X_fund)
    if len(X) < 20:
        raise ValueError("Not enough samples to run train/test validation")
    #below keeps chronological split instead of random split to avoid data leakage in time series context
    split_idx = int(len(X) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(X) - 1))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    y_var = float(np.var(y_test))
    r2 = float(1 - mse / y_var) if y_var > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

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
    epochs=150,
    batch_size=32
):

    future_start = pd.to_datetime(future_start_date)
    future_end = pd.to_datetime(future_end_date)
    
    # OLD TENSORFLOW TRAINING BLOCK (kept for reference):
    # if epochs > 0:
    #     early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)
    #     model.fit([X_daily, X_fund], y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    #get dates in date range
    future_dates = pd.bdate_range(start=future_start, end=future_end)
    if len(future_dates) == 0:
        raise ValueError("No business days found in the requested future period")

    avail_fund_cols = [c for c in fund_cols if c in fundamentals_df.columns]
    n_fund_features = len(avail_fund_cols)
    if n_fund_features > 0:
        fund_data = (
            fundamentals_df[avail_fund_cols]
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    else:
        fund_data = pd.DataFrame()
    n_daily_features = X_daily.shape[2]

    #get most recent x records
    last_daily_seq = X_daily[-1].copy()   
    last_fund_window = X_fund[-1].copy()  

    # Seed a raw (unscaled) close buffer from the last historical window.
    # We keep raw prices so momentum/volatility features can be recomputed
    # in original dollar space, then re-scaled per feature.
    # price_cols indices: Close=3, Daily_Return=5, Log_Return=6,
    #   MA_20=7, MA_50=8, MA_200=9, Volatility_20=10,
    #   RSI=13, MACD=14, MACD_Signal=15, BB_Upper=16, BB_Lower=17, BB_Width=18

    #unscale values in a column back to original price space using the MinMaxScaler parameters
    #so user can understand
    def scale_col(val, col):
        mn = daily_scaler.data_min_[col]
        mx = daily_scaler.data_max_[col]
        if mx == mn:
            return 0.0
        return float(np.clip((val - mn) / (mx - mn), 0.0, 1.0))

    # Inverse-transform the stored scaled close values to get raw prices
    raw_close_buffer = []
    for sc in last_daily_seq[:, 3]:
        dummy = np.zeros((1, n_daily_features))
        dummy[0, 3] = sc
        raw_close_buffer.append(float(daily_scaler.inverse_transform(dummy)[0, 3])) #store scaled close price of recent rec

    # scaled close buffer for the MA recomputation 
    close_buffer = list(last_daily_seq[:, 3])# get all rows from col 3

    predictions = []

    #Iteratively predict each future day, then update the input sequences with the new prediction and any new fund data that becomes available
    for forecast_date in future_dates:
        feature_row = np.hstack([
            #turn rows into 1d array by flattening the sequence, then concatenate daily and fund features
            last_daily_seq.reshape(1, -1),
            last_fund_window.reshape(1, -1)
        ]).astype(np.float32)
        pred_scaled = float(model.predict(feature_row)[0])

        row_for_inverse = np.zeros((1, n_daily_features)) #dummy row with all zeros
        row_for_inverse[0, 3] = pred_scaled #add prediction
        pred_close = daily_scaler.inverse_transform(row_for_inverse)[0, 3] #unscale to get actual price
        #build predictions
        predictions.append((forecast_date, pred_close))

        #add prediction to buffers for next iteration's feature recomputation
        raw_close_buffer.append(float(pred_close))
        close_buffer.append(float(pred_scaled))
        #store close prices in raw price space for accurate feature recomputation
        # # also keep scaled close values in buffer for moving average calculations that feed into the model features

        closes = np.array(raw_close_buffer, dtype=np.float64)

        next_row = last_daily_seq[-1].copy() #use the latest row we have to build prev row for predictions
        next_row[3] = pred_scaled  # update Close

        # MA_20, MA_50, MA_200 
        if n_daily_features > 7:
            next_row[7] = float(np.mean(close_buffer[-20:]))
        if n_daily_features > 8:
            next_row[8] = float(np.mean(close_buffer[-50:]))
        if n_daily_features > 9:
            next_row[9] = float(np.mean(close_buffer[-200:]))

        #Daily_Return and Log_Return
        if n_daily_features > 5 and len(closes) >= 2 and closes[-2] > 0:
            dr = (closes[-1] - closes[-2]) / closes[-2]
            lr = float(np.log(closes[-1] / closes[-2]))
            next_row[5] = scale_col(dr, 5)
            if n_daily_features > 6:
                next_row[6] = scale_col(lr, 6)

        # Volatility_20 
        if n_daily_features > 10 and len(closes) >= 21:
            rets = np.diff(closes[-21:]) / closes[-21:-1]
            next_row[10] = scale_col(float(np.std(rets)), 10)

        # RSI
        if n_daily_features > 13 and len(closes) >= 15:
            deltas = np.diff(closes[-15:]) #get prev 15 closes
            gains = float(np.mean(np.where(deltas > 0, deltas, 0.0)))
            losses = float(np.mean(np.where(deltas < 0, -deltas, 0.0)))
            rsi_val = 100.0 - (100.0 / (1.0 + gains / losses)) if losses > 0 else 100.0
            next_row[13] = scale_col(rsi_val, 13)

        # MACD and Signal line 
        if n_daily_features > 15 and len(closes) >= 26:
            cs = pd.Series(closes)
            ema12 = cs.ewm(span=12, adjust=False).mean()
            ema26 = cs.ewm(span=26, adjust=False).mean()
            macd_series = ema12 - ema26
            signal_series = macd_series.ewm(span=9, adjust=False).mean()
            next_row[14] = scale_col(float(macd_series.iloc[-1]), 14)
            next_row[15] = scale_col(float(signal_series.iloc[-1]), 15)

        # Bollinger Bands — 20-day MA ± 2σ
        if n_daily_features > 18 and len(closes) >= 20:
            ma20_raw = float(np.mean(closes[-20:]))
            std20_raw = float(np.std(closes[-20:]))
            bb_upper = ma20_raw + 2 * std20_raw
            bb_lower = ma20_raw - 2 * std20_raw
            next_row[16] = scale_col(bb_upper, 16)
            next_row[17] = scale_col(bb_lower, 17)
            next_row[18] = scale_col(bb_upper - bb_lower, 18)

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














