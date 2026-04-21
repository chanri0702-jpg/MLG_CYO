import os
import pathlib
import datetime

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html


#Constants / paths

ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent / "artifacts"

DROPDOWN_STYLE = {
    "backgroundColor": "#1a1a1a",
    "color": "#00FF41",
    "border": "1px solid #00FF41",
    "borderRadius": "5px",
}

#data fetching
#subject to change 
def _fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Attempt to fetch OHLCV data via yfinance.
    Returns an empty DataFrame on failure.
    """
    try:
        import yfinance as yf  # optional dependency
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # yfinance sometimes returns MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
        df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        return df[["date", "close"]].dropna()
    except Exception:
        return pd.DataFrame()


def _get_company_fundamentals(ticker: str) -> dict:
    """Return a dict of key fundamental metrics (requires yfinance)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "Market Cap": f"${info.get('marketCap', 'N/A'):,}" if isinstance(info.get("marketCap"), (int, float)) else "N/A",
            "P/E Ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
            "EPS": round(info.get("trailingEps", 0), 2) if info.get("trailingEps") else "N/A",
            "52-Wk High": f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52-Wk Low": f"${info.get('fiftyTwoWeekLow', 'N/A')}",
            "Dividend Yield": f"{round(info.get('dividendYield', 0)*100, 2)}%" if info.get("dividendYield") else "N/A",
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
        }
    except Exception:
        return {}


def _get_market_sentiment(ticker: str) -> dict:
    """
    Very lightweight sentiment proxy: 50-day vs 200-day moving average.
    Returns signal + supporting stats.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return {}
        close = hist["Close"]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]
        rsi = _compute_rsi(close)
        if ma50 > ma200 and current > ma50:
            signal = "Bullish"
        elif ma50 < ma200 and current < ma50:
            signal = "Bearish"
        else:
            signal = "Neutral"
        return {
            "Signal": signal,
            "RSI (14)": round(rsi, 1),
            "MA 50": f"${round(ma50, 2)}",
            "MA 200": f"${round(ma200, 2)}",
            "Current Price": f"${round(current, 2)}",
        }
    except Exception:
        return {}


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


def _simple_forecast(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Fit a simple linear trend on log-prices and extrapolate `periods` days.
    Returns a DataFrame with columns [date, close, type] where type ∈ {historical, predicted}.
    """
    if df.empty or len(df) < 10:
        return pd.DataFrame()
    hist = df.copy()
    hist["type"] = "historical"
    prices = hist["close"].values.astype(float)
    log_prices = np.log(prices + 1e-9)
    x = np.arange(len(log_prices))
    coeffs = np.polyfit(x, log_prices, 1)
    last_date = pd.to_datetime(hist["date"].iloc[-1])
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(periods)]
    future_x = np.arange(len(log_prices), len(log_prices) + periods)
    future_log = np.polyval(coeffs, future_x)
    future_prices = np.exp(future_log)
    pred = pd.DataFrame({"date": future_dates, "close": future_prices, "type": "predicted"})
    return pd.concat([hist, pred], ignore_index=True)


def _build_price_figure(ticker: str, df_full: pd.DataFrame) -> go.Figure:
    """Build a Plotly figure from the combined historical + predicted DataFrame."""
    fig = go.Figure()
    if df_full.empty:
        fig.update_layout(
            title="No data available",
            paper_bgcolor="#111",
            plot_bgcolor="#111",
            font_color="#e0e0e0",
        )
        return fig

    hist = df_full[df_full["type"] == "historical"]
    pred = df_full[df_full["type"] == "predicted"]

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["close"],
        mode="lines", name="Historical",
        line=dict(color="#00c853", width=2),
    ))
    if not pred.empty:
        # Connect last historical point to first predicted for continuity
        connector = pd.concat([hist.tail(1), pred.head(1)])
        fig.add_trace(go.Scatter(
            x=connector["date"], y=connector["close"],
            mode="lines", showlegend=False,
            line=dict(color="#ff6d00", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=pred["date"], y=pred["close"],
            mode="lines", name="Predicted (trend)",
            line=dict(color="#ff6d00", width=2, dash="dot"),
        ))

    fig.update_layout(
        title=f"{ticker.upper()} Stock Price",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#e0e0e0",
        legend=dict(bgcolor="#111", bordercolor="#333"),
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", tickprefix="$"),
    )
    return fig


def _build_sentiment_figure(sentiment: dict, ticker: str) -> go.Figure:
    """Gauge chart for RSI-based sentiment."""
    fig = go.Figure()
    if not sentiment:
        fig.update_layout(
            title="No sentiment data",
            paper_bgcolor="#111",
            plot_bgcolor="#111",
            font_color="#e0e0e0",
        )
        return fig

    rsi_val = sentiment.get("RSI (14)", 50)
    signal = sentiment.get("Signal", "Neutral")
    color = "#00c853" if signal == "Bullish" else "#ff5252" if signal == "Bearish" else "#ffb300"

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=rsi_val,
        title={"text": f"RSI (14) — {signal}", "font": {"color": "#e0e0e0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#e0e0e0"},
            "bar": {"color": color},
            "bgcolor": "#1a1a1a",
            "steps": [
                {"range": [0, 30], "color": "#1a3a1a"},
                {"range": [30, 70], "color": "#1a1a1a"},
                {"range": [70, 100], "color": "#3a1a1a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": rsi_val,
            },
        },
        number={"font": {"color": "#e0e0e0"}},
    ))
    fig.update_layout(
        paper_bgcolor="#111",
        font_color="#e0e0e0",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig



# App

app = dash.Dash(__name__)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Stock Price Predictor</title>
    {%css%}
    <style>
      body { margin: 0; padding: 0; background: #0a0a0a; font-family: "Courier New", monospace; }
      .page-root { background: #0a0a0a; color: #e0e0e0; min-height: 100vh; padding: 24px; box-sizing: border-box; }
      h1 { color: #00FF41; margin-bottom: 4px; font-size: 1.6rem; letter-spacing: 2px; }
      p.subtitle { color: #888; margin-top: 0; margin-bottom: 20px; font-size: 0.85rem; }
      .layout-row { display: flex; gap: 20px; align-items: flex-start; }
      .left-panel {
        width: 350px; min-width: 260px; flex-shrink: 0;
        background: #111; border: 1px solid #00FF41;
        border-radius: 8px; padding: 14px; box-sizing: border-box;
      }
      .left-panel h3 { color: ##ed11c8; margin-top: 0; font-size: 0.8rem; letter-spacing: 1px; }
      .right-panel { flex: 1; display: flex; flex-direction: column; gap: 16px; }
      .field-card { margin-bottom: 14px; }
      .field-card label { display: block; color: #aaa; font-size: 0.8rem; margin-bottom: 4px; letter-spacing: 1px; text-transform: uppercase; }
      .text-input {
        width: 100%; box-sizing: border-box;
        background: #0a0a0a; color: #00FF41;
        border: 1px solid #00FF41; border-radius: 4px;
        padding: 7px 10px; font-family: "Courier New", monospace; font-size: 0.9rem;
        outline: none;
      }
      .text-input:focus { border-color: #00FF41; box-shadow: 0 0 6px rgba(0,255,65,0.3); }
      .checkbox-group {  display: flex;  flex-direction: column;  gap: 8px;}
      .checkbox-label {  display: flex;  align-items: center;  gap: 8px;  cursor: pointer;  color: #ccc;  font-size: 0.75rem;}
      .checkbox-input { width: 16px; height: 16px; margin: 0; accent-color: #00FF41;}
      .checkbox-group label span {  font-size: 0.75rem;  color: #ccc;}
      .predict-btn {
        width: 100%; padding: 10px; margin-top: 10px;
        background: transparent; color: #00FF41;
        border: 1px solid #00FF41; border-radius: 4px;
        font-family: "Courier New", monospace; font-size: 0.9rem;
        cursor: pointer; letter-spacing: 1px; transition: all 0.2s;
      }
      .predict-btn:hover { background: #00FF41; color: #0a0a0a; }
      .info-card {
        background: #111; border: 1px solid #333; border-radius: 8px;
        padding: 14px; font-size: 0.82rem;
      }
      .info-card h4 { color: #00FF41; margin: 0 0 8px; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase; }
      .info-row { display: flex; justify-content: space-between; border-bottom: 1px solid #1e1e1e; padding: 4px 0; }
      .info-row:last-child { border-bottom: none; }
      .info-key { color: #888; }
      .info-val { color: #e0e0e0; }
      .action-badge {
        display: inline-block; padding: 4px 14px; border-radius: 4px;
        font-size: 0.85rem; font-weight: bold; letter-spacing: 1px;
        margin-bottom: 6px;
      }
      .action-BUY { background: #00c853; color: #0a0a0a; }
      .action-SELL { background: #ff5252; color: #fff; }
      .action-HOLD { background: #ffb300; color: #0a0a0a; }
      .metric-row { display: flex; gap: 10px; }
      .metric-mini { background: #111; border: 1px solid #333; border-radius: 6px; padding: 10px 14px; flex: 1; }
      .metric-mini .label { color: #666; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; }
      .metric-mini .value { color: #00FF41; font-size: 1rem; margin-top: 2px; }
      .dark-dropdown div { background-color: #0a0a0a !important; }
      /* Force each checklist item inline */
     .dash-options-list-option {  display: flex !important;  align-items: center !important;  gap: 8px;}
     .dash-options-list-option-wrapper {  display: flex;  align-items: center;}
     .dash-options-list-option-text {  display: flex;  align-items: center;}
     .dash-options-list-option input[type="checkbox"] {  width: 16px;  height: 16px;  margin: 0;  accent-color: #00FF41;}
     .dash-options-list-option-text span {  font-size: 0.75rem;  color: #ccc;}
    </style>
  </head>
  <body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
  </body>
</html>
'''

# Layout

app.layout = html.Div(
    [
        html.H1("STOCK PRICE PREDICTOR"),
        html.P("Enter a ticker, date range, and options — then click Predict.", className="subtitle"),

        html.Div(
            [
                # ── Left panel ──────────────────────────────────────────────
                html.Div(
                    [
                        html.H3("Enter Stock Ticker"),

                        html.Div([
                            html.Label("Ticker"),
                            dcc.Input(
                                id="input-ticker", type="text", value="AAPL",
                                debounce=True, className="text-input",
                                placeholder="e.g. AAPL, TSLA, MSFT"
                            ),
                        ], className="field-card"),

                        html.Div([
                            html.Label("Start Date"),
                            dcc.Input(
                                id="input-start", type="text", value="2023-01-01",
                                debounce=True, className="text-input",
                                placeholder="YYYY-MM-DD"
                            ),
                        ], className="field-card"),

                        html.Div([
                            html.Label("End Date"),
                            dcc.Input(
                                id="input-end", type="text",
                                value=str(datetime.date.today()),
                                debounce=True, className="text-input",
                                placeholder="YYYY-MM-DD"
                            ),
                        ], className="field-card"),

                        # ── Checkboxes ──────────────────────────────
                        html.Div([
                            dcc.Checklist(
                                options=[
                                    {"label": "Include Company Fundamentals", "value": "fundamentals"},
                                    {"label": "Include Market Sentiment", "value": "sentiment"},
                                    {"label": "View Current Fundamentals", "value": "curr_fundamentals"},
                                    {"label": "View Current Market Sentiment", "value": "curr_sentiment"},
                                ],
                                value=[],  # default checked values
                                id="checkbox-options",
                                className="checkbox-group",
                            )
                        ], className="field-card"),

                        html.Button(
                            "Predict Stock Price",
                            id="predict-btn", n_clicks=0,
                            className="predict-btn"
                        ),

                        html.Div(
                            id="error-msg",
                            style={"color": "#ff5252", "marginTop": "10px", "fontSize": "0.8rem"}
                        ),
                    ],
                    className="left-panel",
                ),


                # ── Right panel ─────────────────────────────────────────────
                html.Div(
                    [
                        # Stock price graph
                        dcc.Graph(
                            id="graph-stock",
                            figure=go.Figure().update_layout(
                                title="Graph: Stock Prices",
                                paper_bgcolor="#111", plot_bgcolor="#111",
                                font_color="#555",
                                margin=dict(l=8, r=8, t=40, b=8),
                            ),
                            style={"height": "300px", "border": "1px solid #333", "borderRadius": "8px"},
                        ),

                        # Recommended action + predicted fundamentals label
                        html.Div(id="recommended-action", style={"fontSize": "0.85rem", "color": "#888"}),

                        # Market sentiment gauge
                        dcc.Graph(
                            id="graph-sentiment",
                            figure=go.Figure().update_layout(
                                title="Market Sentiment",
                                paper_bgcolor="#111", plot_bgcolor="#111",
                                font_color="#555",
                                margin=dict(l=20, r=20, t=60, b=20),
                            ),
                            style={"height": "260px", "border": "1px solid #333", "borderRadius": "8px"},
                        ),

                        # Current company fundamentals
                        html.Div(id="current-fundamentals"),

                        # Current market sentiment text
                        html.Div(id="current-sentiment"),

                        # Model accuracy row
                        html.Div(id="model-accuracy"),
                    ],
                    className="right-panel",
                ),
            ],
            className="layout-row",
        ),
    ],
    className="page-root",
)

# Callback

@app.callback(
    Output("graph-stock", "figure"),
    Output("graph-sentiment", "figure"),
    Output("recommended-action", "children"),
    Output("current-fundamentals", "children"),
    Output("current-sentiment", "children"),
    Output("model-accuracy", "children"),
    Output("error-msg", "children"),
    Input("predict-btn", "n_clicks"),
    State("input-ticker", "value"),
    State("input-start", "value"),
    State("input-end", "value"),
    State("chk-fundamentals", "value"),
    State("chk-sentiment", "value"),
    State("chk-curr-fundamentals", "value"),
    State("chk-curr-sentiment", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, ticker, start, end,
                   inc_fund, inc_sent, view_fund, view_sent):
    empty_fig = go.Figure().update_layout(
        paper_bgcolor="#111", plot_bgcolor="#111",
        font_color="#555", margin=dict(l=8, r=8, t=40, b=8),
    )

    if not ticker or not ticker.strip():
        return empty_fig, empty_fig, None, None, None, None, "Please enter a ticker symbol."

    ticker = ticker.strip().upper()

    # --- Validate dates ---
    try:
        start_dt = datetime.datetime.strptime(start.strip(), "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end.strip(), "%Y-%m-%d").date()
        if start_dt >= end_dt:
            raise ValueError("Start must be before end.")
    except (ValueError, AttributeError) as exc:
        return empty_fig, empty_fig, None, None, None, None, f"Date error: {exc}"

    # --- Fetch price data ---
    df = _fetch_stock_data(ticker, str(start_dt), str(end_dt))
    if df.empty:
        return (
            empty_fig, empty_fig,
            html.Span(f"No price data found for {ticker}. Is yfinance installed?", style={"color": "#ff5252"}),
            None, None, None,
            f"Could not retrieve data for {ticker}.",
        )

    # --- Forecast ---
    df_full = _simple_forecast(df, periods=30)
    price_fig = _build_price_figure(ticker, df_full)

    # --- Sentiment ---
    sentiment = _get_market_sentiment(ticker) if inc_sent or view_sent else {}
    sentiment_fig = _build_sentiment_figure(sentiment, ticker) if sentiment else empty_fig

    # --- Recommended action ---
    signal = sentiment.get("Signal", "N/A") if sentiment else "N/A"
    action = "BUY" if signal == "Bullish" else "SELL" if signal == "Bearish" else "HOLD"
    if sentiment:
        rec_children = html.Div([
            html.Span("Recommended Action: ", style={"color": "#888", "fontSize": "0.82rem"}),
            html.Span(action, className=f"action-badge action-{action}"),
            html.Div(
                "Predicted Company Fundamentals: based on linear trend extrapolation (30-day horizon)",
                style={"color": "#555", "fontSize": "0.75rem", "marginTop": "4px"},
            ),
        ])
    else:
        rec_children = html.Div(
            "Enable 'Include Market Sentiment' to see a recommended action.",
            style={"color": "#555", "fontSize": "0.78rem"},
        )

    # --- Current fundamentals ---
    fund_children = None
    if view_fund or inc_fund:
        fundamentals = _get_company_fundamentals(ticker)
        if fundamentals:
            fund_children = html.Div([
                html.H4("Current Company Fundamentals:"),
                *[html.Div([
                    html.Span(k, className="info-key"),
                    html.Span(str(v), className="info-val"),
                ], className="info-row") for k, v in fundamentals.items()],
            ], className="info-card")
        else:
            fund_children = html.Div("Fundamentals unavailable (install yfinance).", className="info-card",
                                     style={"color": "#555"})

    # --- Current market sentiment text ---
    sent_children = None
    if view_sent or inc_sent:
        if sentiment:
            sent_children = html.Div([
                html.H4("Current Market Sentiment:"),
                *[html.Div([
                    html.Span(k, className="info-key"),
                    html.Span(str(v), className="info-val"),
                ], className="info-row") for k, v in sentiment.items()],
            ], className="info-card")
        else:
            sent_children = html.Div("Sentiment data unavailable.", className="info-card",
                                     style={"color": "#555"})

    # --- Model accuracy (simple in-sample metric on linear model) ---
    if not df_full.empty:
        hist_only = df_full[df_full["type"] == "historical"]
        actual = hist_only["close"].values.astype(float)
        x = np.arange(len(actual))
        log_actual = np.log(actual + 1e-9)
        coeffs = np.polyfit(x, log_actual, 1)
        fitted = np.exp(np.polyval(coeffs, x))
        ss_res = np.sum((actual - fitted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(actual - fitted)))

        accuracy_children = html.Div([
            html.Div([
                html.Div([html.Div("Model Accuracy (R²)", className="label"),
                          html.Div(f"{r2:.4f}", className="value")], className="metric-mini"),
                html.Div([html.Div("MAE ($)", className="label"),
                          html.Div(f"${mae:.2f}", className="value")], className="metric-mini"),
                html.Div([html.Div("Model Type", className="label"),
                          html.Div("Linear Trend", className="value")], className="metric-mini"),
                html.Div([html.Div("Forecast Horizon", className="label"),
                          html.Div("30 days", className="value")], className="metric-mini"),
            ], className="metric-row"),
        ])
    else:
        accuracy_children = None

    return price_fig, sentiment_fig, rec_children, fund_children, sent_children, accuracy_children, ""


# Entry point

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)