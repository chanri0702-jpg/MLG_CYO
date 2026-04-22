import os
import pathlib
import datetime

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from data import build_model as bm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input as KInput, Dense, LSTM, Dropout, Concatenate  # type: ignore

#Constants / paths

ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent / "artifacts"

DROPDOWN_STYLE = {
    "backgroundColor": "#1a1a1a",
    "color": "#00FF41",
    "border": "1px solid #00FF41",
    "borderRadius": "5px",
}




#-----------------------------------------------------------ADD CHARTS------------------------------------------------------------------


def _build_price_figure(ticker: str, df_full: pd.DataFrame) -> go.Figure:
    """Build a Plotly figure from the combined historical + predicted DataFrame."""
    fig = go.Figure()
    
    return fig


def _build_sentiment_figure(sentiment: dict, ticker: str) -> go.Figure:
    """Gauge chart for RSI-based sentiment."""
    fig = go.Figure()
    
    return fig


#---------------------------------------------------------------------------------------------------------------------------------------
# App

app = dash.Dash(
    __name__,
    assets_folder=str(pathlib.Path(__file__).parent.parent / "assets"),
)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Stock Price Predictor</title>
    {%css%}
  </head>
  <body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
  </body>
</html>
'''
#------------------------------------------------------------------------------------------------------------------------------------------
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
                                id="input-start", type="text", value=str(datetime.date.today()),
                                debounce=True, className="text-input",
                                placeholder="YYYY-MM-DD"
                            ),
                        ], className="field-card"),

                        html.Div([
                            html.Label("End Date"),
                            dcc.Input(
                                id="input-end", type="text", value=str(datetime.date.today() + datetime.timedelta(days=7)),
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
                                #Add selecor based titles
                                title="Graph: Stock Prices",
                                paper_bgcolor="#111", plot_bgcolor="#111",
                                font_color="#555",
                                margin=dict(l=8, r=8, t=40, b=8),
                            ),
                            style={"height": "300px", "border": "1px solid #333", "borderRadius": "8px"},
                        ),

                        # Recommended action + predicted fundamentals label
                        html.Div(
                             id="recommended-action", 
                                      style={"color": "#888", "fontSize": "0.82rem"}),
                        

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
                        html.Div(
                        id="current-fundamentals",
                                      style={"color": "#888", "fontSize": "0.82rem"},
                        ),

                        # Current market sentiment text
                        html.Div(
                            id="current-sentiment",
                                      style={"color": "#888", "fontSize": "0.82rem"},
                        ),

                        # Model accuracy row
                        html.Div(
                          id="model-accuracy",
                                      style={"color": "#888", "fontSize": "0.82rem"},
                        ),
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
    State("checkbox-options", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, ticker, start_input, end_input, checklist_values):
    checklist_values = checklist_values or []
    inc_fund = "fundamentals" in checklist_values
    inc_sent = "sentiment" in checklist_values
    view_fund = "curr_fundamentals" in checklist_values
    view_sent = "curr_sentiment" in checklist_values

    empty_fig = go.Figure().update_layout(
        paper_bgcolor="#111", plot_bgcolor="#111",
        font_color="#555", margin=dict(l=8, r=8, t=40, b=8),
    )

    if not ticker or not ticker.strip():
        return empty_fig, empty_fig, None, None, None, None, "Please enter a ticker symbol."

    ticker = ticker.strip().upper()

    if not bm.ticker_exists(ticker):
        return empty_fig, empty_fig, None, None, None, None, f"Ticker '{ticker}' does not exist."

    today = datetime.date.today()

    # --- Parse dates ---
    try:
        start_dt = datetime.datetime.strptime(start_input.strip(), "%Y-%m-%d").date() if start_input else today
        end_dt = datetime.datetime.strptime(end_input.strip(), "%Y-%m-%d").date() if end_input else today + datetime.timedelta(days=30)
        if start_dt >= end_dt:
            raise ValueError("Start must be before end.")
    except (ValueError, AttributeError) as exc:
        return empty_fig, empty_fig, None, None, None, None, f"Date error: {exc}"

    # --- Fetch and build dataset using full historical window ---
    try:
        hist_start, hist_end = bm.get_date_range(ticker)
        dataset, sentiment_df, fundamentals_df, price_data = bm.build_full_dataset(ticker, hist_start, hist_end)
    except Exception as exc:
        return empty_fig, empty_fig, None, None, None, None, f"Data fetch error: {exc}"

    if dataset.empty:
        return empty_fig, empty_fig, None, None, None, None, f"No data available for '{ticker}'."

    price_data.index = pd.to_datetime(price_data.index)
    latest_actual_date = price_data.index.max().date()
    earliest_actual_date = price_data.index.min().date()
    requested_span = end_dt - start_dt

    # Use a same-length backtest window inside retrieved data, anchored to the most recent actual values.
    backtest_start = start_dt - requested_span
    backtest_end = end_dt - requested_span
    backtest_shift = latest_actual_date - backtest_end
    backtest_start = backtest_start + backtest_shift
    backtest_end = backtest_end + backtest_shift

    if backtest_start < earliest_actual_date:
        backtest_start = earliest_actual_date

    # --- Select historical price window to display ---
    if start_dt <= today:
        # Show only the user's requested historical window
        hist_mask = (price_data.index >= pd.to_datetime(start_dt)) & (price_data.index <= pd.to_datetime(today))
    else:
        # Start is in the future — show last 60 days as context
        context_start = today - datetime.timedelta(days=60)
        hist_mask = price_data.index >= pd.to_datetime(context_start)
    price_display = price_data[hist_mask] if hist_mask.any() else price_data

    hist_chart = price_display[["Close"]].rename(columns={"Close": "close"}).copy()
    hist_chart.index.name = "date"
    hist_chart = hist_chart.reset_index()
    hist_chart["type"] = "historical"

    # --- ML pipeline: recent-window backtest for accuracy, then full-data forecast for the chart ---
    backtest_forecast = None
    graph_forecast = None
    accuracy_children = None

    try:
        train_cutoff = pd.to_datetime(backtest_start) - pd.Timedelta(days=1)
        train_dataset = dataset[dataset.index <= train_cutoff]
        if len(train_dataset) < 90:
            raise ValueError("Not enough pre-period history to train the model for this test window")

        X_daily, X_fund, y, sequence_dates, daily_scaler, fund_scaler = bm.build_sequences(
            train_dataset, fundamentals_df, sequence_length=60, n_quarters=4
        )

        n_daily_features = X_daily.shape[2] #get number of features for daily data
        n_fund_features = X_fund.shape[2] #get number of features for fundamental data

        model = bm.configure_model(60, n_daily_features, 4, n_fund_features)

        fund_cols = fundamentals_df.select_dtypes(include=[np.number]).columns.tolist()
        backtest_forecast = bm.train_and_predict_future_period(
            model=model,
            X_daily=X_daily,
            X_fund=X_fund,
            y=y,
            daily_scaler=daily_scaler,
            fund_scaler=fund_scaler,
            fundamentals_df=fundamentals_df,
            fund_cols=fund_cols,
            sequence_length=60,
            n_quarters=4,
            future_start_date=str(backtest_start),
            future_end_date=str(backtest_end),
            epochs=20,
            batch_size=32,
        )

        if backtest_forecast is not None and not backtest_forecast.empty:
            actual_period = price_data[["Close"]].copy().rename(columns={"Close": "Actual_Close"}) #get & rename col
            actual_period = actual_period[
                (actual_period.index >= pd.to_datetime(backtest_start))
                & (actual_period.index <= pd.to_datetime(backtest_end))
            ] #get columns in test range

            eval_df = backtest_forecast.join(actual_period, how="inner") #join datasets

            if len(eval_df) > 0:
                pred_vals = eval_df["Predicted_Close"].values.astype(float)
                actual_vals = eval_df["Actual_Close"].values.astype(float)

                rmse = float(np.sqrt(mean_squared_error(actual_vals, pred_vals)))
                mae_val = float(mean_absolute_error(actual_vals, pred_vals))
                ss_res = float(np.sum((actual_vals - pred_vals) ** 2))
                ss_tot = float(np.sum((actual_vals - actual_vals.mean()) ** 2))
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                accuracy_children = html.Div([
                    html.Div([
                        html.Div([html.Div("R²", className="label"), html.Div(f"{r2:.4f}", className="value")], className="metric-mini"),
                        html.Div([html.Div("RMSE ($)", className="label"), html.Div(f"${rmse:.2f}", className="value")], className="metric-mini"),
                        html.Div([html.Div("MAE ($)", className="label"), html.Div(f"${mae_val:.2f}", className="value")], className="metric-mini"),
                        html.Div([html.Div("Test Window", className="label"), html.Div(f"{backtest_start} to {backtest_end}", className="value")], className="metric-mini"),
                        html.Div([html.Div("Predicted prices get more inaccurate as the forecast horizon increases.", className="label")], className="metric-mini"),
                    ], className="metric-row"),
                ])
            else:
                accuracy_children = html.Div(
                    "No actual closes are available for the selected period yet, so accuracy cannot be computed.",
                    className="info-card",
                    style={"color": "#555"},
                )
        #get actual prediction
        full_X_daily, full_X_fund, full_y, _full_dates, full_daily_scaler, full_fund_scaler = bm.build_sequences(
            dataset, fundamentals_df, sequence_length=60, n_quarters=4
        )
        full_model = bm.configure_model(60, full_X_daily.shape[2], 4, full_X_fund.shape[2])
        graph_forecast = bm.train_and_predict_future_period(
            model=full_model,
            X_daily=full_X_daily,
            X_fund=full_X_fund,
            y=full_y,
            daily_scaler=full_daily_scaler,
            fund_scaler=full_fund_scaler,
            fundamentals_df=fundamentals_df,
            fund_cols=fund_cols,
            sequence_length=60,
            n_quarters=4,
            future_start_date=str(start_dt),
            future_end_date=str(end_dt),
            epochs=20,
            batch_size=32,
        )

    except Exception:
        pass

    # --- Build chart data (historical context + predicted values) ---
    if graph_forecast is not None and not graph_forecast.empty:
        pred_chart = graph_forecast.reset_index().rename(
            columns={"Date": "date", "Predicted_Close": "close"}
        )
        pred_chart["type"] = "predicted"
        df_full = pd.concat([hist_chart, pred_chart], ignore_index=True)
    else:
        # Model trained but no future range; show historical context only
        df_full = hist_chart

    price_fig = _build_price_figure(ticker, df_full)

    # --- Sentiment: derived from price_data already fetched by build_full_dataset ---
    sentiment = {}
    if inc_sent or view_sent:
        last = price_data.iloc[-1]
        ma50 = last.get("MA_50") if "MA_50" in price_data.columns else None
        ma200 = last.get("MA_200") if "MA_200" in price_data.columns else None
        current_price = float(last["Close"])
        rsi = float(last["RSI"]) if "RSI" in price_data.columns and pd.notna(last["RSI"]) else 50.0
        if pd.notna(ma50) and pd.notna(ma200):
            if ma50 > ma200 and current_price > ma50:
                signal = "Bullish"
            elif ma50 < ma200 and current_price < ma50:
                signal = "Bearish"
            else:
                signal = "Neutral"
        else:
            signal = "Neutral"
        sentiment = {
            "Signal": signal,
            "RSI (14)": round(rsi, 1),
            "MA 50": f"${round(float(ma50), 2)}" if pd.notna(ma50) else "N/A",
            "MA 200": f"${round(float(ma200), 2)}" if pd.notna(ma200) else "N/A",
            "Current Price": f"${round(current_price, 2)}",
        }
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sentiment_fig = _build_sentiment_figure(sentiment, ticker) if sentiment else empty_fig

    # --- Recommended action ---
    signal = sentiment.get("Signal", "N/A") if sentiment else "N/A"
    action = "BUY" if signal == "Bullish" else "SELL" if signal == "Bearish" else "HOLD"
    if sentiment:
        rec_children = html.Div([
            html.Span("Recommended Action: ", style={"color": "#888", "fontSize": "0.82rem"}),
            html.Span(action, className=f"action-badge action-{action}"),
        ])
    else:
        rec_children = html.Div(
            "Enable 'Include Market Sentiment' to see a recommended action.",
            style={"color": "#555", "fontSize": "0.78rem"},
        )

    # --- Current fundamentals: derived from fundamentals_df already fetched by build_full_dataset ---
    fund_children = None
    if view_fund or inc_fund:
        if not fundamentals_df.empty:
            latest_fund = fundamentals_df.iloc[-1]
            display_cols = [
                "Total Revenue", "Gross Profit", "Operating Income", "Net Income",
                "EBITDA", "Total Assets", "Total Debt", "Free Cash Flow", "Operating Cash Flow",
                "Diluted EPS",
            ]
            fund_display = {}
            for col in display_cols:
                if col in latest_fund and pd.notna(latest_fund[col]):
                    val = float(latest_fund[col])
                    fund_display[col] = f"${val:,.0f}" if abs(val) >= 1 else f"{val:.4f}"
            for ratio_col, label in [("Gross_Margin", "Gross Margin"), ("Net_Margin", "Net Margin"),
                                     ("Operating_Margin", "Operating Margin"), ("ROA", "ROA"),
                                     ("Debt_To_Equity", "Debt / Equity")]:
                if ratio_col in latest_fund and pd.notna(latest_fund[ratio_col]):
                    fund_display[label] = f"{float(latest_fund[ratio_col])*100:.1f}%"
            fund_children = html.Div([
                html.H4("Latest Quarterly Fundamentals:"),
                *[html.Div([
                    html.Span(k, className="info-key"),
                    html.Span(str(v), className="info-val"),
                ], className="info-row") for k, v in fund_display.items()],
            ], className="info-card") if fund_display else html.Div(
                "Fundamentals unavailable.", className="info-card", style={"color": "#555"})
        else:
            fund_children = html.Div("Fundamentals unavailable.", className="info-card", style={"color": "#555"})

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
            sent_children = html.Div("Sentiment data unavailable.", className="info-card", style={"color": "#555"})

    return price_fig, sentiment_fig, rec_children, fund_children, sent_children, accuracy_children, ""


# Entry point

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)