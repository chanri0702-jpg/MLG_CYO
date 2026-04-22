import os
import pathlib
import datetime

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from data import build_model as bm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
# from tensorflow.keras.models import Model  # type: ignore
# from tensorflow.keras.layers import Input as KInput, Dense, LSTM, Dropout, Concatenate  # type: ignore

#Constants / paths

ARTIFACTS_DIR = pathlib.Path(__file__).parent.parent / "artifacts"

DROPDOWN_STYLE = {
    "backgroundColor": "#1a1a1a",
    "color": "#00FF41",
    "border": "1px solid #00FF41",
    "borderRadius": "5px",
}




#-----------------------------------------------------------ADD CHARTS------------------------------------------------------------------
#Price Prediction Chart:
def _build_price_figure(ticker: str, df_full: pd.DataFrame) -> go.Figure:
    """Build a Plotly figure from the combined historical + predicted DataFrame."""
    fig = go.Figure()

    #Filtering the data into 2 groups: the past data as well as the future predictions
    hist_df = df_full[df_full["type"] == "historical"]
    pred_df = df_full[df_full["type"] == "predicted"]

    #adding a line to represent the actual historical stick prices
    fig.add_trace(go.Scatter(
        x=hist_df["date"],
        y=hist_df["close"],
        name="Historical",
        line=dict(color="#36854a", width=2.5)
    ))
    #adding in a dotted line for the predicted prices
    if not pred_df.empty:
        #getting the last real price to attach the prediction line to
        last_hist = hist_df.iloc[-1:]
        conn_pred = pd.concat([last_hist, pred_df], ignore_index=True)
        fig.add_trace(go.Scatter(
            x=conn_pred["date"],
            y=conn_pred["close"],
            name="Predicted",
            line=dict(color="#cf3e3e", width=2, dash="dot")
        ))

    #refining the display of the graph
    fig.update_layout(
        title=f"Price Forecast: {ticker}",
        template="plotly_dark",
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        font=dict(family="Courier New", color="#f5f5f5"),
        xaxis=dict(showgrid=False, title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#333", title="Price ($)"),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    return fig

#Sentiment Gauge Chart
def _build_sentiment_figure(sentiment: dict, ticker: str) -> go.Figure:
    """Gauge chart for RSI-based sentiment."""
    #we are extracting the RSI value and the calculated signal from the data
    rsi_val = sentiment.get("RSI (14)", 50)
    signal = sentiment.get("Signal", "Neutral")

    #the logic in order to change the guage color
    #green: bullish (price going up)
    #red: bearish (price going down)
    #yellow: neutral
    color = "#ffb300"
    if signal == "Bullish":
        color = "#00c853"
    elif signal == "Bearish":
        color = "#ff5252"

    #setting up the gauge/indicator component
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        title={'text': f"RSI Indicator ({signal})", 'font': {'size': 14}},
        number={'font': {'color': color}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'bgcolor': "#222",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255, 82, 82, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(0, 200, 83, 0.1)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'value': rsi_val,
            },
        },
    ))
    #matching the gauge background and font to the rest of our dashboard
    fig.update_layout(
        paper_bgcolor="#222",
        font={'color': "#f5f5f5", 'family': "Courier New"},
        margin=dict(l=30, r=30, t=50, b=20),
        height=260,
    )
    return fig


def _build_feature_importance_figure(importance_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    if importance_df is not None and not importance_df.empty:
        plot_df = importance_df.head(10).sort_values("importance", ascending=True)
        fig.add_trace(go.Bar(
            x=plot_df["importance"],
            y=plot_df["feature"],
            orientation="h",
            marker=dict(color="#00b894"),
            name="Importance",
        ))

    fig.update_layout(
        title=f"Top 10 Features Influencing {ticker} Price",
        template="plotly_dark",
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        font=dict(family="Courier New", color="#f5f5f5"),
        xaxis=dict(title="Aggregated Importance"),
        yaxis=dict(title="Feature"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _build_top_feature_trends_figure(dataset: pd.DataFrame, top_features: list, ticker: str) -> go.Figure:
    valid_features = [f for f in top_features if f in dataset.columns]
    if not valid_features:
        fig = go.Figure()
        fig.update_layout(
            title=f"Top Feature Trends: {ticker}",
            template="plotly_dark",
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            font=dict(family="Courier New", color="#f5f5f5"),
        )
        return fig

    rows, cols = 5, 2
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=valid_features[:10],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    recent = dataset.tail(180)
    for i, feature in enumerate(valid_features[:10]):
        r = i // cols + 1
        c = i % cols + 1
        series = pd.to_numeric(recent[feature], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=series,
                mode="lines",
                line=dict(width=1.8, color="#4fc3f7"),
                name=feature,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        title=f"Top 10 Feature Trends (Last 180 Rows): {ticker}",
        template="plotly_dark",
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        font=dict(family="Courier New", color="#f5f5f5"),
        height=1200,
        margin=dict(l=10, r=10, t=70, b=20),
    )
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
                        dcc.Tabs(
                            id="view-tabs",
                            value="tab-predict",
                            className="analytics-tabs",
                            parent_className="analytics-tabs-parent",
                            content_className="analytics-tabs-content",
                            children=[
                                dcc.Tab(
                                    label="Prediction",
                                    value="tab-predict",
                                    className="analytics-tab",
                                    selected_className="analytics-tab--selected",
                                    children=[
                                        dcc.Graph(
                                            id="graph-stock",
                                            figure=go.Figure().update_layout(
                                                title="Graph: Stock Prices",
                                                paper_bgcolor="#111", plot_bgcolor="#111",
                                                font_color="#555",
                                                margin=dict(l=8, r=8, t=40, b=8),
                                            ),
                                            className="graph-stock",
                                            style={"height": "clamp(280px, 38vh, 420px)", "border": "1px solid #333", "borderRadius": "8px"},
                                        ),
                                        html.Div(
                                            id="recommended-action",
                                            style={"color": "#888", "fontSize": "0.82rem"},
                                        ),
                                        dcc.Graph(
                                            id="graph-sentiment",
                                            figure=go.Figure().update_layout(
                                                title="Market Sentiment",
                                                paper_bgcolor="#111", plot_bgcolor="#111",
                                                font_color="#555",
                                                margin=dict(l=20, r=20, t=60, b=20),
                                            ),
                                            className="graph-sentiment",
                                            style={"height": "clamp(220px, 32vh, 320px)", "border": "1px solid #333", "borderRadius": "8px"},
                                        ),
                                        html.Div(
                                            id="current-fundamentals",
                                            style={"color": "#888", "fontSize": "0.82rem"},
                                        ),
                                        html.Div(
                                            id="current-sentiment",
                                            style={"color": "#888", "fontSize": "0.82rem"},
                                        ),
                                        html.Div(
                                            id="model-accuracy",
                                            style={"color": "#888", "fontSize": "0.82rem"},
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Data Insights",
                                    value="tab-insights",
                                    className="analytics-tab",
                                    selected_className="analytics-tab--selected",
                                    children=[
                                        html.Div(
                                            id="dataset-overview",
                                            style={"color": "#cfd8dc", "fontSize": "0.82rem", "marginBottom": "12px"},
                                        ),
                                        dcc.Graph(
                                            id="feature-importance-graph",
                                            figure=go.Figure().update_layout(
                                                title="Top 10 Feature Importance",
                                                paper_bgcolor="#111", plot_bgcolor="#111",
                                                font_color="#555",
                                                margin=dict(l=20, r=20, t=60, b=20),
                                            ),
                                            className="graph-importance",
                                            style={"height": "clamp(320px, 48vh, 520px)", "border": "1px solid #333", "borderRadius": "8px"},
                                        ),
                                        dcc.Graph(
                                            id="top-features-trend-graph",
                                            figure=go.Figure().update_layout(
                                                title="Top Feature Trends",
                                                paper_bgcolor="#111", plot_bgcolor="#111",
                                                font_color="#555",
                                                margin=dict(l=20, r=20, t=60, b=20),
                                            ),
                                            className="graph-top-features",
                                            style={"height": "clamp(540px, 95vh, 1200px)", "border": "1px solid #333", "borderRadius": "8px"},
                                        ),
                                    ],
                                ),
                            ],
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
    Output("dataset-overview", "children"),
    Output("feature-importance-graph", "figure"),
    Output("top-features-trend-graph", "figure"),
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

    empty_overview = html.Div("Run a prediction to view dataset insights.", style={"color": "#777"})

    if not ticker or not ticker.strip():
        return empty_fig, empty_fig, None, None, None, None, empty_overview, empty_fig, empty_fig, "Please enter a ticker symbol."

    ticker = ticker.strip().upper()

    if not bm.ticker_exists(ticker):
        return empty_fig, empty_fig, None, None, None, None, empty_overview, empty_fig, empty_fig, f"Ticker '{ticker}' does not exist."

    today = datetime.date.today()

    # --- Parse dates ---
    try:
        start_dt = datetime.datetime.strptime(start_input.strip(), "%Y-%m-%d").date() if start_input else today
        end_dt = datetime.datetime.strptime(end_input.strip(), "%Y-%m-%d").date() if end_input else today + datetime.timedelta(days=30)
        if start_dt >= end_dt:
            raise ValueError("Start must be before end.")
    except (ValueError, AttributeError) as exc:
        return empty_fig, empty_fig, None, None, None, None, empty_overview, empty_fig, empty_fig, f"Date error: {exc}"

    # --- Fetch and build dataset using full historical window ---
    try:
        hist_start, hist_end = bm.get_date_range(ticker)
        dataset, sentiment_df, fundamentals_df, price_data = bm.build_full_dataset(ticker, hist_start, hist_end)
    except Exception as exc:
        return empty_fig, empty_fig, None, None, None, None, empty_overview, empty_fig, empty_fig, f"Data fetch error: {exc}"

    if dataset.empty:
        return empty_fig, empty_fig, None, None, None, None, empty_overview, empty_fig, empty_fig, f"No data available for '{ticker}'."

    dataset_overview_children = html.Div([
        html.H4("Dataset Summary"),
        html.Div(f"Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns"),
        html.Div("Columns:"),
        html.Ul([html.Li(col) for col in dataset.columns.tolist()]),
    ], className="info-card")

    feature_importance_fig = empty_fig
    top_feature_trend_fig = empty_fig

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
    # Always show 0 days of context ending at today (or at forecast start if it's in the past)
    context_anchor = min(start_dt, today)
    context_start = context_anchor - datetime.timedelta(days=90)
    hist_mask = (price_data.index >= pd.to_datetime(context_start)) & (price_data.index <= pd.to_datetime(today))
    price_display = price_data[hist_mask] if hist_mask.any() else price_data

    hist_chart = price_display[["Close"]].rename(columns={"Close": "close"}).copy()
    hist_chart.index.name = "date"
    hist_chart = hist_chart.reset_index()
    hist_chart["type"] = "historical"

    # --- ML pipeline: recent-window backtest for accuracy, then full-data forecast for the chart ---
    # fund_cols defined here so it's available to both the backtest and full-model blocks
    fund_cols = fundamentals_df.select_dtypes(include=[np.number]).columns.tolist()

    backtest_forecast = None
    graph_forecast = None
    accuracy_children = None

    try:
        # --- BACKTEST BLOCK COMMENTED OUT (trains a second model; too slow) ---
        # train_cutoff = pd.to_datetime(backtest_start) - pd.Timedelta(days=1)
        # train_dataset = dataset[dataset.index <= train_cutoff]
        # if len(train_dataset) < 90:
        #     raise ValueError("Not enough pre-period history to train the model for this test window")
        #
        # X_daily, X_fund, y, sequence_dates, daily_scaler, fund_scaler = bm.build_sequences(
        #     train_dataset, fundamentals_df, sequence_length=60, n_quarters=4
        # )
        # n_daily_features = X_daily.shape[2]
        # n_fund_features = X_fund.shape[2]
        # model = bm.configure_model(60, n_daily_features, 4, n_fund_features)
        #
        # backtest_forecast = bm.train_and_predict_future_period(
        #     model=model,
        #     X_daily=X_daily,
        #     X_fund=X_fund,
        #     y=y,
        #     daily_scaler=daily_scaler,
        #     fund_scaler=fund_scaler,
        #     fundamentals_df=fundamentals_df,
        #     fund_cols=fund_cols,
        #     sequence_length=60,
        #     n_quarters=4,
        #     future_start_date=str(backtest_start),
        #     future_end_date=str(backtest_end),
        #     epochs=150,
        #     batch_size=32,
        # )
        #
        # if backtest_forecast is not None and not backtest_forecast.empty:
        #     actual_period = price_data[["Close"]].copy().rename(columns={"Close": "Actual_Close"})
        #     actual_period = actual_period[
        #         (actual_period.index >= pd.to_datetime(backtest_start))
        #         & (actual_period.index <= pd.to_datetime(backtest_end))
        #     ]
        #     eval_df = backtest_forecast.join(actual_period, how="inner")
        #     if len(eval_df) > 0:
        #         pred_vals = eval_df["Predicted_Close"].values.astype(float)
        #         actual_vals = eval_df["Actual_Close"].values.astype(float)
        #         rmse = float(np.sqrt(mean_squared_error(actual_vals, pred_vals)))
        #         mae_val = float(mean_absolute_error(actual_vals, pred_vals))
        #         ss_res = float(np.sum((actual_vals - pred_vals) ** 2))
        #         ss_tot = float(np.sum((actual_vals - actual_vals.mean()) ** 2))
        #         r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        #         accuracy_children = html.Div([...])
        # --- END BACKTEST BLOCK ---

        # Build and validate a fresh XGBoost model for every request (no cache).
        full_X_daily, full_X_fund, full_y, _full_dates, full_daily_scaler, full_fund_scaler = bm.build_sequences(
            dataset, fundamentals_df, sequence_length=60, n_quarters=4
        )
        full_model = bm.configure_model()
        validation_metrics = bm.train_test_validate_model(
            full_model,
            full_X_daily,
            full_X_fund,
            full_y,
            test_size=0.2,
        )

        # Aggregate flattened sequence importances back to base feature names.
        importances = getattr(full_model, "feature_importances_", None)
        top_features = []
        if importances is not None and len(importances) > 0:
            daily_cols = dataset.columns.tolist()
            n_daily = len(daily_cols)
            n_fund = len(fund_cols)
            daily_span = 60 * n_daily
            if len(importances) >= daily_span:
                daily_imp = np.array(importances[:daily_span]).reshape(60, n_daily).sum(axis=0)
                fund_part = np.array(importances[daily_span:])
                if n_fund > 0 and len(fund_part) >= 4 * n_fund:
                    fund_imp = fund_part[:4 * n_fund].reshape(4, n_fund).sum(axis=0)
                else:
                    fund_imp = np.zeros(n_fund, dtype=np.float32)

                agg = {}
                for i, col in enumerate(daily_cols):
                    agg[col] = agg.get(col, 0.0) + float(daily_imp[i])
                for i, col in enumerate(fund_cols):
                    agg[col] = agg.get(col, 0.0) + float(fund_imp[i])

                imp_df = (
                    pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
                    .sort_values("importance", ascending=False)
                )
                top_features = imp_df["feature"].head(10).tolist()
                feature_importance_fig = _build_feature_importance_figure(imp_df, ticker)
                top_feature_trend_fig = _build_top_feature_trends_figure(dataset, top_features, ticker)

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
            epochs=0,
            batch_size=32,
        )

        # XGBoost validation metrics come from holdout test split (scaled space).
        mse_scaled = float(validation_metrics["mse"])
        mae_scaled = float(validation_metrics["mae"])

        scale_range = full_daily_scaler.data_max_[3] - full_daily_scaler.data_min_[3]
        rmse    = float(np.sqrt(mse_scaled)) * scale_range
        mae_val = mae_scaled * scale_range
        r2      = float(validation_metrics["r2"])

        # --- OLD TensorFlow evaluate() approach (kept for reference) ---
        # eval_results = full_model.evaluate([full_X_daily, full_X_fund], full_y, verbose=0)
        # mse_scaled = float(eval_results[0])
        # mae_scaled = float(eval_results[1])
        # scale_range = full_daily_scaler.data_max_[3] - full_daily_scaler.data_min_[3]
        # rmse    = float(np.sqrt(mse_scaled)) * scale_range
        # mae_val = mae_scaled * scale_range
        # y_var   = float(np.var(full_y))
        # r2      = float(1 - mse_scaled / y_var) if y_var > 0 else 0.0

        # --- OLD accuracy approach (manual predict + sklearn) — kept for reference ---
        # train_pred_scaled = full_model.predict([full_X_daily, full_X_fund], verbose=0).flatten()
        # scale_min = full_daily_scaler.data_min_[3]
        # scale_range = full_daily_scaler.data_max_[3] - scale_min
        # train_pred_dollars = train_pred_scaled * scale_range + scale_min
        # actual_dollars = full_y * scale_range + scale_min
        # rmse = float(np.sqrt(mean_squared_error(actual_dollars, train_pred_dollars)))
        # mae_val = float(mean_absolute_error(actual_dollars, train_pred_dollars))
        # ss_res = float(np.sum((actual_dollars - train_pred_dollars) ** 2))
        # ss_tot = float(np.sum((actual_dollars - actual_dollars.mean()) ** 2))
        # r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        accuracy_children = html.Div([
            html.Div([
                html.Div([html.Div("R²", className="label"), html.Div(f"{r2:.4f}", className="value")], className="metric-mini"),
                html.Div([html.Div("RMSE ($)", className="label"), html.Div(f"${rmse:.2f}", className="value")], className="metric-mini"),
                html.Div([html.Div("MAE ($)", className="label"), html.Div(f"${mae_val:.2f}", className="value")], className="metric-mini"),
                html.Div([html.Div("Reported on holdout test split (20%) using XGBoost.", className="label")], className="metric-mini"),
                html.Div([html.Div("Predicted prices get more inaccurate as the forecast horizon increases.", className="label")], className="metric-mini"),
            ], className="metric-row"),
        ])

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
            # Stock-level sentiment (RSI, MAs, signal)
            stock_sent_rows = [html.Div([
                html.Span(k, className="info-key"),
                html.Span(str(v), className="info-val"),
            ], className="info-row") for k, v in sentiment.items()]

            # Macro market sentiment — latest row from sentiment_df fetched by build_full_dataset
            macro_rows = []
            if sentiment_df is not None and not sentiment_df.empty:
                latest_macro = sentiment_df.iloc[-1]
                macro_as_of = sentiment_df.index[-1]
                macro_display = {
                    "VIX (Fear Index)": f"{float(latest_macro['VIX']):.2f}" if "VIX" in latest_macro and pd.notna(latest_macro["VIX"]) else "N/A",
                    "VIX 20-day MA": f"{float(latest_macro['VIX_MA20']):.2f}" if "VIX_MA20" in latest_macro and pd.notna(latest_macro["VIX_MA20"]) else "N/A",
                    "S&P 500": f"${float(latest_macro['SP500']):,.2f}" if "SP500" in latest_macro and pd.notna(latest_macro["SP500"]) else "N/A",
                    "S&P 500 Daily Return": f"{float(latest_macro['SP500_Return'])*100:.2f}%" if "SP500_Return" in latest_macro and pd.notna(latest_macro["SP500_Return"]) else "N/A",
                    "10Y Treasury Yield": f"{float(latest_macro['Treasury_10Y']):.3f}%" if "Treasury_10Y" in latest_macro and pd.notna(latest_macro["Treasury_10Y"]) else "N/A",
                    "Market Regime": "Bull Market" if latest_macro.get("Market_Regime") == 1 else "Bear Market",
                }
                macro_rows = [
                    html.H4(f"Macro Market Data (as of {macro_as_of}):"),
                    *[html.Div([
                        html.Span(k, className="info-key"),
                        html.Span(str(v), className="info-val"),
                    ], className="info-row") for k, v in macro_display.items()],
                ]

            sent_children = html.Div([
                html.H4("Current Stock Sentiment:"),
                *stock_sent_rows,
                *macro_rows,
            ], className="info-card")
        else:
            sent_children = html.Div("Sentiment data unavailable.", className="info-card", style={"color": "#555"})

    return (
        price_fig,
        sentiment_fig,
        rec_children,
        fund_children,
        sent_children,
        accuracy_children,
        dataset_overview_children,
        feature_importance_fig,
        top_feature_trend_fig,
        "",
    )


# Entry point

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
