# Energy Usage Forecasting

## Setup

```bash
python -m pip install -r requirements.txt
```

## Streamlit dashboard

Run the production-ready dashboard locally:

```bash
streamlit run streamlit_app.py
```

Notes:
- The dashboard loads `data/processed/global_active_power_hourly.csv` by default. Override with `ENERGY_APP_DATA_CSV=/path/to/your.csv`.
- For model-based forecasts, run the training scripts at least once to generate `outputs/` artifacts (e.g., `run_ml_models.py`, `prophet_forecasting.py`, `arima_forecasting.py`, `run_model_comparison.py`).
- Plot PNG downloads use `kaleido` (included in `requirements.txt`).

## Data preparation

Downloads the UCI household power consumption dataset (if missing), builds an hourly target series, engineers time-based features, and saves temporal train/val/test splits under `data/processed/`.

```bash
python prepare_time_series_data.py
```

## Prophet forecasting (train → test)

Runs Prophet with:
- daily + weekly seasonality enabled
- yearly seasonality enabled only if there is at least ~2 years of training data
- calendar regressors derived from existing time features (hour/day/month sin/cos, weekend/business hour)
- optional country holidays (default: France)

```bash
python prophet_forecasting.py --output-dir outputs/prophet
```

Artifacts:
- `outputs/prophet/prophet_forecast.csv`
- `outputs/prophet/prophet_forecast_vs_actual.png`
- `outputs/prophet/prophet_components.png`

## ARIMA baseline (SARIMAX, train → test)

Fits a small SARIMAX model on the last N training points (default 1500) and forecasts the test horizon.

```bash
python arima_forecasting.py --output-dir outputs/arima --max-train-points 1500
```

Artifacts:
- `outputs/arima/arima_forecast.csv`
- `outputs/arima/arima_forecast_vs_actual.png`

## Compare Prophet vs ARIMA

Runs both models, writes a metrics table, and saves an overlay plot.

```bash
python compare_prophet_arima.py --output-dir outputs --max-arima-train-points 1500
```

Artifacts:
- `outputs/compare/metrics_comparison.csv`
- `outputs/compare/forecast_overlay.png`

## LSTM neural network (24-hour lookback → next hour)

Trains an LSTM on sequences of the previous 24 hours to predict the next hour. It:
- scales inputs/targets with `MinMaxScaler` (fit on train only)
- uses `EarlyStopping` + `ModelCheckpoint`
- saves loss curves and test predictions/metrics

Note: TensorFlow support depends on your Python version. If `pip install tensorflow` fails on your interpreter, use a Python version supported by TensorFlow (commonly 3.10–3.12) or install a compatible TensorFlow build for your environment.

```bash
python run_lstm.py --lookback-hours 24 --epochs 50 --batch-size 128
```

Optional: include calendar/time features (still fed as sequences):

```bash
python run_lstm.py --include-time-features --lookback-hours 24
```

Artifacts:
- `outputs/lstm/best_model.keras`
- `outputs/lstm/training_history.png`
- `outputs/lstm/lstm_test_predictions.csv`
- `outputs/lstm/lstm_pred_vs_actual.png`
- `outputs/lstm/model_comparison.csv`
