# Adaptive Time Series Lab — Thesis Research Application

A bilingual (Ukrainian / English) Streamlit research app for nonstationary time series, stochastic forecasting and uncertainty quantification.

## Modules
- CSV / XLSX upload, column selection, descriptive statistics, ADF/KPSS, STL, PELT
- SARIMA (manual order or bounded AIC search) and State-Space/Kalman forecasting
- **Adaptive validation**: three strategies on the same rolling-origin dates:
  1. Static (fit once)
  2. Periodic refitting
  3. PELT-triggered refitting on recent change points in first differences
- Walk-forward evaluation with user-selected horizon and rolling training window
- Raw Gaussian prediction intervals at 80/90/95/99%, optional past-error-based conformal calibration
- MAE, RMSE, PICP, MPIW and Interval Score; exports of predictions, metrics and refit events

## Important methodology
All backtest forecasts use only training observations before the prediction origin. Calibration uses only previously observed errors. A fixed SARIMA specification (1,0,0)(1,1,0,s) is used inside the adaptive comparison to isolate the effect of the refitting policy; the Forecasting page offers order selection. Change-point detection is applied only to the training window, never the test period. PELT is applied to first differences; it is not a causal explanation of changes. A small number of test origins is only a functional demonstration and not adequate for statistical conclusions. No results are fabricated or substituted from the historical ENTSO-E benchmark for uploaded data.

## Deployment
Upload `app.py`, `adaptive_page.py`, `adaptive_engine.py`, `requirements.txt`, `.streamlit/config.toml`, and `data/` to the root of the existing GitHub repository. In Streamlit Community Cloud set the entry point to `app.py`. Python 3.11 or 3.12 is recommended. Dependencies install from `requirements.txt`.

## Data
ENTSO-E Italian daily electricity load 2016–2025 is supplied as a case study. The user can upload other CSV/XLSX datasets. The original historical 2025 benchmark, when displayed, is explicitly labelled as historical and is not recalculated for a different dataset.
