# Adaptive Time Series Lab — final UI edition

A bilingual (Ukrainian / English) research web application for adaptive stochastic modelling of nonstationary time series, prediction intervals and uncertainty assessment.

## Features

- Upload CSV/XLSX or use the supplied historical ENTSO-E Italy 2016–2025 daily series.
- Time series inspection, ACF/PACF, rolling statistics, ADF/KPSS, robust STL, PELT.
- SARIMA (manual or bounded AIC selection) and local-level seasonal state-space/Kalman forecasting.
- Adaptive walk-forward comparison: static, scheduled refitting, PELT-triggered refitting.
- 80%, 90%, 95%, 99% intervals, empirical coverage (PICP), mean interval width (MPIW), interval score, optional historical-residual calibration.
- Export detailed forecasts, metrics and refitting events to CSV.

## Deploy

Upload all files from this project directory to the **root** of the existing GitHub repository, preserving `.streamlit/config.toml` and the `data/` folder. The main file in Streamlit Community Cloud is `app.py` on branch `main`.

Locally: `pip install -r requirements.txt` then `streamlit run app.py`.

## Important methodological notes

- All current-dataset metrics are computed on the selected series; the 2025 benchmark is explicitly marked as a separate archived experiment.
- In the adaptive module, SARIMA order is intentionally fixed at (1,0,0)(1,1,0,s) to compare *refitting strategies* without confounding order selection. The forecasting page supports manual orders and bounded AIC search.
- The adaptive module is a retrospective historical experiment; it does not continuously watch live data. New data are used when the user uploads them and reruns the experiment.
- For multi-step horizons, calibration residuals are used only once their target date is already observed.
- Conformal-style empirical calibration on very small samples has no claimed formal finite-sample guarantee under time-series dependence.
- Small default test samples are intended to verify functionality, not to establish thesis-level statistical conclusions. Expand test periods and report compute budget, parameters and limitations.


## Scientific checks in v5

- Stationarity p-values displayed in readable scientific notation.
- Uploaded datasets are not mixed with saved ENTSO-E benchmarks.
- Methodology & reproducibility page includes limitations and an exportable dataset protocol.
- PELT-triggered strategy retains the last known changepoint when no new point is detected.
- Tested with syntax compilation; deploy and evaluate on your Streamlit Cloud instance.

**Research limitation:** no software can guarantee an examiner will have no questions; report test sizes, convergence, source data, and uncertainty transparently.

## Version 6.0 — Scientific experiment

The new **Scientific experiment / Науковий експеримент** page adds:
- Naive and seasonal naive out-of-sample benchmarks on identical origins;
- static, periodic, and PELT-triggered SARIMA / State-Space backtests;
- out-of-sample error bias, ACF and Ljung–Box diagnostics;
- nominal and past-error-calibrated 80/90/95/99% intervals with PICP, MPIW and interval score;
- PELT penalty sensitivity experiments;
- reproducibility metadata, CSV exports, and a PDF research report.

The comparison is a small configurable research experiment, **not** a claim that adaptive strategies are better. Refit and tuning choices require a separate holdout for final conclusions. Historical observations are never used for fitting before their date; interval calibration uses only matured forecast errors. Rolling conformal calibration on dependent observations does not carry a distribution-free coverage guarantee.

For deployment, commit `research_suite.py` next to `app.py`, alongside the other Python modules, and update `requirements.txt` (adds reportlab).
