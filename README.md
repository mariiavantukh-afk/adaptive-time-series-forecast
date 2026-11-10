# Time Series Analytics — bilingual research application

A Streamlit application for stochastic time-series analysis, SARIMA and state-space forecasting, interval estimates, STL decomposition, ADF/KPSS and PELT.

## Language
Use the **Мова / Language** selector in the sidebar to switch between Ukrainian and English. The English translation covers navigation and principal controls; some research annotations and exported labels remain in Ukrainian in this version.

## Forecasting
SARIMA can be fitted manually with p,d,q,P,D,Q and seasonal period s, or chosen by a bounded AIC search over 18 candidates. This is not a global optimization and does not establish forecasting superiority. Forecasts and intervals are recomputed from the currently selected dataset. Historical ENTSO-E 2025 benchmark results remain separately labeled and are not recomputed for uploaded files.

## Deploy
Upload the contents of this folder to the root of your existing GitHub repository. Streamlit Cloud main file: `app.py`; branch: `main`. Keep `.streamlit/config.toml` and the `data` folder.
