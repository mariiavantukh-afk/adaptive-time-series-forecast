from pathlib import Path
from adaptive_page import adaptive_page
import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

st.set_page_config(page_title='Time Series Analytics | Research application', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
NAV = ['Огляд дослідження', 'Дані та підготовка', 'Статистичний аналіз', 'Стаціонарність', 'Декомпозиція STL', 'Структурні зміни PELT', 'Прогнозування', 'Адаптивна валідація', 'Порівняння моделей', 'Методологія та відтворюваність']
TEAL, NAVY, MUTED = '#087F8C', '#172B4D', '#60748B'

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
.stApp {background: #F5F7FB; color:#182B47}
.block-container {padding-top:2.0rem;padding-bottom:3rem;max-width:1440px}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#102C4B 0%,#17395B 100%);border-right:0}
[data-testid="stSidebar"] {color:#EAF2FC}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4,[data-testid="stSidebar"] p {color:#EAF2FC}
[data-testid="stSidebar"] .stRadio label p {color:#EAF2FC}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {background:#F8FAFC}
[data-testid="stSidebar"] [data-testid="stFileUploader"] p {color:#17395B!important}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#C9D7E8!important}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#182B47!important}
[data-testid="stSidebar"] [data-baseweb="select"] {background:white;border-radius:9px}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {border-radius:12px}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {color:#17395B!important}
[data-testid="stSidebar"] .stRadio label {padding:.30rem .5rem;border-radius:8px;font-size:.91rem}
[data-testid="stSidebar"] hr {border-color:#46617E}
h1,h2,h3 {font-family:'Manrope',sans-serif;color:#142D4E;letter-spacing:-.035em}
h1 {font-weight:800!important} h2 {font-weight:750!important}
[data-testid="stAppViewContainer"] {background:#F5F7FB}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#E3EAF2}
[data-testid="stAlert"] {border-radius:12px}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#265475;border-radius:10px}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#234765}
[data-testid="stHeader"] {background:rgba(245,247,251,.94)}
[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E0E7F0;border-radius:15px;padding:19px 20px;box-shadow:0 6px 25px rgba(15,39,70,.04)}
[data-testid="stMetricLabel"] {color:#6C7D94}
[data-testid="stMetricValue"] {color:#142D4E;font-family:'Manrope',sans-serif;font-size:1.65rem}
[data-testid="stPlotlyChart"] {background:white;border:1px solid #E2E9F2;border-radius:16px;padding:12px;box-shadow:0 5px 20px rgba(15,39,70,.035)}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden}
.stButton>button[kind="primary"] {background:#087F8C;border:1px solid #087F8C;border-radius:10px;font-weight:700;padding:.62rem 1.1rem}
.stButton>button[kind="primary"]:hover {background:#066B76;border-color:#066B76}
.stDownloadButton>button {border-radius:10px;border-color:#D3DFEC}
[data-baseweb="tab-list"] {background:#FFFFFF;border:1px solid #E2E9F2;padding:7px;border-radius:12px;gap:7px}
button[data-baseweb="tab"] {border-radius:8px;padding:10px 17px}
[data-testid="stExpander"] {background:white;border:1px solid #E1E8F0;border-radius:12px}
.hero {background:linear-gradient(115deg,#102C4B 0%,#205577 65%,#087F8C 130%);padding:34px 38px;border-radius:20px;color:white;margin-bottom:25px;box-shadow:0 10px 30px rgba(15,45,76,.14)}
.hero h1 {color:white!important;font-size:2.2rem;margin:0 0 10px 0}
.hero p {color:#D8E9F3;margin:0;font-size:1rem;max-width:780px}
.eyebrow {color:#087F8C;font-size:.77rem;letter-spacing:.14em;text-transform:uppercase;font-weight:800;margin-bottom:8px}
.note {background:#E9F4F5;padding:15px 19px;border-left:4px solid #087F8C;border-radius:9px;margin:12px 0;color:#214F5B}
.small {color:#64748B;font-size:.89rem}
.feature-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:16px 0 24px}
.feature {background:white;border:1px solid #E0E7F0;border-radius:14px;padding:19px 20px}
.feature strong {display:block;color:#142D4E;font-size:1rem;margin-bottom:6px}
.feature span {color:#65788D;font-size:.87rem;line-height:1.45}
@media(max-width:800px){.feature-grid{grid-template-columns:1fr}.hero{padding:22px}.hero h1{font-size:1.6rem}}
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo():
    return pd.read_csv(DATA/'italy_daily_load_2016_2025.csv')

@st.cache_data(show_spinner=False)
def load_validation():
    return pd.read_csv(DATA/'italy_2025_rolling_forecasts.csv',parse_dates=['date']),pd.read_csv(DATA/'italy_2025_rolling_metrics.csv')

def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')

def read_file(upload):
    raw=upload.getvalue()
    if upload.name.lower().endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(raw))
    for encoding in ('utf-8-sig','utf-8','cp1251'):
        try:
            df=pd.read_csv(io.BytesIO(raw),encoding=encoding,sep=None,engine='python')
            if df.shape[1]>1:return df
        except (UnicodeError,pd.errors.ParserError):pass
    return pd.read_csv(io.BytesIO(raw))

def infer_date(cols):
    for c in cols:
        if str(c).strip().lower() in ('date','datetime','timestamp','дата','time','day'):return c
    for c in cols:
        if any(x in str(c).lower() for x in ('date','time','дата')):return c
    return cols[0]

def infer_value(cols,date_col):
    preferred=['mean_load_MW','load_MW','value','load','y']
    for c in preferred:
        if c in cols and c!=date_col:return c
    return next((c for c in cols if c!=date_col),date_col)

def prepare(raw,date_col,value_col,frequency):
    dates=pd.to_datetime(raw[date_col],errors='coerce',dayfirst=False)
    values=pd.to_numeric(raw[value_col].astype(str).str.replace(' ','',regex=False).str.replace(',','.',regex=False),errors='coerce')
    df=pd.DataFrame({'date':dates,'value':values}).dropna().sort_values('date')
    df=df.groupby('date')['value'].mean().sort_index()
    if frequency!='Без агрегування':df=df.resample({'Доба':'D','Тиждень':'W','Місяць':'MS'}[frequency]).mean()
    return df.dropna().astype(float)

def plot_style(fig,height=430,yaxis=None):
    if yaxis is None:yaxis = 'Value' if EN else 'Значення'
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=34,b=20),paper_bgcolor='white',plot_bgcolor='white',font=dict(color=NAVY),legend=dict(orientation='h',y=1.12,x=0),hovermode='x unified',xaxis=dict(showgrid=False,title='Date' if EN else 'Дата'),yaxis=dict(title=yaxis,gridcolor='#E9EEF5'))
    return fig

def line_figure(x,y,label=None,color=TEAL):
    if label is None:label = 'Series' if EN else 'Часовий ряд'
    fig=go.Figure(go.Scatter(x=x,y=y,name=label,line=dict(color=color,width=2)))
    return plot_style(fig)

def section(title,subtitle=None):
    st.markdown(f'<div class="eyebrow">{"MASTER’S RESEARCH" if EN else "МАГІСТЕРСЬКЕ ДОСЛІДЖЕННЯ"}</div><h2 style="margin-top:0">{title}</h2>',unsafe_allow_html=True)
    if subtitle:st.caption(subtitle)

def date_index(series,steps):
    freq=pd.infer_freq(series.index)
    if not freq:
        differences=series.index.to_series().diff().dropna()
        if len(differences) and differences.nunique()==1:freq=differences.iloc[0]
        else:raise ValueError('Для прогнозування потрібен регулярний часовий індекс. Оберіть агрегацію «Доба», «Тиждень» або «Місяць».')
    return pd.date_range(start=series.index[-1],periods=steps+1,freq=freq)[1:]

language = st.sidebar.selectbox('Мова / Language', ['Українська', 'English'], key='ui_language')
EN = language == 'English'
TRANSLATIONS = {'Методологія та відтворюваність': 'Methodology & reproducibility', 'Огляд дослідження': 'Research overview', 'Дані та підготовка': 'Data preparation', 'Статистичний аналіз': 'Statistical analysis', 'Стаціонарність': 'Stationarity', 'Декомпозиція STL': 'STL decomposition', 'Структурні зміни PELT': 'PELT change points', 'Прогнозування': 'Forecasting', 'Адаптивна валідація': 'Adaptive validation', 'Порівняння моделей': 'Model comparison', 'Навігація': 'Navigation', 'Джерело даних': 'Data source', 'Завантажити CSV або XLSX': 'Upload CSV or XLSX', 'Стовпець дати': 'Date column', 'Числова змінна': 'Numeric variable', 'Агрегування': 'Aggregation', 'Без агрегування': 'No aggregation', 'Доба': 'Daily', 'Тиждень': 'Weekly', 'Місяць': 'Monthly', 'Коректних спостережень': 'Valid observations', 'Сезонний період': 'Seasonal period', 'Горизонт прогнозу': 'Forecast horizon', 'Прогнозна модель': 'Forecast model', 'Кількість останніх спостережень для навчання': 'Training observations', 'Побудувати прогноз': 'Generate forecast', 'Обчислити точки зміни': 'Detect change points', 'Ручні параметри SARIMA': 'Manual SARIMA parameters', 'Автоматичний підбір (AIC)': 'Automatic selection (AIC)', 'Режим SARIMA': 'SARIMA mode', 'Рівень прогнозного інтервалу': 'Prediction interval confidence', 'Параметри несезонної частини': 'Nonseasonal parameters', 'Параметри сезонної частини': 'Seasonal parameters', 'Побудова прогнозів': 'Forecast generation', 'Прогнозні значення та 95% інтервали обчислюються на обраному часовому ряді.': 'Point forecasts and prediction intervals are calculated for the selected time series.', 'Порівняння прогнозних моделей': 'Forecast model comparison', 'Збережені результати окремого експерименту rolling-origin за 2025 рік.': 'Saved results from the 2025 rolling-origin experiment.', 'Перевірка стаціонарності': 'Stationarity testing', 'Сезонно-трендова декомпозиція STL': 'Seasonal-trend decomposition (STL)', 'Виявлення структурних змін методом PELT': 'Change-point detection (PELT)', 'Описова статистика та автокореляція': 'Descriptive statistics and autocorrelation', 'Вхідні дані та попередня обробка': 'Input data and preprocessing', 'Спостережень': 'Observations', 'Середнє': 'Mean', 'Мінімум': 'Minimum', 'Максимум': 'Maximum', 'Динаміка досліджуваного показника': 'Time series overview', 'Про дослідження': 'About the research', 'Джерело': 'Source', 'Медіана': 'Median', 'Стандартне відхилення': 'Standard deviation', 'Коефіцієнт варіації': 'Coefficient of variation', 'Кількість лагів': 'Number of lags', 'Вікно ковзних статистик': 'Rolling window', 'Вихідний ряд': 'Observed', 'Тренд': 'Trend', 'Сезонність': 'Seasonality', 'Залишок': 'Residual', 'Поточний ряд': 'Current series', 'Мінімальний розмір сегмента': 'Minimum segment size', 'Множник штрафу': 'Penalty multiplier', 'Виявлених точок зміни': 'Detected change points', 'Фактичні дані': 'Observed data', 'Прогноз': 'Forecast', 'Фактичні значення': 'Observed values', 'Модель': 'Model', 'Одиничний корінь': 'Unit root', 'Стаціонарність за рівнем': 'Level stationarity', 'Статистика': 'Statistic', 'Критерій': 'Test', 'Нульова гіпотеза': 'Null hypothesis', '↓ Експортувати прогноз': '↓ Export forecast', '↓ Експортувати результати тестів': '↓ Export test results', '↓ Завантажити компоненти STL': '↓ Download STL components', '↓ Експортувати сегменти': '↓ Export segments', '↓ Таблиця результатів': '↓ Export metrics', '↓ 365 прогнозів за 2025 рік': '↓ Export 2025 forecasts', 'Рядків у файлі': 'File rows', 'Частота': 'Frequency', 'Таблиця спостережень': 'Observations table', 'Контроль вихідного файлу': 'Input file quality', 'Завантажити': 'Download'}
def tr(value):
    return TRANSLATIONS.get(value, value) if EN else value

st.sidebar.markdown('### ◈ Adaptive Time Series Lab')
st.sidebar.caption('STOCHASTIC FORECASTING  /  RESEARCH PLATFORM')
st.sidebar.markdown('##### '+('WORKSPACE' if EN else 'РОБОЧИЙ ПРОСТІР'))
page_label=st.sidebar.radio(tr('Навігація'),[tr(x) for x in NAV],label_visibility='collapsed')
page=NAV[[tr(x) for x in NAV].index(page_label)]
st.sidebar.divider()
st.sidebar.markdown('#### '+tr('Джерело даних'))
upload=st.sidebar.file_uploader(tr('Завантажити CSV або XLSX'),type=['csv','xlsx'])
try:
    raw=read_file(upload) if upload else load_demo()
except Exception as exc:
    st.error(f'Помилка читання файлу: {exc}');st.stop()
if not upload:st.sidebar.info('ENTSO-E data · Italy · 2016–2025' if EN else 'Дані ENTSO-E · Італія · 2016–2025')
cols=list(raw.columns)
if len(cols)<2:st.error('Файл повинен містити щонайменше два стовпці: дату та значення.');st.stop()
preferred_date=infer_date(cols)
date_col=st.sidebar.selectbox(tr('Стовпець дати'),cols,index=cols.index(preferred_date))
value_candidates=[c for c in cols if c!=date_col]
preferred_value=infer_value(cols,date_col)
value_col=st.sidebar.selectbox(tr('Числова змінна'),value_candidates,index=value_candidates.index(preferred_value) if preferred_value in value_candidates else 0)
frequency=st.sidebar.selectbox(tr('Агрегування'),['Без агрегування','Доба','Тиждень','Місяць'])
series=prepare(raw,date_col,value_col,frequency)
if series.index.has_duplicates:
    st.sidebar.warning('Duplicate timestamps were aggregated by their mean.' if EN else 'Повторні часові мітки агреговано середнім.')

st.sidebar.caption((f'Valid observations: {len(series):,}' if EN else f'Коректних спостережень: {len(series):,}').replace(',',' '))
if len(series)<15:
    st.error('Потрібно щонайменше 15 коректних спостережень. Перевірте вибір стовпців дати й числової змінної.');st.stop()

if page=='Огляд дослідження':
    st.markdown('<div class="hero"><div style="color:#9FDEE1;font-weight:700;letter-spacing:.15em;font-size:.75rem;margin-bottom:10px">RESEARCH WORKSPACE  /  VERSION 4.0</div><h1>Adaptive Time Series Lab</h1><p>Stochastic modelling · change-point adaptation · walk-forward validation · prediction intervals</p></div>' if EN else '<div class="hero"><div style="color:#9FDEE1;font-weight:700;letter-spacing:.15em;font-size:.75rem;margin-bottom:10px">ДОСЛІДНИЦЬКА ПЛАТФОРМА  /  ВЕРСІЯ 4.0</div><h1>Адаптивне моделювання часових рядів</h1><p>Нестаціонарність · структурні зміни · адаптивне прогнозування · оцінювання невизначеності</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric(tr('Спостережень'),f'{len(series):,}'.replace(',',' '))
    b.metric(tr('Середнє'),f'{series.mean():,.0f}'.replace(',',' '))
    c.metric(tr('Мінімум'),f'{series.min():,.0f}'.replace(',',' '))
    d.metric(tr('Максимум'),f'{series.max():,.0f}'.replace(',',' '))
    st.markdown('<div class="feature-grid"><div class="feature"><strong>01 · '+('Diagnostics' if EN else 'Діагностика')+'</strong><span>ADF / KPSS · STL · ACF / PACF · PELT</span></div><div class="feature"><strong>02 · '+('Modelling' if EN else 'Моделювання')+'</strong><span>SARIMA · State-Space / Kalman · '+('automatic order selection' if EN else 'підбір параметрів')+'</span></div><div class="feature"><strong>03 · '+('Validation & uncertainty' if EN else 'Валідація та невизначеність')+'</strong><span>Walk-forward · PICP · MPIW · Interval Score</span></div></div>',unsafe_allow_html=True)
    overview_tab,info_tab=st.tabs(['📈 '+('Series dashboard' if EN else 'Аналітична панель'),'📖 '+('Research context' if EN else 'Про дослідження')])
    with overview_tab:
        st.subheader(tr('Динаміка досліджуваного показника'))
        st.plotly_chart(line_figure(series.index,series.values),use_container_width=True)
    with info_tab:
        st.markdown('#### '+('Purpose' if EN else 'Мета застосунку'))
        st.write('A research platform for nonstationary time series, structural changes, adaptive stochastic models and interval forecasting. All calculations use the selected series.' if EN else 'Дослідницька платформа для аналізу нестаціонарних часових рядів, структурних змін, адаптивних стохастичних моделей та інтервальних прогнозів. Обчислення виконуються для вибраного ряду.')
        st.info(('Dataset: ' + upload.name) if upload else ('Demo dataset: ENTSO-E · Italy · 2016–2025' if EN else 'Демонстраційний набір: ENTSO-E · Італія · 2016–2025'))
        st.caption(('Aggregation: ' if EN else 'Агрегування: ')+str(frequency)+' · '+str(value_col))

elif page=='Дані та підготовка':
    section(tr('Вхідні дані та попередня обробка'),'Автоматичний вибір полів; обробка пропусків, повторів і часової частоти.')
    a,b,c=st.columns(3)
    a.metric(tr('Рядків у файлі'),len(raw));b.metric('Коректних дат і значень',len(series));c.metric(tr('Частота'),'Початкова' if frequency=='Без агрегування' else frequency)
    st.plotly_chart(line_figure(series.index,series.values),use_container_width=True)
    st.subheader(tr('Таблиця спостережень'))
    result=series.rename('value').reset_index().rename(columns={'index':'date'})
    st.dataframe(result,use_container_width=True,hide_index=True,height=350)
    st.download_button('↓ Експортувати підготовлений ряд (CSV)',csv_bytes(result),'prepared_timeseries.csv',mime='text/csv')
    with st.expander(tr('Контроль вихідного файлу')):
        st.write('Назви полів:',cols)
        st.write('Пропущені значення за стовпцями:')
        st.dataframe(raw.isna().sum().rename('Пропуски'))
        st.caption('Повторні часові позначки усереднюються; при агрегуванні обчислюється середнє за обраний період. Пропущені значення не заповнюються вигаданими даними.')

elif page=='Статистичний аналіз':
    section(tr('Описова статистика та автокореляція'))
    a,b,c=st.columns(3)
    a.metric(tr('Стандартне відхилення'),f'{series.std():,.2f}')
    b.metric(tr('Медіана'),f'{series.median():,.2f}')
    c.metric(tr('Коефіцієнт варіації'),f'{series.std()/abs(series.mean())*100:.2f}%' if series.mean()!=0 else '—')
    st.dataframe(series.describe().rename('Значення').to_frame(),use_container_width=True)
    lag=st.slider(tr('Кількість лагів'),3,min(90,len(series)//2-2),min(35,len(series)//3))
    left,right=st.columns(2)
    with left:
        fig=go.Figure(go.Bar(x=list(range(lag+1)),y=acf(series,nlags=lag,fft=True),marker_color=TEAL,name='ACF'))
        fig.update_layout(title='Автокореляційна функція')
        st.plotly_chart(plot_style(fig,340,'Кореляція'),use_container_width=True)
    with right:
        fig=go.Figure(go.Bar(x=list(range(lag+1)),y=pacf(series,nlags=lag,method='ywm'),marker_color='#3F638E',name='PACF'))
        fig.update_layout(title='Часткова автокореляція')
        st.plotly_chart(plot_style(fig,340,'Кореляція'),use_container_width=True)
    window=st.slider(tr('Вікно ковзних статистик'),3,min(90,len(series)//2),min(7,len(series)//2))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=series.index,y=series,name='Ряд',line=dict(color='#B8C7D7',width=1)))
    fig.add_trace(go.Scatter(x=series.index,y=series.rolling(window).mean(),name='Ковзне середнє',line=dict(color=TEAL,width=2)))
    st.plotly_chart(plot_style(fig),use_container_width=True)

elif page=='Стаціонарність':
    section(tr('Перевірка стаціонарності'),'ADF: нульова гіпотеза — одиничний корінь; KPSS: нульова гіпотеза — стаціонарність за рівнем.')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adf=adfuller(series,autolag='AIC')
            kp=kpss(series,regression='c',nlags='auto')
        table=pd.DataFrame([{'Критерій':'ADF','Статистика':adf[0],'p-value':f'{adf[1]:.3g}','Нульова гіпотеза':'Одиничний корінь'},{'Критерій':'KPSS','Статистика':kp[0],'p-value':f'< {kp[1]:.2g}' if any('smaller than' in str(w.message).lower() for w in caught) else f'{kp[1]:.3g}','Нульова гіпотеза':'Стаціонарність за рівнем'}])
        st.dataframe(table.rename(columns={k:tr(k) for k in table.columns}),hide_index=True,use_container_width=True)
        st.info('ADF and KPSS test different null hypotheses. Conflicting results do not identify the cause of nonstationarity.' if EN else 'ADF та KPSS перевіряють різні нульові гіпотези. Суперечливі результати не встановлюють причини нестаціонарності.')
        if any('interpolation' in str(w.message).lower() for w in caught):st.caption('KPSS: the reported p-value is a table bound, not an exact probability.' if EN else 'KPSS: наведене p-value є межею табличного діапазону, а не точним значенням.')
        st.download_button(tr('↓ Експортувати результати тестів'),csv_bytes(table),'stationarity_tests.csv')
    except Exception as exc:st.warning(f'Тести неможливо виконати для цього ряду: {exc}')

elif page=='Декомпозиція STL':
    section(tr('Сезонно-трендова декомпозиція STL'))
    default_period=min(7,max(2,len(series)//3))
    period=st.number_input('Сезонний період (спостережень)',min_value=2,max_value=max(2,min(365,len(series)//2-1)),value=default_period)
    if len(series)<=2*period:st.warning('Недостатньо спостережень для обраного сезонного періоду.')
    else:
        fit=STL(series,period=int(period),robust=True).fit()
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.055,subplot_titles=['Вихідний ряд','Тренд','Сезонність','Залишок'])
        for i,(v,c) in enumerate([(series,NAVY),(fit.trend,TEAL),(fit.seasonal,'#6B65B7'),(fit.resid,'#D08451')],1):
            fig.add_trace(go.Scatter(x=series.index,y=v,mode='lines',line=dict(color=c,width=1.4),showlegend=False),row=i,col=1)
        fig.update_layout(height=760,margin=dict(l=10,r=10,t=50,b=15),paper_bgcolor='white',plot_bgcolor='white')
        fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor='#E9EEF5')
        st.plotly_chart(fig,use_container_width=True)
        components=pd.DataFrame({'date':series.index,'observed':series.values,'trend':fit.trend.values,'seasonal':fit.seasonal.values,'residual':fit.resid.values})
        st.download_button(tr('↓ Завантажити компоненти STL'),csv_bytes(components),'stl_components.csv')

elif page=='Структурні зміни PELT':
    section(tr('Виявлення структурних змін методом PELT'),'Алгоритм ruptures.Pelt, квадратична вартість сегмента (l2); кількість змін залежить від штрафу.')
    mode=st.selectbox('Агрегування для сегментації',['Місяць','Тиждень','Поточний ряд'])
    seg=series.resample('MS').mean().dropna() if mode=='Місяць' else series.resample('W').mean().dropna() if mode=='Тиждень' else series
    if len(seg)<12:st.warning('Для сегментації потрібно щонайменше 12 спостережень.');st.stop()
    a,b=st.columns(2)
    with a:min_size=st.number_input(tr('Мінімальний розмір сегмента'),min_value=2,max_value=max(2,len(seg)//3),value=min(6,max(2,len(seg)//6)))
    with b:multiplier=st.slider(tr('Множник штрафу'),.1,20.,3.,.1)
    st.caption('Штраф = множник × дисперсія × ln(n). Це налаштування для експерименту, а не універсально оптимальне значення.')
    if st.button(tr('Обчислити точки зміни'),type='primary'):
        try:
            import ruptures as rpt
            penalty=float(multiplier*np.var(seg.values)*np.log(len(seg)))
            stops=rpt.Pelt(model='l2',min_size=int(min_size)).fit(seg.values).predict(pen=penalty)
            fig=line_figure(seg.index,seg.values,'Агрегований ряд')
            rows=[];start=0
            for stop in stops:
                if stop>start:rows.append({'Початок':seg.index[start].date(),'Кінець':seg.index[stop-1].date(),'Середнє':float(seg.iloc[start:stop].mean()),'Спостережень':stop-start})
                if stop<len(seg):fig.add_vline(x=seg.index[stop].timestamp()*1000,line_color='#D97757',line_dash='dash')
                start=stop
            st.metric(tr('Виявлених точок зміни'),len(stops)-1)
            st.plotly_chart(fig,use_container_width=True)
            table=pd.DataFrame(rows)
            st.dataframe(table,hide_index=True,use_container_width=True)
            st.download_button(tr('↓ Експортувати сегменти'),csv_bytes(table),'pelt_segments.csv')
            st.info('Точки зміни є статистичними межами сегментів, а не доказом конкретних економічних чи соціальних причин.')
        except Exception as exc:st.error(f'Не вдалося виконати PELT: {exc}')

elif page=='Прогнозування':
    section(tr('Побудова прогнозів'), 'Model estimates and prediction intervals are computed from the selected series.' if EN else 'Прогнозні значення та інтервали обчислюються для вибраного часового ряду.')
    model=st.selectbox(tr('Прогнозна модель'),['SARIMA','State-Space / Kalman'])
    c1,c2,c3=st.columns(3)
    with c1:horizon=st.number_input(tr('Горизонт прогнозу'),min_value=1,max_value=90,value=7)
    with c2:seasonal=st.number_input(tr('Сезонний період'),min_value=2,max_value=52,value=7)
    with c3:confidence=st.selectbox(tr('Рівень прогнозного інтервалу'),[90,95,99],index=1)
    order=(1,0,0); seasonal_order=(1,1,0,int(seasonal))
    if model=='SARIMA':
        mode=st.radio(tr('Режим SARIMA'),['Автоматичний підбір (AIC)','Ручні параметри SARIMA'],horizontal=True)
        if mode=='Ручні параметри SARIMA':
            st.markdown('**'+tr('Параметри несезонної частини')+'**')
            a,b,c=st.columns(3)
            with a:p=st.number_input('p',0,3,1)
            with b:d=st.number_input('d',0,2,0)
            with c:q=st.number_input('q',0,3,0)
            st.markdown('**'+tr('Параметри сезонної частини')+'**')
            a,b,c=st.columns(3)
            with a:P=st.number_input('P',0,2,1)
            with b:D=st.number_input('D',0,1,1)
            with c:Q=st.number_input('Q',0,2,0)
            order=(int(p),int(d),int(q));seasonal_order=(int(P),int(D),int(Q),int(seasonal))
        else:
            st.caption('Bounded AIC search over candidate orders; not a guarantee of optimal forecast accuracy.' if EN else 'Обмежений пошук за AIC серед кандидатів; не гарантує найкращої прогнозної точності.')
    max_train=min(1200,len(series));min_train=min(45,max_train)
    train_size=st.slider(tr('Кількість останніх спостережень для навчання'),min_train,max_train,min(731,max_train))
    if st.button(tr('Побудувати прогноз'),type='primary'):
        try:
            y=series.iloc[-train_size:]
            if len(y)<max(45,3*seasonal):raise ValueError('Not enough observations for this seasonal period.' if EN else 'Замало даних для обраного сезонного періоду.')
            ix=date_index(y,int(horizon))
            with st.spinner('Fitting the model…' if EN else 'Оцінювання параметрів моделі...'):
                if model=='SARIMA':
                    if mode=='Автоматичний підбір (AIC)':
                        candidates=[]
                        for od in [(0,1,1),(1,0,0),(1,1,0),(1,1,1),(2,1,0),(0,1,2)]:
                            for sod in [(0,1,1,int(seasonal)),(1,1,0,int(seasonal)),(0,0,1,int(seasonal))]:
                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter('ignore')
                                        f=SARIMAX(y,order=od,seasonal_order=sod,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=55)
                                    if np.isfinite(f.aic) and f.mle_retvals.get('converged',False):candidates.append((float(f.aic),od,sod,f))
                                except Exception:pass
                        if not candidates:raise ValueError('No candidate model converged.' if EN else 'Жодна модель-кандидат не збіглася.')
                        _,order,seasonal_order,fit=min(candidates,key=lambda x:x[0])
                    else:
                        fit=SARIMAX(y,order=order,seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=100)
                    st.info(f'SARIMA{order} × {seasonal_order} · AIC = {fit.aic:.2f}')
                else:fit=UnobservedComponents(y,level='local level',seasonal=int(seasonal)).fit(disp=False,maxiter=100)
                pred=fit.get_forecast(steps=int(horizon));ci=pred.conf_int(alpha=1-confidence/100)
            result=pd.DataFrame({'date':ix,'forecast':np.asarray(pred.predicted_mean),f'lower_{confidence}':ci.iloc[:,0].to_numpy(),f'upper_{confidence}':ci.iloc[:,1].to_numpy()})
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=y.index[-120:],y=y.iloc[-120:],name='Observed' if EN else 'Фактичні дані',line=dict(color=NAVY,width=2)))
            fig.add_trace(go.Scatter(x=result.date,y=result.iloc[:,-1],mode='lines',line=dict(width=0),showlegend=False))
            fig.add_trace(go.Scatter(x=result.date,y=result.iloc[:,-2],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(8,127,140,.15)',name=f'{confidence}% interval'))
            fig.add_trace(go.Scatter(x=result.date,y=result.forecast,name='Forecast' if EN else 'Прогноз',line=dict(color=TEAL,width=3)))
            st.plotly_chart(plot_style(fig,490),use_container_width=True)
            st.dataframe(result,hide_index=True,use_container_width=True)
            st.download_button(tr('↓ Експортувати прогноз'),csv_bytes(result),'forecast.csv')
            if hasattr(fit,'mle_retvals') and not fit.mle_retvals.get('converged',True):st.warning('Optimization did not converge.' if EN else 'Оптимізатор не підтвердив збіжність.')
        except Exception as exc:st.error(f'Forecast failed: {exc}' if EN else f'Прогноз не побудовано: {exc}')

elif page=='Адаптивна валідація':
    adaptive_page(series,EN)

elif page=='Порівняння моделей':
    section(tr('Порівняння прогнозних моделей'), 'Historical ENTSO-E benchmark (separate from current dataset)' if EN else 'Архівний експеримент ENTSO-E (окремо від поточного набору даних)')
    st.info('Saved ENTSO-E 2025 benchmark. Uploading a different dataset does not recompute these historical benchmarks.' if EN else 'Збережений експеримент ENTSO-E за 2025 рік. Для іншого набору даних ці показники не підставляються як нові результати.')
    if upload:st.warning('These benchmarks refer to ENTSO-E only. Use Forecasting to fit your uploaded series.' if EN else 'Наведені нижче показники стосуються лише ENTSO-E. Для завантаженого ряду використайте розділ «Прогнозування».')
    if upload:
        st.info('Open Adaptive validation to calculate a benchmark on your uploaded data.' if EN else 'Для порівняння моделей на власному наборі відкрийте «Адаптивна валідація».')
        st.stop()
    try:
        val,metrics=load_validation()
        mapping={'Seasonal naive':'Сезонна наївна','SARIMA(1,0,0)(1,1,0,7)':'SARIMA','State-space local level seasonal 7':'State-Space/Kalman'}
        table=metrics.copy();table['model']=table['model'].replace(mapping)
        st.dataframe(table.rename(columns={'model':'Модель','MAE_MW':'MAE, МВт','RMSE_MW':'RMSE, МВт','bias_MW':'Зміщення, МВт','PICP_95_pct':'PICP, %','MPIW_95_MW':'MPIW, МВт'}),hide_index=True,use_container_width=True)
        fig=go.Figure()
        for key,label,color in [('actual_MW','Фактичні значення',NAVY),('seasonal_naive_MW','Наївний прогноз','#B1BFCF'),('sarima_MW','SARIMA','#6B65B7'),('state_space_MW','State-Space',TEAL)]:
            if key in val:fig.add_trace(go.Scatter(x=val.date,y=val[key],name=label,line=dict(width=1.5,color=color)))
        st.plotly_chart(plot_style(fig,490),use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure()
            for key,label,color in [('MAE_MW','MAE',TEAL),('RMSE_MW','RMSE','#6B65B7')]:fig.add_trace(go.Bar(x=table['model'],y=table[key],name=label,marker_color=color))
            st.plotly_chart(plot_style(fig,360,'МВт'),use_container_width=True)
        with c2:
            coverage=table.dropna(subset=['PICP_95_pct'])
            fig=go.Figure(go.Bar(x=coverage['model'],y=coverage['PICP_95_pct'],marker_color=TEAL,name='PICP'))
            fig.add_hline(y=95,line_dash='dash',line_color='#D97757',annotation_text='Номінальні 95%')
            st.plotly_chart(plot_style(fig,360,'Покриття, %'),use_container_width=True)
        st.download_button(tr('↓ Таблиця результатів'),csv_bytes(metrics),'italy_2025_rolling_metrics.csv')
        st.download_button(tr('↓ 365 прогнозів за 2025 рік'),csv_bytes(val),'italy_2025_rolling_forecasts.csv')
        st.caption('Model differences in this experiment do not establish the effectiveness of PELT-based adaptation; separate validation is required.' if EN else 'Відмінність моделей у цьому експерименті не доводить ефективності адаптації за PELT; вона потребує окремої перевірки.')
    except Exception as exc:st.error(f'Неможливо відкрити результати: {exc}')

elif page=='Методологія та відтворюваність':
    section('Methodology & reproducibility' if EN else 'Методологія та відтворюваність')
    st.info('Research prototype: verify results before drawing scientific conclusions.' if EN else 'Дослідницький прототип: перед науковими висновками перевірте результати та припущення.')
    protocol=pd.DataFrame([
      {'Field':'Source','Value':upload.name if upload else 'ENTSO-E Italy 2016–2025'},
      {'Field':'Date column','Value':str(date_col)},
      {'Field':'Value column','Value':str(value_col)},
      {'Field':'Aggregation','Value':str(frequency)},
      {'Field':'Observations','Value':str(len(series))},
      {'Field':'Start','Value':str(series.index.min())},
      {'Field':'End','Value':str(series.index.max())},
    ])
    st.dataframe(protocol,hide_index=True,use_container_width=True)
    st.markdown('### '+('Validation protocol' if EN else 'Протокол валідації'))
    st.markdown(('**Walk-forward:** all strategies use the same forecast origins and only past data for fitting. **PELT:** changes in first differences are candidates for refitting, not proof of causal events. **Intervals:** nominal Gaussian bounds and optional calibration using past out-of-sample errors. **Limits:** AIC does not optimize test accuracy; short samples produce unstable PICP.' if EN else '**Walk-forward:** усі стратегії оцінюються на однакових датах; під час навчання використовуються лише минулі спостереження. **PELT:** зміни в перших різницях є сигналами для перенавчання, але не доводять причинних подій. **Інтервали:** гауссові межі та калібрування за минулими позавибірковими помилками. **Обмеження:** AIC не оптимізує тестову точність; короткі вибірки дають нестійку оцінку PICP.'))
    st.markdown('### '+('Scientific references' if EN else 'Наукові джерела'))
    st.markdown('''- Dickey & Fuller (1979), JASA; Kwiatkowski et al. (1992), Journal of Econometrics.
- Cleveland et al. (1990), Journal of Official Statistics; Killick et al. (2012), JASA.
- Durbin & Koopman (2012), Time Series Analysis by State Space Methods; Gneiting & Raftery (2007), JASA.''')
    st.download_button('Download protocol' if EN else 'Завантажити протокол',protocol.to_csv(index=False).encode('utf-8-sig'),'research_protocol.csv',mime='text/csv')

st.divider()
st.caption('Research application · Results depend on the dataset, parameters and validation method.' if EN else 'Дослідницький застосунок · Результати залежать від даних, параметрів і методики валідації.')


