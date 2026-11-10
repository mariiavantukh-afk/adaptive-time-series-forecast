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

st.set_page_config(page_title='Time Series Analytics | Research application', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
NAV = ['Огляд дослідження', 'Дані та підготовка', 'Статистичний аналіз', 'Стаціонарність', 'Декомпозиція STL', 'Структурні зміни PELT', 'Прогнозування', 'Адаптивна валідація', 'Порівняння моделей']
TEAL, NAVY, MUTED = '#087F8C', '#172B4D', '#60748B'

st.markdown('''<style>
.block-container {padding-top:1.3rem; padding-bottom:2.2rem; max-width:1480px}
[data-testid="stSidebar"] {background: #EDF2F8; border-right:1px solid #DAE3ED}
[data-testid="stSidebar"] .stRadio label {font-size:0.93rem}
[data-testid="stMetric"] {background:white; border:1px solid #DCE5EF; border-radius:13px; padding:15px 18px; box-shadow:0 3px 12px rgba(23,43,77,.035)}
[data-testid="stMetricLabel"] {color:#60748B}
[data-testid="stMetricValue"] {color:#172B4D; font-size:1.65rem}
h1,h2,h3 {color:#172B4D;letter-spacing:-.025em}
.hero {background:linear-gradient(112deg,#172B4D,#244A68);padding:28px 30px;border-radius:17px;color:white;margin-bottom:20px}
.hero h1 {color:white!important;font-size:2rem;margin:0 0 7px 0}
.hero p {color:#D5E7EF;margin:0;font-size:0.95rem}
.eyebrow {color:#087F8C;font-size:.8rem;letter-spacing:.10em;text-transform:uppercase;font-weight:700;margin-bottom:4px}
.note {background:#E8F4F5;padding:13px 17px;border-left:4px solid #087F8C;border-radius:8px;margin:12px 0;color:#1F4854}
.small {color:#60748B;font-size:.88rem}
[data-testid="stPlotlyChart"] {background:white;border:1px solid #E1E8F0;border-radius:13px;padding:7px}
.stButton>button[kind="primary"] {background:#087F8C;border-color:#087F8C}
</style>''', unsafe_allow_html=True)

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
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=34,b=20),paper_bgcolor='white',plot_bgcolor='white',font=dict(color=NAVY),legend=dict(orientation='h',y=1.12,x=0),hovermode='x unified',xaxis=dict(showgrid=False,title='Дата'),yaxis=dict(title=yaxis,gridcolor='#E9EEF5'))
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
TRANSLATIONS = {'Огляд дослідження': 'Research overview', 'Дані та підготовка': 'Data preparation', 'Статистичний аналіз': 'Statistical analysis', 'Стаціонарність': 'Stationarity', 'Декомпозиція STL': 'STL decomposition', 'Структурні зміни PELT': 'PELT change points', 'Прогнозування': 'Forecasting', 'Адаптивна валідація': 'Adaptive validation', 'Порівняння моделей': 'Model comparison', 'Навігація': 'Navigation', 'Джерело даних': 'Data source', 'Завантажити CSV або XLSX': 'Upload CSV or XLSX', 'Стовпець дати': 'Date column', 'Числова змінна': 'Numeric variable', 'Агрегування': 'Aggregation', 'Без агрегування': 'No aggregation', 'Доба': 'Daily', 'Тиждень': 'Weekly', 'Місяць': 'Monthly', 'Коректних спостережень': 'Valid observations', 'Сезонний період': 'Seasonal period', 'Горизонт прогнозу': 'Forecast horizon', 'Прогнозна модель': 'Forecast model', 'Кількість останніх спостережень для навчання': 'Training observations', 'Побудувати прогноз': 'Generate forecast', 'Обчислити точки зміни': 'Detect change points', 'Ручні параметри SARIMA': 'Manual SARIMA parameters', 'Автоматичний підбір (AIC)': 'Automatic selection (AIC)', 'Режим SARIMA': 'SARIMA mode', 'Рівень прогнозного інтервалу': 'Prediction interval confidence', 'Параметри несезонної частини': 'Nonseasonal parameters', 'Параметри сезонної частини': 'Seasonal parameters', 'Побудова прогнозів': 'Forecast generation', 'Прогнозні значення та 95% інтервали обчислюються на обраному часовому ряді.': 'Point forecasts and prediction intervals are calculated for the selected time series.', 'Порівняння прогнозних моделей': 'Forecast model comparison', 'Збережені результати окремого експерименту rolling-origin за 2025 рік.': 'Saved results from the 2025 rolling-origin experiment.', 'Перевірка стаціонарності': 'Stationarity testing', 'Сезонно-трендова декомпозиція STL': 'Seasonal-trend decomposition (STL)', 'Виявлення структурних змін методом PELT': 'Change-point detection (PELT)', 'Описова статистика та автокореляція': 'Descriptive statistics and autocorrelation', 'Вхідні дані та попередня обробка': 'Input data and preprocessing', 'Спостережень': 'Observations', 'Середнє': 'Mean', 'Мінімум': 'Minimum', 'Максимум': 'Maximum', 'Динаміка досліджуваного показника': 'Time series overview', 'Про дослідження': 'About the research', 'Джерело': 'Source', 'Медіана': 'Median', 'Стандартне відхилення': 'Standard deviation', 'Коефіцієнт варіації': 'Coefficient of variation', 'Кількість лагів': 'Number of lags', 'Вікно ковзних статистик': 'Rolling window', 'Вихідний ряд': 'Observed', 'Тренд': 'Trend', 'Сезонність': 'Seasonality', 'Залишок': 'Residual', 'Поточний ряд': 'Current series', 'Мінімальний розмір сегмента': 'Minimum segment size', 'Множник штрафу': 'Penalty multiplier', 'Виявлених точок зміни': 'Detected change points', 'Фактичні дані': 'Observed data', 'Прогноз': 'Forecast', 'Фактичні значення': 'Observed values', 'Модель': 'Model', 'Одиничний корінь': 'Unit root', 'Стаціонарність за рівнем': 'Level stationarity', 'Статистика': 'Statistic', 'Критерій': 'Test', 'Нульова гіпотеза': 'Null hypothesis', '↓ Експортувати прогноз': '↓ Export forecast', '↓ Експортувати результати тестів': '↓ Export test results', '↓ Завантажити компоненти STL': '↓ Download STL components', '↓ Експортувати сегменти': '↓ Export segments', '↓ Таблиця результатів': '↓ Export metrics', '↓ 365 прогнозів за 2025 рік': '↓ Export 2025 forecasts', 'Рядків у файлі': 'File rows', 'Частота': 'Frequency', 'Таблиця спостережень': 'Observations table', 'Контроль вихідного файлу': 'Input file quality', 'Завантажити': 'Download'}
def tr(value):
    return TRANSLATIONS.get(value, value) if EN else value

st.sidebar.markdown('### ◈ Time Series Analytics')
st.sidebar.caption('Research system · Python / Streamlit' if EN else 'Дослідницька система · Python / Streamlit')
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
st.sidebar.caption((f'Valid observations: {len(series):,}' if EN else f'Коректних спостережень: {len(series):,}').replace(',',' '))
if len(series)<15:
    st.error('Потрібно щонайменше 15 коректних спостережень. Перевірте вибір стовпців дати й числової змінної.');st.stop()

if page=='Огляд дослідження':
    st.markdown('<div class="hero"><h1>Time Series Analytics & Forecasting</h1><p>Adaptive stochastic modelling · uncertainty quantification · prediction intervals</p></div>' if EN else '<div class="hero"><h1>Аналітика та прогнозування часових рядів</h1><p>Адаптивне стохастичне моделювання · оцінювання невизначеності · інтервальні прогнози</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric(tr('Спостережень'),f'{len(series):,}'.replace(',',' '))
    b.metric(tr('Середнє'),f'{series.mean():,.0f}'.replace(',',' '))
    c.metric(tr('Мінімум'),f'{series.min():,.0f}'.replace(',',' '))
    d.metric(tr('Максимум'),f'{series.max():,.0f}'.replace(',',' '))
    st.subheader(tr('Динаміка досліджуваного показника'))
    st.plotly_chart(line_figure(series.index,series.values),use_container_width=True)
    a,b=st.columns([2,1])
    with a:
        st.markdown('#### About the research' if EN else '#### Про дослідження')
        st.write('Analyze time series, test stationarity, perform STL decomposition and PELT change-point detection, fit SARIMA and state-space models, and quantify forecast uncertainty.' if EN else 'Система призначена для аналізу часових рядів, перевірки стаціонарності, STL-декомпозиції, пошуку структурних змін PELT, прогнозування SARIMA та State-Space/Kalman і оцінювання прогнозних інтервалів.')
    with b:
        st.markdown('#### Source' if EN else '#### Джерело')
        st.write('ENTSO-E · Italy · 2016–2025' if EN else 'ENTSO-E · Італія · 2016–2025') if not upload else st.write(f'Користувацький файл: {upload.name}')
        st.caption(f'Частота: {frequency}; вибрана змінна: {value_col}')

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
        table=pd.DataFrame([{'Критерій':'ADF','Статистика':adf[0],'p-value':adf[1],'Нульова гіпотеза':'Одиничний корінь'},{'Критерій':'KPSS','Статистика':kp[0],'p-value':f'< {kp[1]:.2g}' if any('smaller than' in str(w.message).lower() for w in caught) else kp[1],'Нульова гіпотеза':'Стаціонарність за рівнем'}])
        st.dataframe(table.rename(columns={k:tr(k) for k in table.columns}),hide_index=True,use_container_width=True)
        st.info('ADF and KPSS test different null hypotheses. Conflicting results do not identify the cause of nonstationarity.' if EN else 'ADF та KPSS перевіряють різні нульові гіпотези. Суперечливі результати не встановлюють причини нестаціонарності.')
        if caught:st.caption('KPSS: the reported p-value is a table bound, not an exact probability.' if EN else 'KPSS: наведене p-value є межею табличного діапазону, а не точним значенням.')
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
    section(tr('Порівняння прогнозних моделей'),'Збережені результати окремого експерименту rolling-origin за 2025 рік.')
    st.info('Saved ENTSO-E 2025 benchmark. Uploading a different dataset does not recompute these historical benchmarks.' if EN else 'Збережений експеримент ENTSO-E за 2025 рік. Для іншого набору даних ці показники не підставляються як нові результати.')
    if upload:st.warning('These benchmarks refer to ENTSO-E only. Use Forecasting to fit your uploaded series.' if EN else 'Наведені нижче показники стосуються лише ENTSO-E. Для завантаженого ряду використайте розділ «Прогнозування».')
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

st.divider()
st.caption('Research application · Results depend on the dataset, parameters and validation method.' if EN else 'Дослідницький застосунок · Результати залежать від даних, параметрів і методики валідації.')
