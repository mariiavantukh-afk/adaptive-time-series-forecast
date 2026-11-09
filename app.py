from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

st.set_page_config(page_title='Адаптивне прогнозування часових рядів',page_icon='📈',layout='wide')
BASE=Path(__file__).parent
st.title('📈 Адаптивне прогнозування часових рядів')
st.caption('Дослідницький застосунок • ENTSO-E, Італія, 2016–2025 • Python / Streamlit')

@st.cache_data
def demo_data():
    return pd.read_csv(BASE/'data/italy_daily_load_2016_2025.csv',parse_dates=['date'])

@st.cache_data
def validation_data():
    return pd.read_csv(BASE/'data/italy_2025_rolling_forecasts.csv',parse_dates=['date'])

@st.cache_data
def metrics_data():
    return pd.read_csv(BASE/'data/italy_2025_rolling_metrics.csv')

def line_chart(df,cols,title):
    fig=go.Figure()
    for col in cols: fig.add_trace(go.Scatter(x=df.index,y=df[col],name=col,mode='lines'))
    fig.update_layout(title=title,xaxis_title='Дата',yaxis_title='Значення',height=400,margin=dict(t=55,b=25))
    st.plotly_chart(fig,use_container_width=True)

def prepare(df,time_col,value_col,freq):
    out=pd.DataFrame({'date':pd.to_datetime(df[time_col],errors='coerce'), 'value':pd.to_numeric(df[value_col],errors='coerce')}).dropna()
    out=out.groupby('date',as_index=True)['value'].mean().sort_index()
    if freq!='Без агрегування':out=out.resample({'Доба':'D','Тиждень':'W','Місяць':'MS'}[freq]).mean()
    return out.dropna()

st.sidebar.header('Вхідні дані')
upload=st.sidebar.file_uploader('CSV або Excel',type=['csv','xlsx'])
try:
    if upload:
        if upload.name.lower().endswith('.csv'):
            raw=pd.read_csv(upload)
        else:raw=pd.read_excel(upload)
        st.sidebar.success('Файл завантажено')
    else:
        raw=demo_data()
        st.sidebar.info('Демонстраційні дані ENTSO-E: 2016–2025')
    date_col=st.sidebar.selectbox('Стовпець дати',list(raw.columns),index=list(raw.columns).index('date') if 'date' in raw else 0)
    candidates=[c for c in raw.columns if c!=date_col]
    default='mean_load_MW' if 'mean_load_MW' in candidates else candidates[0]
    val_col=st.sidebar.selectbox('Числова змінна',candidates,index=candidates.index(default))
    freq=st.sidebar.selectbox('Частота', ['Без агрегування','Доба','Тиждень','Місяць'],index=0)
    s=prepare(raw,date_col,val_col,freq)
    if len(s)<15:st.error('Потрібно щонайменше 15 коректних спостережень.');st.stop()
except Exception as e:
    st.error(f'Не вдалося прочитати дані: {e}');st.stop()

tabs=st.tabs(['Огляд','Статистика','Стаціонарність','STL','Точки зміни','Прогнозування','Перевірка 2025'])
with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric('Спостережень',f'{len(s):,}'.replace(',',' '));b.metric('Середнє',f'{s.mean():,.1f}');c.metric('Мінімум',f'{s.min():,.1f}');d.metric('Максимум',f'{s.max():,.1f}')
    line_chart(s.to_frame('Значення'),['Значення'],'Вхідний часовий ряд')
    st.dataframe(s.rename('Значення').to_frame().tail(20),use_container_width=True)
with tabs[1]:
    st.dataframe(s.describe().rename('Показник').to_frame(),use_container_width=True)
    nlag=min(40,len(s)//3)
    a,b=st.columns(2)
    with a:
        v=acf(s,nlags=nlag,fft=True)
        st.bar_chart(pd.DataFrame({'ACF':v},index=range(len(v))))
    with b:
        v=pacf(s,nlags=min(nlag,len(s)//2-1),method='ywm')
        st.bar_chart(pd.DataFrame({'PACF':v},index=range(len(v))))
    st.line_chart(pd.DataFrame({'Ряд':s,'Ковзне середнє (7)':s.rolling(7).mean(),'Ковзне σ (7)':s.rolling(7).std()}))
with tabs[2]:
    try:
        adf=adfuller(s,autolag='AIC')
        kp=kpss(s,regression='c',nlags='auto')
        st.dataframe(pd.DataFrame([{'Тест':'ADF','Статистика':adf[0],'p-value':adf[1],'Нульова гіпотеза':'Є одиничний корінь'},{'Тест':'KPSS','Статистика':kp[0],'p-value':kp[1],'Нульова гіпотеза':'Ряд стаціонарний за рівнем'}]),hide_index=True,use_container_width=True)
        st.info('ADF і KPSS мають різні нульові гіпотези. Значення p для KPSS можуть бути обмежені табличними критичними значеннями.')
    except Exception as e:st.warning(f'Тести недоступні для цього ряду: {e}')
with tabs[3]:
    period=st.number_input('Сезонний період (спостережень)',min_value=2,max_value=max(2,min(365,len(s)//2)),value=min(7,max(2,len(s)//2)))
    if len(s)>2*period:
        fit=STL(s,period=int(period),robust=True).fit()
        fig=go.Figure()
        for name,v in [('Спостереження',s),('Тренд',fit.trend),('Сезонність',fit.seasonal),('Залишки',fit.resid)]:fig.add_trace(go.Scatter(x=v.index,y=v,name=name,visible=True if name in ('Спостереження','Тренд') else 'legendonly'))
        fig.update_layout(height=450,title='STL: компоненти (перемикайте в легенді)')
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Завантажити компоненти STL',pd.DataFrame({'date':s.index,'trend':fit.trend.values,'seasonal':fit.seasonal.values,'residual':fit.resid.values}).to_csv(index=False).encode('utf-8-sig'),'stl_components.csv')
    else:st.warning('Замало спостережень для заданого періоду.')
with tabs[4]:
    st.write('PELT із квадратичною функцією втрат для змін середнього рівня. Результат залежить від штрафу та масштабу даних.')
    aggregation=st.selectbox('Частота для сегментації',['Місяць','Тиждень','Без агрегування'])
    seg=s.resample('MS').mean().dropna() if aggregation=='Місяць' else s.resample('W').mean().dropna() if aggregation=='Тиждень' else s
    min_size=st.number_input('Мінімальний розмір сегмента',min_value=2,max_value=max(2,len(seg)//2),value=min(6,max(2,len(seg)//4)))
    scale=st.slider('Множник штрафу',0.1,20.0,3.0,0.1)
    if st.button('Виконати PELT',type='primary'):
        try:
            import ruptures as rpt
            penalty=float(scale*np.var(seg.to_numpy())*np.log(len(seg)))
            bkps=rpt.Pelt(model='l2',min_size=int(min_size)).fit(seg.to_numpy()).predict(pen=penalty)
            fig=go.Figure(go.Scatter(x=seg.index,y=seg.values,name='Ряд'))
            rows=[];start=0
            for stop in bkps:
                rows.append({'Початок':str(seg.index[start].date()),'Кінець':str(seg.index[stop-1].date()),'Середнє':float(seg.iloc[start:stop].mean()),'Кількість':stop-start})
                if stop<len(seg):fig.add_vline(x=seg.index[stop].timestamp()*1000,line_dash='dash',line_color='red')
                start=stop
            fig.update_layout(title=f'PELT: {len(bkps)-1} точок зміни; штраф {penalty:,.2f}')
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.download_button('Завантажити сегменти',pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'),'pelt_segments.csv')
        except Exception as e:st.error(f'Помилка PELT: {e}')
with tabs[5]:
    st.write('Прогноз на задану кількість кроків після останнього спостереження. Обчислення моделі може тривати кілька хвилин.')
    model=st.selectbox('Модель',['SARIMA(1,0,0)(1,1,0,7)','State-Space / Kalman'])
    horizon=st.slider('Горизонт (кроків)',1,60,7)
    train_window=st.slider('Кількість останніх спостережень для навчання',min_value=min(60,len(s)),max_value=len(s),value=min(731,len(s)),step=1)
    if st.button('Побудувати прогноз',type='primary'):
        if train_window<30:st.warning('Для сезонної моделі потрібно більше спостережень.')
        else:
            with st.spinner('Оцінювання параметрів моделі...'):
                try:
                    y=s.iloc[-train_window:]
                    if model.startswith('SARIMA'):
                        fitted=SARIMAX(y,order=(1,0,0),seasonal_order=(1,1,0,7),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=80)
                    else:fitted=UnobservedComponents(y,level='local level',seasonal=7).fit(disp=False,maxiter=100)
                    pred=fitted.get_forecast(steps=horizon)
                    ci=pred.conf_int(alpha=.05)
                    index=pd.date_range(y.index[-1],periods=horizon+1,freq=pd.infer_freq(y.index) or 'D')[1:]
                    result=pd.DataFrame({'date':index,'forecast':np.asarray(pred.predicted_mean),'lower_95':ci.iloc[:,0].to_numpy(),'upper_95':ci.iloc[:,1].to_numpy()})
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=y.index[-min(120,len(y)):],y=y.iloc[-min(120,len(y)):],name='Історія'))
                    fig.add_trace(go.Scatter(x=result.date,y=result.upper_95,line=dict(width=0),showlegend=False))
                    fig.add_trace(go.Scatter(x=result.date,y=result.lower_95,fill='tonexty',fillcolor='rgba(99,102,241,0.2)',line=dict(width=0),name='95% інтервал'))
                    fig.add_trace(go.Scatter(x=result.date,y=result.forecast,name='Прогноз'))
                    st.plotly_chart(fig,use_container_width=True)
                    st.dataframe(result,use_container_width=True,hide_index=True)
                    st.download_button('Завантажити прогноз CSV',result.to_csv(index=False).encode('utf-8-sig'),'forecast.csv')
                except Exception as e:st.error(f'Не вдалося оцінити модель: {e}')
with tabs[6]:
    st.write('Збережені результати попереднього експерименту rolling-origin на фактичних спостереженнях 2025 року. Вони не перераховуються при завантаженні іншого набору даних.')
    try:
        val=validation_data(); met=metrics_data()
        st.dataframe(met,hide_index=True,use_container_width=True)
        fig=go.Figure()
        for col in ['actual_MW','seasonal_naive_MW','sarima_MW','state_space_MW']:
            if col in val:fig.add_trace(go.Scatter(x=val.date,y=val[col],name=col))
        fig.update_layout(title='Фактичні значення та однодобові прогнози, 2025',height=450)
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Завантажити прогнози за 2025',val.to_csv(index=False).encode('utf-8-sig'),'italy_2025_validation.csv')
    except Exception as e:st.warning(f'Не вдалося відкрити результати перевірки: {e}')

st.caption('Навчально-дослідницький інструмент. Висновки залежать від якості вхідних даних, налаштувань і процедури валідації.')
