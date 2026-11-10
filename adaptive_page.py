import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from adaptive_engine import run_backtest


def adaptive_page(series, english=False):
    def t(ua,en):return en if english else ua
    st.markdown('### '+t('Адаптивне моделювання та перевірка прогнозів','Adaptive modelling and forecast validation'))
    st.caption(t('Статична модель · періодичне перенавчання · оновлення за PELT · walk-forward · інтервальна оцінка',
                 'Static model · scheduled refitting · PELT-triggered updates · walk-forward · interval evaluation'))
    st.info(t('Усі показники обчислюються на поточному часовому ряді. Навчання на кожному кроці використовує тільки минулі спостереження. Показники не завантажуються з історичних таблиць.',
              'All metrics are computed on the selected series. Each training step uses past observations only. No stored benchmark values are substituted.'))
    if len(series)<130:
        st.warning(t('Для адаптивного експерименту потрібно щонайменше 130 спостережень.','At least 130 observations are required for adaptive validation.'))
        return
    signature=(len(series),str(series.index[0]),str(series.index[-1]),round(float(series.sum()),3))
    if st.session_state.get('adaptive_signature')!=signature:
        st.session_state.pop('adaptive_result',None)
        st.session_state['adaptive_signature']=signature
    a,b,c=st.columns(3)
    with a:model=st.selectbox(t('Модель','Model'),['SARIMA','State-Space / Kalman'],key='ad_model')
    with b:seasonal=st.number_input(t('Сезонний період','Seasonal period'),min_value=2,max_value=30,value=7,key='ad_season')
    with c:horizon=st.number_input(t('Горизонт перевірки (кроки)','Validation horizon (steps)'),min_value=1,max_value=7,value=1,key='ad_horizon')
    a,b,c=st.columns(3)
    with a:origins=st.slider(t('Кількість тестових дат','Test origins'),4,20,8,key='ad_origins')
    with b:
        min_window=max(90,seasonal*4)
        max_window=min(800,len(series)-origins-horizon)
        if max_window<min_window:
            st.warning(t('Зменште сезонний період або кількість тестових дат.','Reduce the seasonal period or number of test origins.'))
            return
        window=st.slider(t('Навчальне вікно','Training window'),min_window,max_window,min(280,max_window),key='ad_window')
    with c:refit_every=st.slider(t('Періодичність перенавчання','Refit interval (origins)'),2,8,3,key='ad_refit')
    a,b=st.columns(2)
    with a:penalty=st.slider(t('Штраф PELT','PELT penalty multiplier'),1.0,12.0,3.0,0.5,key='ad_penalty')
    with b:calibration_size=st.slider(t('Мінімум минулих помилок для калібрування','Min. prior errors for calibration'),2,8,3,key='ad_cal')
    st.caption(t('PELT аналізує перші різниці навчального вікна; оновлення відбувається за нової близької до кінця вікна точки зміни. Для SARIMA використано базову специфікацію (1,0,0)(1,1,0,s).',
                 'PELT analyses first differences of training data; refitting is triggered by a newly detected recent change point. SARIMA uses baseline order (1,0,0)(1,1,0,s).'))
    if st.button(t('Запустити порівняння стратегій','Run strategy comparison'),type='primary',key='ad_run'):
        try:
            if len(series)<window+origins+horizon:raise ValueError(t('Недостатньо даних: зменште навчальне вікно або кількість тестових дат.','Not enough data: reduce the training window or test origins.'))
            with st.spinner(t('Навчання моделей і перевірка на історичних датах...','Fitting and backtesting models...')):
                detail,summary,events=run_backtest(series,model,seasonal,origins,horizon,window,refit_every,penalty,calibration_size=calibration_size)
            st.session_state['adaptive_result']=(detail,summary,events,english)
        except Exception as exc:st.error(str(exc))
    if 'adaptive_result' not in st.session_state:return
    detail,summary,events,_=st.session_state['adaptive_result']
    st.divider()
    st.subheader(t('Результати walk-forward validation','Walk-forward validation results'))
    st.dataframe(summary[['strategy','n','MAE','RMSE','refits']].round(3),use_container_width=True,hide_index=True)
    fig=go.Figure()
    colors={'Static':'#8899AA','Periodic':'#087F8C','PELT-triggered':'#7059B8'}
    for strategy,g in detail.groupby('strategy',sort=False):
        fig.add_trace(go.Scatter(x=g.date,y=g.forecast,name=strategy,mode='lines+markers',line=dict(color=colors[strategy])))
    g=detail[detail.strategy=='Static']
    fig.add_trace(go.Scatter(x=g.date,y=g.actual,name=t('Фактичні значення','Observed'),mode='lines+markers',line=dict(color='#172B4D',width=3)))
    fig.update_layout(height=420,margin=dict(l=10,r=10,t=20,b=15),paper_bgcolor='white',plot_bgcolor='white',hovermode='x unified',xaxis_title=t('Дата','Date'),yaxis_title=t('Значення','Value'))
    st.plotly_chart(fig,use_container_width=True)
    st.subheader(t('Оцінювання невизначеності','Uncertainty evaluation'))
    level=st.select_slider(t('Рівень інтервалу','Interval level'),options=[80,90,95,99],value=95,key='ad_level')
    calibrated=st.toggle(t('Калібровані інтервали за попередніми помилками','Calibrated intervals using prior errors'),value=False,key='ad_calibrated')
    prefix='calibrated' if calibrated else 'raw'
    columns=['strategy',f'{prefix}_PICP_{level}',f'{prefix}_MPIW_{level}',f'{prefix}_IS_{level}']
    display=summary[columns].copy()
    display.columns=['strategy',f'PICP {level}%','MPIW','Interval Score']
    st.dataframe(display.round(4),hide_index=True,use_container_width=True)
    st.caption(t('PICP — частка фактичних значень усередині інтервалу; MPIW — середня ширина; Interval Score штрафує широкі інтервали та промахи. Калібрування використовує тільки помилки, відомі до моменту прогнозу. Для малих вибірок результати нестабільні.',
                 'PICP measures empirical coverage; MPIW measures average width; Interval Score penalizes width and misses. Calibration uses only errors available before each forecast. Small validation samples are unstable.'))
    strategy=st.selectbox(t('Стратегія для графіка інтервалів','Strategy for interval chart'),['Static','Periodic','PELT-triggered'],key='ad_plot_strategy')
    g=detail[detail.strategy==strategy].copy()
    lo=f'{"cal_" if calibrated else ""}lower_{level}'
    hi=f'{"cal_" if calibrated else ""}upper_{level}'
    if g[lo].notna().any():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=g.date,y=g[hi],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=g.date,y=g[lo],line=dict(width=0),fill='tonexty',fillcolor='rgba(8,127,140,.15)',name=f'{level}%'))
        fig.add_trace(go.Scatter(x=g.date,y=g.actual,name=t('Факт','Observed'),line=dict(color='#172B4D',width=2)))
        fig.add_trace(go.Scatter(x=g.date,y=g.forecast,name=t('Прогноз','Forecast'),line=dict(color='#087F8C',width=2)))
        fig.update_layout(height=390,margin=dict(l=10,r=10,t=20,b=15),paper_bgcolor='white',plot_bgcolor='white',hovermode='x unified')
        st.plotly_chart(fig,use_container_width=True)
    else:st.warning(t('Ще недостатньо минулих помилок для калібрування.','Not enough prior errors to calibrate intervals.'))
    st.subheader(t('Журнал перенавчання та структурних змін','Refitting and change-point log'))
    st.dataframe(events,hide_index=True,use_container_width=True)
    a,b,c=st.columns(3)
    with a:st.download_button(t('Завантажити прогнози CSV','Download forecasts CSV'),detail.to_csv(index=False).encode('utf-8-sig'),'walk_forward_forecasts.csv',mime='text/csv')
    with b:st.download_button(t('Завантажити метрики CSV','Download metrics CSV'),summary.to_csv(index=False).encode('utf-8-sig'),'adaptive_metrics.csv',mime='text/csv')
    with c:st.download_button(t('Завантажити журнал CSV','Download event log CSV'),events.to_csv(index=False).encode('utf-8-sig'),'adaptive_events.csv',mime='text/csv')
    st.caption(t('Обмеження: SARIMA-параметри в цьому модулі фіксовані для чесного порівняння стратегій; автоматичний підбір є в окремому розділі. PELT реагує на зміни перших різниць. Оцінки на 8 датах є демонстрацією методу, а не остаточним висновком магістерської.',
                 'Limitations: SARIMA order is fixed here to isolate the effect of adaptation; automatic order search is available in Forecasting. PELT detects shifts in first differences. Eight origins demonstrate the method, not a definitive thesis conclusion.'))
