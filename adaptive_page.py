"""Adaptive experiment page. No precomputed metrics are shown as current results."""
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from adaptive_engine import run_backtest

COLORS={'Static':'#8B9BAF','Periodic':'#087F8C','PELT-triggered':'#7959B3'}

def adaptive_page(series, english=False):
    def t(ua,en):return en if english else ua
    st.markdown('<div class="eyebrow">'+t('МОДУЛЬ ДОСЛІДЖЕННЯ','RESEARCH MODULE')+'</div>',unsafe_allow_html=True)
    st.title(t('Адаптивне моделювання','Adaptive modelling'))
    st.caption(t('Структурні зміни · перенавчання · walk-forward · оцінка прогнозних інтервалів',
                 'Change points · refitting · walk-forward · prediction interval evaluation'))
    st.markdown('<div class="feature-grid"><div class="feature"><strong>01 · '+t('Статична модель','Static baseline')+'</strong><span>'+t('Один початковий fit без перенавчання','One initial fit, no refitting')+'</span></div><div class="feature"><strong>02 · '+t('Періодична адаптація','Scheduled adaptation')+'</strong><span>'+t('Оновлення через задану кількість кроків','Refit at a fixed origin interval')+'</span></div><div class="feature"><strong>03 · PELT</strong><span>'+t('Оновлення після виявлення нової зміни','Refit after detecting a new change point')+'</span></div></div>',unsafe_allow_html=True)
    if len(series)<130:
        st.warning(t('Потрібно щонайменше 130 спостережень.','At least 130 observations are required.'))
        return
    digest=hashlib.sha256(pd.util.hash_pandas_object(series,index=True).values.tobytes()).hexdigest()
    if st.session_state.get('adaptive_signature')!=digest:
        st.session_state.pop('adaptive_result',None)
        st.session_state['adaptive_signature']=digest
    with st.expander('⚙️ '+t('Налаштування експерименту','Experiment settings'),expanded=True):
        st.caption(t('Усі три стратегії використовують однакові тестові дати. Майбутні спостереження не входять до навчальної вибірки.',
                     'All three strategies use the same test dates. Future observations are excluded from training.'))
        a,b,c=st.columns(3)
        with a:model=st.selectbox(t('Базова модель','Base model'),['SARIMA','State-Space / Kalman'],key='ad_model')
        with b:seasonal=st.number_input(t('Сезонний період','Seasonal period'),2,30,7,key='ad_season')
        with c:horizon=st.number_input(t('Горизонт (кроки)','Horizon (steps)'),1,7,1,key='ad_horizon')
        a,b,c=st.columns(3)
        with a:origins=st.slider(t('Тестових дат','Test origins'),4,30,8,key='ad_origins')
        with b:
            min_window=max(90,seasonal*4)
            max_window=min(800,len(series)-origins-horizon)
            if max_window<min_window:
                st.warning(t('Недостатньо даних для параметрів.','Not enough data for these settings.'))
                return
            window=st.slider(t('Навчальне вікно','Training window'),min_window,max_window,min(280,max_window),key='ad_window')
        with c:refit_every=st.slider(t('Крок перенавчання','Refit every N origins'),2,10,3,key='ad_refit')
        a,b=st.columns(2)
        with a:penalty=st.slider(t('Множник штрафу PELT','PELT penalty multiplier'),1.,12.,3.,.5,key='ad_penalty')
        with b:calibration_size=st.slider(t('Мінімум минулих помилок','Minimum prior errors'),2,10,3,key='ad_cal')
        st.caption(t('Для SARIMA у цьому порівнянні фіксовано (1,0,0)(1,1,0,s), щоб ізолювати ефект адаптації. Автоматичний підбір параметрів доступний у розділі прогнозування.',
                     'SARIMA uses fixed (1,0,0)(1,1,0,s) to isolate adaptation effects. Automatic order selection is available in Forecasting.'))
        run=st.button('▶  '+t('Запустити експеримент','Run experiment'),type='primary',key='ad_run')
    settings=(model,seasonal,horizon,origins,window,refit_every,penalty,calibration_size)
    if st.session_state.get('adaptive_settings')!=settings:
        st.session_state.pop('adaptive_result',None)
    if run:
        try:
            with st.spinner(t('Обчислюємо прогнози на історичних датах…','Running historical forecasts…')):
                detail,summary,events=run_backtest(series,model,seasonal,origins,horizon,window,refit_every,penalty,calibration_size=calibration_size)
            st.session_state['adaptive_result']=(detail,summary,events)
            st.session_state['adaptive_settings']=settings
        except Exception as exc:st.error(str(exc))
    if 'adaptive_result' not in st.session_state:
        st.info(t('Налаштуй параметри й натисни «Запустити експеримент», щоб отримати фактичні результати.',
                  'Set the parameters and run the experiment to calculate actual results.'))
        return
    detail,summary,events=st.session_state['adaptive_result']
    st.markdown('### '+t('Результати експерименту','Experiment results'))
    best=summary.loc[summary.MAE.idxmin()]
    a,b,c,d=st.columns(4)
    a.metric(t('Найменший MAE','Lowest MAE'),f'{best.MAE:,.2f}')
    b.metric(t('Стратегія за MAE','Best by MAE'),str(best.strategy))
    c.metric(t('Тестових прогнозів','Test forecasts'),str(int(best.n)))
    d.metric(t('Оновлень моделі','Model fits'),str(int(best.refits)))
    tab1,tab2,tab3,tab4=st.tabs(['📈 '+t('Прогнози','Forecasts'),'🛡️ '+t('Невизначеність','Uncertainty'),'🔄 '+t('Адаптація','Adaptation'),'⬇️ '+t('Експорт','Exports')])
    with tab1:
        st.caption(t('Порівняння прогнозів на однакових тестових датах.','Comparison on identical test dates.'))
        fig=go.Figure()
        for strategy,g in detail.groupby('strategy',sort=False):
            fig.add_trace(go.Scatter(x=g.date,y=g.forecast,name=strategy,mode='lines+markers',line=dict(color=COLORS[strategy],width=2)))
        g=detail[detail.strategy=='Static']
        fig.add_trace(go.Scatter(x=g.date,y=g.actual,name=t('Фактичні','Observed'),mode='lines+markers',line=dict(color='#142D4E',width=3)))
        fig.update_layout(height=450,paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=20,r=20,t=30,b=20),hovermode='x unified',legend=dict(orientation='h',y=1.12),yaxis=dict(gridcolor='#E8EDF5'),xaxis=dict(showgrid=False))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(summary[['strategy','n','MAE','RMSE','refits']].round(3),use_container_width=True,hide_index=True)
        st.caption(t('Найменша помилка на короткій вибірці не гарантує переваги методу в інших умовах.',
                     'Lowest error on a short sample does not establish general superiority.'))
    with tab2:
        a,b,c=st.columns(3)
        with a:level=st.selectbox(t('Рівень інтервалу','Interval level'),[80,90,95,99],index=2,key='ad_level')
        with b:calibrated=st.toggle(t('Калібрований','Calibrated'),value=False,key='ad_calibrated')
        with c:strategy=st.selectbox(t('Стратегія','Strategy'),['Static','Periodic','PELT-triggered'],key='ad_plot_strategy')
        prefix='calibrated' if calibrated else 'raw'
        cols=['strategy',f'{prefix}_PICP_{level}',f'{prefix}_MPIW_{level}',f'{prefix}_IS_{level}']
        display=summary[cols].copy()
        display.columns=['Strategy',f'PICP ({level}%)','MPIW','Interval Score']
        st.dataframe(display.round(4),hide_index=True,use_container_width=True)
        g=detail[detail.strategy==strategy].copy()
        lo=f'{"cal_" if calibrated else ""}lower_{level}'
        hi=f'{"cal_" if calibrated else ""}upper_{level}'
        if g[lo].notna().any():
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=g.date,y=g[hi],line=dict(width=0),showlegend=False))
            fig.add_trace(go.Scatter(x=g.date,y=g[lo],line=dict(width=0),fill='tonexty',fillcolor='rgba(8,127,140,.15)',name=f'{level}%'))
            fig.add_trace(go.Scatter(x=g.date,y=g.actual,name=t('Фактичні','Observed'),line=dict(color='#142D4E',width=2)))
            fig.add_trace(go.Scatter(x=g.date,y=g.forecast,name=t('Прогноз','Forecast'),line=dict(color='#087F8C',width=2)))
            fig.update_layout(height=430,paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=20,r=20,t=30,b=20),hovermode='x unified',yaxis=dict(gridcolor='#E8EDF5'),xaxis=dict(showgrid=False))
            st.plotly_chart(fig,use_container_width=True)
        else:st.warning(t('Поки недостатньо попередніх помилок для калібрування.','Not enough previous errors for calibration.'))
        st.caption(t('PICP — емпіричне покриття; MPIW — середня ширина; Interval Score враховує ширину та промахи. Калібрування використовує лише доступні на дату прогнозу помилки.',
                     'PICP is empirical coverage; MPIW is mean width; Interval Score penalizes width and misses. Calibration uses errors known at forecast time.'))
    with tab3:
        st.markdown('#### '+t('Журнал структурних змін і перенавчання','Change-point and refitting log'))
        st.dataframe(events,hide_index=True,use_container_width=True)
        counts=summary[['strategy','refits']].copy()
        st.bar_chart(counts.set_index('strategy'),color='#087F8C')
        st.caption(t('PELT аналізує перші різниці тільки навчального вікна. Точка зміни є статистичною ознакою, а не доказом зовнішньої причини.',
                     'PELT analyzes first differences in the training window only. A detected breakpoint is not evidence of an external cause.'))
    with tab4:
        a,b,c=st.columns(3)
        with a:st.download_button(t('Прогнози CSV','Forecasts CSV'),detail.to_csv(index=False).encode('utf-8-sig'),'walk_forward_forecasts.csv',mime='text/csv')
        with b:st.download_button(t('Метрики CSV','Metrics CSV'),summary.to_csv(index=False).encode('utf-8-sig'),'adaptive_metrics.csv',mime='text/csv')
        with c:st.download_button(t('Журнал CSV','Events CSV'),events.to_csv(index=False).encode('utf-8-sig'),'adaptive_events.csv',mime='text/csv')
        st.caption(t('Усі файли містять результати поточного експерименту.','All downloads reflect the current experiment.'))
