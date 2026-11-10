"""Reproducible scientific evaluation on the active time series, without preset scores."""
import io, json, hashlib, platform, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import scipy, statsmodels
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from adaptive_engine import run_backtest, fit_model

COLORS={'Static':'#8B9BAF','Periodic':'#087F8C','PELT-triggered':'#7959B3','Naive':'#E3A34D','Seasonal naive':'#2875B9'}
LEVELS=(80,90,95,99)

def baseline_backtest(y,origins,horizon,seasonal,calibration_size):
    """Same forecast origins as run_backtest; historical residuals are matured before calibration."""
    start=len(y)-horizon-origins+1
    rows=[]
    for origin in range(start,len(y)-horizon+1):
        target=origin+horizon-1
        for name,lag in [('Naive',1),('Seasonal naive',seasonal)]:
            # Seasonal naive at horizon h: most recent same-season observation.
            anchor=origin-lag+((horizon-1)%lag) if lag>1 else origin-1
            prediction=float(y.iloc[anchor]);actual=float(y.iloc[target])
            old=np.array([abs(r['error']) for r in rows if r['strategy']==name and r['_target']<origin])
            rec={'origin':str(y.index[origin]),'date':str(y.index[target]),'strategy':name,'model':name,
                 'actual':actual,'forecast':prediction,'error':actual-prediction,'_target':target,'retrained':False}
            for level in LEVELS:
                a=1-level/100
                if len(old)>=calibration_size:
                    rank=int(np.ceil((len(old)+1)*(1-a)))
                    q=float(np.sort(old)[min(rank,len(old))-1])
                    rec[f'cal_lower_{level}']=prediction-q;rec[f'cal_upper_{level}']=prediction+q
                else:
                    rec[f'cal_lower_{level}']=np.nan;rec[f'cal_upper_{level}']=np.nan
                rec[f'lower_{level}']=np.nan;rec[f'upper_{level}']=np.nan
            rows.append(rec)
    return pd.DataFrame(rows).drop(columns=['_target'])

def summarize(frame):
    out=[]
    for name,g in frame.groupby('strategy',sort=False):
        r={'strategy':name,'n':len(g),'MAE':float(np.mean(np.abs(g.error))),
           'RMSE':float(np.sqrt(np.mean(g.error**2))),'Bias':float(np.mean(g.error))}
        for level in LEVELS:
            a=1-level/100
            for prefix,tag in [('', 'raw'),('cal_','calibrated')]:
                lo=g[f'{prefix}lower_{level}'];hi=g[f'{prefix}upper_{level}']
                valid=lo.notna()&hi.notna()
                r[f'{tag}_n_{level}']=int(valid.sum())
                if valid.any():
                    actual=g.loc[valid,'actual'].to_numpy();l=lo[valid].to_numpy();u=hi[valid].to_numpy()
                    r[f'{tag}_PICP_{level}']=float(np.mean((actual>=l)&(actual<=u)))
                    r[f'{tag}_MPIW_{level}']=float(np.mean(u-l))
                    r[f'{tag}_IS_{level}']=float(np.mean(u-l+(2/a)*np.maximum(l-actual,0)+(2/a)*np.maximum(actual-u,0)))
                else:
                    for key in ('PICP','MPIW','IS'):r[f'{tag}_{key}_{level}']=np.nan
        out.append(r)
    return pd.DataFrame(out)

def make_pdf(source,series,config,metrics,diagnostics=None):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    import os
    buf=io.BytesIO();font='Helvetica'
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if os.path.exists(path):pdfmetrics.registerFont(TTFont('LabUnicode',path));font='LabUnicode';break
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=42)
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='LabTitle',fontName=font,fontSize=16,leading=23,spaceAfter=12,textColor=colors.HexColor('#142D4E')))
    styles.add(ParagraphStyle(name='LabBody',fontName=font,fontSize=9,leading=15,spaceAfter=8))
    items=[Paragraph('Adaptive Time Series Lab — Research Report',styles['LabTitle'])]
    info=[('Dataset',source),('Observations',len(series)),('Start',str(series.index.min())),('End',str(series.index.max())),('Settings',json.dumps(config,ensure_ascii=False))]
    for k,v in info:items.append(Paragraph(f'{k}: {v}',styles['LabBody']))
    items.append(Spacer(1,12));items.append(Paragraph('Out-of-sample performance',styles['LabTitle']))
    cols=['strategy','n','MAE','RMSE','Bias','calibrated_PICP_95','calibrated_MPIW_95','calibrated_IS_95']
    display=metrics.reindex(columns=cols).copy()
    for c in cols[1:]:display[c]=display[c].map(lambda v:'—' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))
    cells=[[Paragraph(str(c),styles['LabBody']) for c in cols]]+[[Paragraph(str(x),styles['LabBody']) for x in row] for row in display.itertuples(index=False,name=None)]
    table=Table(cells,repeatRows=1,colWidths=[84,26,48,48,45,77,77,77],hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F7')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#CFD8E3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    items.append(table);items.append(Spacer(1,15))
    items.append(Paragraph('Protocol: chronological rolling-origin evaluation; no future observations used to fit models. Intervals are calibrated only from earlier observed forecast errors. Small calibration sets do not imply finite-sample coverage guarantees for dependent data.',styles['LabBody']))
    items.append(Paragraph('Interpretation: lower MAE/RMSE/interval score is preferable; PICP should be compared with its nominal level jointly with interval width. Results depend on selected origins and hyperparameters.',styles['LabBody']))
    if diagnostics is not None:items.append(Paragraph('Residual diagnostics: '+str(diagnostics),styles['LabBody']))
    doc.build(items);return buf.getvalue()

def research_suite(series,english=False,source_name='uploaded data'):
    def t(ua,en):return en if english else ua
    st.markdown('<div class="eyebrow">RESEARCH / BENCHMARK</div>',unsafe_allow_html=True)
    st.title(t('Науковий експеримент і діагностика','Scientific experiment & diagnostics'))
    st.caption(t('Базові моделі · адаптивність · залишки · чутливість · невизначеність · PDF-звіт',
                 'Baselines · adaptation · residuals · sensitivity · uncertainty · PDF report'))
    if len(series)<130:
        st.warning(t('Для експерименту потрібно щонайменше 130 спостережень.','At least 130 observations required.'));return
    signature=hashlib.sha256(pd.util.hash_pandas_object(series,index=True).values.tobytes()).hexdigest()
    if st.session_state.get('research_data_signature')!=signature:
        for k in ('research_results','research_config'):st.session_state.pop(k,None)
        st.session_state['research_data_signature']=signature
    with st.container(border=True):
        st.subheader(t('Протокол експерименту','Experiment protocol'))
        c1,c2,c3=st.columns(3)
        with c1:model=st.selectbox(t('Стохастична модель','Stochastic model'),['State-Space / Kalman','SARIMA'],key='rs_model')
        with c2:seasonal=st.number_input(t('Сезонний період','Seasonal period'),2,30,7,key='rs_season')
        with c3:horizon=st.number_input(t('Горизонт прогнозу','Forecast horizon'),1,7,1,key='rs_horizon')
        c1,c2,c3=st.columns(3)
        with c1:origins=st.slider(t('Тестових дат','Test origins'),6,40,12,key='rs_origins')
        with c2:
            low=max(90,seasonal*4);high=min(700,len(series)-origins-horizon)
            if high<low:st.error(t('Недостатньо даних','Insufficient data'));return
            window=st.slider(t('Навчальне вікно','Training window'),low,high,min(280,high),key='rs_window')
        with c3:refit=st.slider(t('Період перенавчання','Refit period'),2,12,3,key='rs_refit')
        c1,c2=st.columns(2)
        with c1:penalty=st.slider(t('Штраф PELT','PELT penalty'),1.,12.,3.,.5,key='rs_penalty')
        with c2:cal=st.slider(t('Мінімум минулих помилок','Minimum prior errors'),3,12,5,key='rs_cal')
        st.caption(t('Калібрування використовує лише вже відомі помилки; для залежних даних формальна гарантія покриття не заявляється.',
                     'Calibration uses only matured forecast errors; no formal coverage guarantee is claimed for dependent observations.'))
        run=st.button(t('▶ Запустити всі стратегії','▶ Run all strategies'),type='primary')
    config=dict(model=model,seasonal=int(seasonal),horizon=int(horizon),origins=origins,train_window=window,refit_every=refit,penalty=penalty,calibration_size=cal)
    if st.session_state.get('research_config')!=config:st.session_state.pop('research_results',None)
    if run:
        try:
            with st.spinner(t('Виконується послідовна валідація…','Running rolling-origin validation…')):
                detail,_,events=run_backtest(series,model,seasonal,origins,horizon,window,refit,penalty,calibration_size=cal)
                base=baseline_backtest(series,origins,horizon,seasonal,cal)
                combined=pd.concat([detail,base],ignore_index=True)
                summary=summarize(combined)
            st.session_state['research_results']=(combined,summary,events)
            st.session_state['research_config']=config
        except Exception as exc:st.error(f'{t("Помилка експерименту","Experiment error")}: {exc}')
    if 'research_results' not in st.session_state:
        st.info(t('Налаштуй експеримент і запусти обчислення. Жодні результати не підставляються наперед.',
                  'Configure and run the experiment. No scores are prefilled.'));return
    detail,summary,events=st.session_state['research_results']
    tabs=st.tabs([t('Порівняння','Comparison'),t('Невизначеність','Uncertainty'),t('Діагностика','Diagnostics'),t('Чутливість','Sensitivity'),t('Звіт і експорт','Report & export')])
    with tabs[0]:
        st.subheader(t('Однакові тестові дати для всіх стратегій','Identical test origins across strategies'))
        st.dataframe(summary[['strategy','n','MAE','RMSE','Bias']],use_container_width=True,hide_index=True)
        fig=go.Figure()
        for name,g in detail.groupby('strategy'):
            fig.add_trace(go.Scatter(x=g.date,y=g.forecast,name=name,mode='lines+markers',line_color=COLORS.get(name)))
        actual=detail[detail.strategy=='Static']
        fig.add_trace(go.Scatter(x=actual.date,y=actual.actual,name=t('Фактичні','Actual'),line=dict(color='#142D4E',width=3)))
        fig.update_layout(template='plotly_white',height=430,legend=dict(orientation='h'),margin=dict(l=15,r=15,t=25,b=15))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(t('Naive повторює останнє значення, Seasonal naive — останнє значення відповідної фази сезону.',
                     'Naive repeats the latest value; seasonal naive repeats the latest observation at the corresponding seasonal phase.'))
    with tabs[1]:
        level=st.select_slider(t('Рівень інтервалу','Interval level'),options=list(LEVELS),value=95)
        cols=['strategy',f'calibrated_n_{level}',f'calibrated_PICP_{level}',f'calibrated_MPIW_{level}',f'calibrated_IS_{level}']
        st.dataframe(summary[cols],hide_index=True,use_container_width=True)
        fig=go.Figure()
        for _,r in summary.iterrows():
            if pd.notna(r[f'calibrated_PICP_{level}']):fig.add_trace(go.Bar(name=r.strategy,x=[r.strategy],y=[100*r[f'calibrated_PICP_{level}']],marker_color=COLORS.get(r.strategy)))
        fig.add_hline(y=level,line_dash='dash',annotation_text=f'{level}%')
        fig.update_layout(template='plotly_white',height=340,showlegend=False,yaxis_title='PICP, %')
        st.plotly_chart(fig,use_container_width=True)
        st.warning(t('Для малих тестових вибірок PICP нестійкий; порівнюй його разом із шириною інтервалів та Interval Score.',
                     'PICP is unstable for short test samples; compare it jointly with interval width and interval score.'))
    with tabs[2]:
        st.subheader(t('Позавибіркові помилки','Out-of-sample forecast errors'))
        selected=st.selectbox(t('Стратегія','Strategy'),summary.strategy.tolist(),key='rs_diag_model')
        errors=detail.loc[detail.strategy==selected,'error'].to_numpy()
        lag=min(10,max(1,len(errors)//4))
        lb=acorr_ljungbox(errors,lags=[lag],return_df=True)
        c1,c2,c3=st.columns(3)
        c1.metric('Bias',f'{errors.mean():.3f}');c2.metric('RMSE',f'{np.sqrt(np.mean(errors**2)):.3f}');c3.metric(f'Ljung–Box p ({lag})',f'{lb.lb_pvalue.iloc[0]:.4g}')
        f=go.Figure(go.Bar(x=list(range(1,min(15,len(errors)-1)+1)),y=acf(errors,nlags=min(15,len(errors)-1),fft=False)[1:],marker_color='#087F8C'))
        f.update_layout(template='plotly_white',height=320,xaxis_title=t('Лаг','Lag'),yaxis_title='ACF')
        st.plotly_chart(f,use_container_width=True)
        st.caption(t('Діагностика виконана для позавибіркових помилок. При перекритті горизонтів і малому n тест Льюнга–Бокса має обмеження.',
                     'Diagnostics use out-of-sample errors. Ljung–Box interpretation is limited with overlapping horizons and small n.'))
    with tabs[3]:
        st.subheader(t('Аналіз чутливості','Sensitivity analysis'))
        st.caption(t('Повторний розрахунок для кількох значень штрафу PELT. Моделі можуть навчатися кілька хвилин.',
                     'Repeat the backtest for multiple PELT penalties. Model fitting may take several minutes.'))
        penalties=st.multiselect(t('Множники штрафу','Penalty multipliers'),[1.5,3.0,5.0,8.0],default=[1.5,3.0,5.0])
        if st.button(t('Обчислити чутливість','Run sensitivity'),disabled=not penalties):
            results=[]
            for x in penalties:
                try:
                    d,_,ev=run_backtest(series,model,seasonal,origins,horizon,window,refit,x,calibration_size=cal)
                    g=d[d.strategy=='PELT-triggered']
                    results.append({'penalty':x,'MAE':float(np.abs(g.error).mean()),'RMSE':float(np.sqrt(np.mean(g.error**2))),'refits':int((ev.strategy=='PELT-triggered').sum())})
                except Exception as exc:st.warning(f'{x}: {exc}')
            if results:st.session_state['rs_sensitivity']=(config,pd.DataFrame(results))
        cached=st.session_state.get('rs_sensitivity')
        if cached and cached[0]==config:
            st.dataframe(cached[1],hide_index=True,use_container_width=True)
            st.line_chart(cached[1].set_index('penalty')[['MAE','RMSE']])
        st.caption(t('Підбір параметрів за тестовою вибіркою створює оптимістичне зміщення. Для фінального висновку використай окрему контрольну вибірку.',
                     'Tuning on the test set introduces optimistic bias. Use a separate final holdout for scientific claims.'))
    with tabs[4]:
        st.subheader(t('Відтворюваність і експорт','Reproducibility & export'))
        st.json({'source':source_name,'observations':len(series),'range':[str(series.index.min()),str(series.index.max())],
                 'parameters':config,'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__}})
        st.download_button(t('Завантажити всі прогнози CSV','Download forecasts CSV'),detail.to_csv(index=False).encode('utf-8-sig'),'scientific_forecasts.csv',mime='text/csv')
        st.download_button(t('Завантажити метрики CSV','Download metrics CSV'),summary.to_csv(index=False).encode('utf-8-sig'),'scientific_metrics.csv',mime='text/csv')
        st.download_button(t('Завантажити журнал адаптації','Download refitting log'),events.to_csv(index=False).encode('utf-8-sig'),'refitting_log.csv',mime='text/csv')
        try:
            pdf=make_pdf(source_name,series,config,summary)
            st.download_button(t('Завантажити науковий звіт PDF','Download research report PDF'),pdf,'scientific_report.pdf',mime='application/pdf')
        except Exception as exc:st.error(f'PDF: {exc}')
        st.caption(t('Звіт містить фактичні числові результати цього запуску та методологічні обмеження.',
                     'The report contains numerical results from this run and methodological limitations.'))
