"""Leakage-aware rolling-origin benchmark for thesis demonstration.
No stored metrics or fabricated results. Fits use only observations before each origin.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents


def change_points(y, min_size=21, penalty_multiplier=3.0):
    import ruptures as rpt
    arr=np.asarray(y, dtype=float)
    if len(arr)<max(3*min_size,45): return []
    # Detect mean shifts in first differences to reduce trend-induced false alarms.
    delta=np.diff(arr)
    scale=max(float(np.std(delta)),1e-9)
    standardized=(delta/scale).reshape(-1,1)
    penalty=float(penalty_multiplier)*np.log(len(delta))
    stops=rpt.Pelt(model='l2',min_size=min_size,jump=1).fit(standardized).predict(pen=penalty)
    return [int(i+1) for i in stops[:-1]]


def fit_model(train, model, seasonal):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if model=='SARIMA':
            fitted=SARIMAX(train,order=(1,0,0),seasonal_order=(1,1,0,seasonal),
                           enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=60)
        else:
            fitted=UnobservedComponents(train,level='local level',seasonal=seasonal).fit(disp=False,maxiter=60)
    return fitted


def one_step(fit, steps):
    prediction=fit.get_forecast(steps=int(steps))
    mu=float(np.asarray(prediction.predicted_mean)[-1])
    # End-of-horizon forecast standard error (not the in-sample residual std).
    variance=float(np.asarray(prediction.var_pred_mean)[-1])
    return mu,max(np.sqrt(max(variance,0)),1e-9)


def interval_score(y,lo,hi,alpha):
    return (hi-lo)+(2/alpha)*(lo-y if y<lo else 0)+(2/alpha)*(y-hi if y>hi else 0)


def run_backtest(series, model='SARIMA', seasonal=7, origins=8, horizon=1,
                 train_window=280, refit_every=3, penalty=3.0, levels=(80,90,95,99),
                 calibration_size=3):
    """Three strategies, same chronological origins, rolling training window.
    Static: first fit only; periodic: fit at fixed origins; PELT: refit when a NEW
    breakpoint is found near the most recent observation, else retain model.
    Model predictions at later origins are obtained as multi-step forecasts from
    the last fit (never condition on subsequent true values without refitting).
    Calibration uses *only prior* out-of-sample residuals of that strategy.
    """
    y=pd.Series(series).dropna().astype(float).sort_index()
    if len(y)<max(90,4*seasonal,train_window+origins+horizon):
        raise ValueError('Not enough observations for the chosen training window and test horizon.')
    last_origin=len(y)-horizon
    start=last_origin-origins+1
    if start-train_window<0: raise ValueError('Training window exceeds available history.')
    strategies=('Static','Periodic','PELT-triggered')
    state={k:{'fit':None,'fit_origin':None,'errors':[],'retrain_count':0,'last_cp':None} for k in strategies}
    records=[];events=[]
    for origin in range(start,last_origin+1):
        history=y.iloc[:origin]
        training=history.iloc[-train_window:]
        # Recalculate changepoints ONLY on data known at this origin.
        cps=change_points(training.to_numpy(),min_size=max(8,min(21,seasonal*2)),penalty_multiplier=penalty)
        absolute_cp=origin-train_window+cps[-1] if cps else None
        target_index=origin+horizon-1
        actual=float(y.iloc[target_index])
        for strategy in strategies:
            state_item=state[strategy]
            need_fit=state_item['fit'] is None
            if strategy=='Periodic' and (origin-start)%refit_every==0:need_fit=True
            if strategy=='PELT-triggered' and absolute_cp is not None and absolute_cp!=state_item['last_cp'] and absolute_cp>=origin-max(14,seasonal*3):
                need_fit=True
            if need_fit:
                try:
                    fit=fit_model(training,model,seasonal)
                except Exception as exc:
                    raise RuntimeError(f'{strategy} fitting failed at {y.index[origin]}: {exc}') from exc
                state_item['fit']=fit
                state_item['fit_origin']=origin
                state_item['retrain_count']+=1
                events.append({'origin':str(y.index[origin]),'strategy':strategy,'event':'fit / refit','last_detected_change':str(y.index[absolute_cp]) if absolute_cp is not None else ''})
            if strategy=='PELT-triggered' and absolute_cp is not None:state_item['last_cp']=absolute_cp
            # Forecast target from last fitting origin, without accessing unseen values.
            steps=target_index-state_item['fit_origin']+1
            forecast,se=one_step(state_item['fit'],steps)
            previous=np.asarray([r['error'] for r in records if r['strategy']==strategy and r.get('_target_pos', 10**12)<origin],dtype=float)
            # Simple expanding conformal calibration, only after minimum past residuals.
            # Quantile of |errors|; no future outcomes included.
            calibrated=len(previous)>=calibration_size
            rec={'date':str(y.index[target_index]),'origin':str(y.index[origin]),'strategy':strategy,
                 'model':model,'actual':actual,'forecast':forecast,'error':actual-forecast,
                 'retrained':bool(need_fit),'change_point':str(y.index[absolute_cp]) if absolute_cp is not None else '',
                 'calibrated':calibrated,'forecast_steps_since_fit':steps}
            for level in levels:
                alpha=1-level/100
                radius=float(norm.ppf(1-alpha/2)*se)
                rec[f'lower_{level}']=forecast-radius
                rec[f'upper_{level}']=forecast+radius
                if calibrated:
                    # Finite sample conservative conformal quantile; fallback to max for small n.
                    n=len(previous)
                    rank=min(n,int(np.ceil((n+1)*(1-alpha))))
                    q=float(np.sort(np.abs(previous))[rank-1])
                    rec[f'cal_lower_{level}']=forecast-q
                    rec[f'cal_upper_{level}']=forecast+q
                else:
                    rec[f'cal_lower_{level}']=np.nan
                    rec[f'cal_upper_{level}']=np.nan
            records.append(rec)
            # Outcome is available only after the target date; horizon>1 origins overlap.
            # For strict past-only calibration, the next origin cannot use this outcome
            # until its observation date has passed.
        # append errors only after their target has become observable for next origin
        # Update only with outcomes that will be known at next origin.
        for rec in records[-len(strategies):]:rec['_target_pos']=target_index
        # Calibration residuals are selected at each origin only when their target was already observed.
    detail=pd.DataFrame(records).drop(columns=['_target_pos'])
    summary=[]
    for strategy,g in detail.groupby('strategy',sort=False):
        item={'strategy':strategy,'n':len(g),'MAE':float(np.mean(np.abs(g.error))),
              'RMSE':float(np.sqrt(np.mean(g.error**2))),
              'refits':state[strategy]['retrain_count']}
        for level in levels:
            alpha=1-level/100
            for prefix,name in [('', 'raw'),('cal_','calibrated')]:
                lower=g[f'{prefix}lower_{level}'];upper=g[f'{prefix}upper_{level}']
                mask=lower.notna()&upper.notna()
                if mask.any():
                    gg=g.loc[mask];ll=lower.loc[mask];uu=upper.loc[mask]
                    item[f'{name}_PICP_{level}']=float(((gg.actual>=ll)&(gg.actual<=uu)).mean())
                    item[f'{name}_MPIW_{level}']=float((uu-ll).mean())
                    item[f'{name}_IS_{level}']=float(np.mean([interval_score(a,l,u,alpha) for a,l,u in zip(gg.actual,ll,uu)]))
                else:
                    item[f'{name}_PICP_{level}']=np.nan
                    item[f'{name}_MPIW_{level}']=np.nan
                    item[f'{name}_IS_{level}']=np.nan
        summary.append(item)
    return detail,pd.DataFrame(summary),pd.DataFrame(events)
