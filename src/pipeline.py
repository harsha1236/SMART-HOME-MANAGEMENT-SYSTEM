from pathlib import Path
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
CSV = DATA / 'hourly_energy.csv'
URL = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'


def prepare_data(force=False):
    if CSV.exists() and not force:
        df = pd.read_csv(CSV, parse_dates=['timestamp']).set_index('timestamp')
        return df.asfreq('h')
    raw = DATA / 'household_power_consumption.txt'
    if not raw.exists():
        response = requests.get(URL, timeout=180)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            names = [n for n in z.namelist() if n.endswith('household_power_consumption.txt')]
            if not names:
                raise ValueError('Dataset ZIP missing expected text file')
            with z.open(names[0]) as src, raw.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
    df = pd.read_csv(raw, sep=';', usecols=['Date','Time','Global_active_power'], na_values=['?'], low_memory=False)
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.set_index('timestamp').sort_index()
    # Global_active_power is kW averaged over each minute: sum(kW)/60 yields kWh.
    power = df['Global_active_power'].resample('h').agg(['sum','count'])
    hourly = pd.DataFrame({'kwh': power['sum']/60, 'coverage':power['count']/60})
    # Avoid treating missing minutes as zero consumption.
    hourly.loc[hourly.coverage < 0.9, 'kwh'] = np.nan
    hourly['kwh'] = hourly['kwh'].interpolate(limit=3, limit_direction='both')
    hourly = hourly.dropna(subset=['kwh'])
    # Preserve a contiguous hourly timeline; long gaps are excluded by selecting a continuous tail.
    full = hourly['kwh'].asfreq('h')
    if full.isna().any():
        segments = full.notna().ne(full.notna().shift()).cumsum()
        valid = [(len(s),s) for _,s in full.groupby(segments) if s.notna().all()]
        full = max(valid, key=lambda x:x[0])[1]
    result = pd.DataFrame({'kwh':full})
    result.to_csv(CSV, index_label='timestamp')
    return result

LAGS = (1, 2, 3, 24, 48, 168)

def features(history, at):
    vals = [float(history.loc[at-pd.Timedelta(hours=h)]) for h in LAGS]
    return vals + [at.hour, at.dayofweek, int(at.dayofweek >= 5),
                   np.sin(2*np.pi*at.hour/24), np.cos(2*np.pi*at.hour/24)]


def supervised(series):
    X, y, dates = [], [], []
    for i in range(168, len(series)):
        at = series.index[i]
        X.append(features(series, at))
        y.append(float(series.iloc[i]))
        dates.append(at)
    return np.asarray(X), np.asarray(y), pd.DatetimeIndex(dates)


def recursive_forecast(model, history, horizon=24):
    hist = history.copy()
    out = []
    for _ in range(horizon):
        at = hist.index[-1] + pd.Timedelta(hours=1)
        pred = max(0., float(model.predict(np.asarray(features(hist,at)).reshape(1,-1))[0]))
        hist.loc[at] = pred
        out.append(pred)
    return np.array(out)


def seasonal_forecast(history, horizon=24):
    return np.array([float(history.iloc[-24 + i%24]) for i in range(horizon)])


def rag_numeric_forecast(history, horizon=24):
    # Agent-only branch: tool-based analogous-hour forecast; no separately trained ML forecaster.
    out=[]
    for i in range(horizon):
        at=history.index[-1]+pd.Timedelta(hours=i+1)
        candidates=[]
        for days in (1,2,7,14):
            old=at-pd.Timedelta(days=days)
            if old in history.index:
                candidates.append(float(history.loc[old]))
        out.append(max(0., float(np.median(candidates)) if candidates else float(history.iloc[-24+i%24])))
    return np.array(out)


def evaluate(df, test_hours=168):
    series=df['kwh'].asfreq('h').dropna()
    if len(series)<1000 or len(series)!=len(pd.date_range(series.index[0],series.index[-1],freq='h')):
        raise ValueError('Need at least 1000 contiguous hourly observations')
    test_hours=min(test_hours,len(series)//5)
    train=series.iloc[:-test_hours]
    X,y,_=supervised(train)
    model=XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=4)
    model.fit(X,y)
    results=[]
    forecasts={'ML':[],'Agentic tool':[],'Seasonal naive':[]}
    # Rolling-origin 24h evaluations; no future actuals are supplied to predictions.
    for start in range(len(train),len(series),24):
        hist=series.iloc[:start]
        actual=series.iloc[start:start+24].to_numpy()
        if not len(actual):continue
        preds={'ML':recursive_forecast(model,hist,len(actual)),
               'Agentic tool':rag_numeric_forecast(hist,len(actual)),
               'Seasonal naive':seasonal_forecast(hist,len(actual))}
        for name, values in preds.items():
            forecasts[name].extend(values.tolist())
    actual=series.iloc[-test_hours:].to_numpy()
    for name, values in forecasts.items():
        pred=np.array(values)
        results.append({'method':name,'MAE (kWh)':mean_absolute_error(actual,pred),
                        'RMSE (kWh)':np.sqrt(mean_squared_error(actual,pred))})
    # Fit production model on all available data after holdout evaluation.
    X,y,_=supervised(series)
    model.fit(X,y)
    future_index=pd.date_range(series.index[-1]+pd.Timedelta(hours=1),periods=24,freq='h')
    future=pd.DataFrame({'ML':recursive_forecast(model,series),
                         'Agentic tool':rag_numeric_forecast(series),
                         'Seasonal naive':seasonal_forecast(series)},index=future_index)
    comparison=pd.DataFrame({'actual':actual,**forecasts},index=series.index[-test_hours:])
    return pd.DataFrame(results),comparison,future


def fit_ml_model(series):
    """Train the exact same ML forecaster used by the evaluation pipeline."""
    if len(series) < 1000:
        raise ValueError('At least 1000 hours required')
    X, y, _ = supervised(series)
    model = XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                         tree_method="hist", random_state=42, n_jobs=4)
    model.fit(X, y)
    return model
