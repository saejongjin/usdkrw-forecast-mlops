import os
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def train_and_save_model():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True) # 폴더가 없으면 자동 생성

    MODEL_PATH = os.path.join(PROCESSED_DIR, "predict_model.pkl")
    IMPORTANCE_PATH = os.path.join(PROCESSED_DIR, "feature_importance.csv")
    TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train_data.csv")


    current_utc = pd.Timestamp.now(tz='UTC').floor('min')
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=(current_utc - pd.Timedelta(days=15)).strftime('%Y-%m-%d'), 
        end_date=current_utc.tz_convert('America/New_York').strftime('%Y-%m-%d')
    )
    
    grid_list = [
        pd.date_range(date.strftime('%Y-%m-%d 00:00:00'), date.strftime('%Y-%m-%d 23:59:00'), freq='1min', tz='UTC') 
        for date in schedule.index
    ]
    full_time_grid = grid_list[0].append(grid_list[1:])[grid_list[0].append(grid_list[1:]) <= current_utc][-7200:]


    ticker_map = {
        'BTC-USD': 'BTC', 
        'JPY=X': 'JPY', 
        'CL=F': 'WTI', 
        'GC=F': 'GOLD', 
        'DX-Y.NYB': 'DXY', 
        'KRW=X': 'KRW'
    }
    raw_data = yf.download(list(ticker_map.keys()), period='7d', interval='1m')['Close']
    raw_data.rename(columns=ticker_map, inplace=True)

    if raw_data.index.tz is None: 
        raw_data = raw_data.tz_localize('UTC')
    else: 
        raw_data = raw_data.tz_convert('UTC')

    df = raw_data.reindex(full_time_grid)
    #df = df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    df = df.interpolate(method="linear").ffill().bfill()

    df['KRW_Future'] = df['KRW'].shift(-15)
    df['Target'] = np.where(df['KRW_Future'] > df['KRW'], 1, 0)
    df.dropna(inplace=True)
    df.to_csv(TRAIN_DATA_PATH, index=True)

    features = ['BTC', 'JPY', 'WTI', 'GOLD', 'DXY']
    X = df[features]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)


    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # 변수 중요도 저장
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    importance_df['Importance(%)'] = (importance_df['Importance'] * 100).round(2)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)



if __name__ == "__main__":
    train_and_save_model()