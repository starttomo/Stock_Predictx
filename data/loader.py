# data/loader.py
from sqlalchemy import create_engine
import pandas as pd
from config import DATABASE_CONFIG

def get_engine():
    url = f"mysql+mysqlconnector://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}/{DATABASE_CONFIG['database']}"
    return create_engine(url)

def get_hs300_data():
    engine = get_engine()
    query = "SELECT date, open, high, low, close, volume FROM stocks ORDER BY date"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    return df