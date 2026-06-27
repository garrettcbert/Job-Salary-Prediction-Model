from src.database import get_engine
import pandas as pd

def get_all_jobs():
    engine = get_engine()
    return pd.read_sql_query("SELECT * FROM Jobs", engine)
