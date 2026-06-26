from src.database import get_engine
import pandas as pd

def get_all_jobs():
    engine = get_engine()
    query = """
    SELECT * FROM Jobs
    """
    df = pd.read_sql_query(query, engine)
    return df

# def get_job_location(job, location):
#     engine = get_engine()
#     query = """
#     SELECT * FROM Jobs WHERE searched_title = ? and searched_location = ?
#     """
#     df = pd.read_sql_query(query, engine, params = (job, location))
#     return df

# def get_salary_title_loc(job, location):
#     engine = get_engine()
#     query = """
#     SELECT title, location, salary_avg, salary_max, salary_min FROM Jobs WHERE searched_title = ? and searched_location = ? and salary_avg not Null
#     """
#     df = pd.read_sql_query(query, engine, params = (job, location))
#     return df

# def get_skills_query():
#     engine = get_engine()
#     query = """
#     SELECT title, extracted_skills FROM Jobs
#     """
#     df = pd.read_sql_query(query, engine)
#     return df


    

