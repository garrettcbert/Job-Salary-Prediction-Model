from src.model.train import apply_model_fit
import pandas as pd
import joblib
import os

seniority_mapping = {
    'Intern': 0,
    'Junior': 1,
    'Mid-level': 1.5,
    'Senior': 3,
    'Lead/Principal': 3.5,
    'Manager/Director': 4
}

degree_mapping = {
    'Bachelor\'s': 1,
    'Master\'s': 2,
    'PhD': 3,
    'None': 0
}

def load_model(filename='models/salary_predictor.pkl'):
    if not os.path.exists(filename):
        return apply_model_fit()
    return joblib.load(filename)

def predict(title, location, seniority_label, skills, degree_choice, industry, company_size):
    model = load_model()

    seniority_score = seniority_mapping[seniority_label]
    degree_score = degree_mapping[degree_choice]
    skill_count = len(skills)
    has_cloud = int(any(s in skills for s in ['aws', 'azure', 'gcp']))
    has_ml = int(any(s in skills for s in ['pytorch', 'tensorflow', 'scikit-learn', 'keras']))
    has_bigdata = int(any(s in skills for s in ['spark', 'hadoop', 'hive', 'kafka']))
    has_viz = int(any(s in skills for s in ['tableau', 'power bi', 'plotly']))
    has_db = int(any(s in skills for s in ['postgresql', 'mysql', 'mongodb', 'snowflake']))

    input_df = pd.DataFrame([{
        'searched_title': title,
        'searched_location': location,
        'skills': skills,
        'has_cloud': has_cloud,
        'has_ml': has_ml,
        'has_bigdata': has_bigdata,
        'has_viz': has_viz,
        'has_db': has_db,
        'skill_count': skill_count,
        'seniority_score': float(seniority_score),
        'years_exp': 3.0,
        'senior_cloud': float(seniority_score * has_cloud),
        'senior_ml': float(seniority_score * has_ml),
        'industry': industry,
        'company_size': company_size,
        'degree_score': degree_score
    }])

    sal_predict = model.predict(input_df)[0]
    return round(sal_predict, 2)
