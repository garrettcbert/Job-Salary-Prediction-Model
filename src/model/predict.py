from src.model.train import apply_model_fit
from queries import get_all_jobs
from src.preprocessing.preprocess import apply_preprocess
import pandas as pd

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

def predict(title, location, seniority_label, skills, degree_choice, industry, company_size):
    model = apply_model_fit()

    seniority_score = seniority_mapping[seniority_label]
    degree_score = degree_mapping[degree_choice]
    skill_count = len(skills)

    input_df = pd.DataFrame([{
        'searched_title': title,
        'searched_location': location,
        'skills' : skills,
        'skill_count': skill_count,
        'seniority_score': seniority_score,
        'senior_cloud': seniority_score * int(any(s in skills for s in ['aws', 'azure', 'google cloud'])),
        'senior_ml': seniority_score * int(any(s in skills for s in ['pytorch', 'tensorflow', 'scikit-learn'])),
        'industry': industry,
        'company_size': company_size,
        'degree_score': degree_score
    }])

    sal_predict = model.predict(input_df)[0]
    return round(sal_predict, 2)