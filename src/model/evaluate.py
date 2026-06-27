import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import src.model.train as train
from src.queries import get_all_jobs
from src.preprocessing.preprocess import apply_preprocess, skills as all_skills
from sklearn.model_selection import cross_val_score, KFold
import time

if __name__ == '__main__':
    df = apply_preprocess(get_all_jobs())

    found_skills = set(s for skill_list in df['skills'] for s in skill_list)
    missing_skills = [s for s in all_skills if s not in found_skills]
    if missing_skills:
        print(f"Skills not found in scraped jobs (ignored by model): {missing_skills}\n")

    X = df[['searched_title', 'searched_location', 'skills', 'has_cloud', 'has_ml',
            'has_bigdata', 'has_viz', 'has_db', 'skill_count', 'seniority_score',
            'years_exp', 'senior_cloud', 'senior_ml', 'industry',
            'company_size', 'degree_score']]
    y = df['salary_avg']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model_key in [('Random Forest', 'rf'), ('XGBoost', 'xgb')]:
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            scores = cross_val_score(train.build_pipeline(model_key), X, y, cv=kf, scoring='r2')
        elapsed = time.time() - start
        results[name] = scores
        print(f"{name:<20} R² mean: {scores.mean():.4f}  std: {scores.std():.4f}  ({elapsed:.1f}s)")

    winner = max(results, key=lambda k: results[k].mean())
    print(f"\nBetter model: {winner}")
