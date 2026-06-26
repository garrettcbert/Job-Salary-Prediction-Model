import src.model.train as train
from queries import get_all_jobs
from src.preprocessing.preprocess import apply_preprocess
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split
# import pandas as pd
# import matplotlib.pyplot as plt
import time

start = time.time()
df = apply_preprocess(get_all_jobs())

X = df[['searched_title', 'searched_location', 'skills', 'has_cloud', 'has_ml', \
        'has_bigdata', 'has_viz', 'has_db', 'skill_count', 'seniority_score', \
        'years_exp', 'senior_cloud', 'senior_ml', 'industry', \
        'company_size', 'degree_score']]
y = df['salary_avg']

kf = KFold(n_splits=5, shuffle = True)

model = train.build_pipeline()

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

end = time.time()
print(scores.mean(), end - start)



