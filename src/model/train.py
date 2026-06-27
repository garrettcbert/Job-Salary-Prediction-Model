from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from src.preprocessing.preprocess import apply_preprocess
from src.queries import get_all_jobs
import joblib
import json
from datetime import datetime

BINARY_FEATURES = [
    'has_cloud', 'has_ml', 'has_bigdata', 'has_viz', 'has_db'
]

NUMERIC_FEATURES = [
    'skill_count', 'seniority_score', 'degree_score', 'years_exp', 'senior_cloud', 'senior_ml'
]

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y = None):
        self.mlb.fit(X)
        return self
    
    def transform(self, X, y = None):
        return self.mlb.transform(X)

def build_preprocessor():
    preprocessor = ColumnTransformer([
        ('skills', MultiLabelBinarizerTransformer(), 'skills'),
        ('location', OneHotEncoder(handle_unknown='ignore'), ['searched_location']),
        ('title', OneHotEncoder(handle_unknown='ignore'), ['searched_title']),
        ('industry', OneHotEncoder(handle_unknown='ignore'), ['industry']),
        ('size', OneHotEncoder(handle_unknown='ignore'), ['company_size']),
        ('numeric', StandardScaler(), NUMERIC_FEATURES),
        ('binary', 'passthrough', BINARY_FEATURES)
    ])
    return preprocessor

def build_pipeline(model='rf'):
    if model == 'xgb':
        estimator = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            max_depth=20
        )
    return Pipeline([
        ('preprocess', build_preprocessor()),
        ('model', estimator)
    ])

def apply_model_fit():
    df = apply_preprocess(get_all_jobs())
    X = df[['searched_title', 'searched_location', 'skills', 'has_cloud', 'has_ml', \
            'has_bigdata', 'has_viz', 'has_db', 'skill_count', 'seniority_score', \
            'years_exp', 'senior_cloud', 'senior_ml', 'industry', \
            'company_size', 'degree_score']]
    y = df['salary_avg']
    pipeline = build_pipeline(model='xgb')

    pipeline.fit(X, y)
    joblib.dump(pipeline, 'models/salary_predictor.pkl')

    metadata = {
        'trained_at': datetime.now().strftime('%Y-%m-%d'),
        'job_count': len(df)
    }

    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f)

    return pipeline

