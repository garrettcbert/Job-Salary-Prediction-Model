import os
import json
import gradio as gr
import src.model.train as train
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

pio.templates['salary_theme'] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", size=13, color="#e8e8e8"),
        plot_bgcolor="#1f1f1f",
        paper_bgcolor="#1f1f1f",
        colorway=["#6FA8DC", "#93C47D", "#E06666", "#A78BCB", "#E8C56C"],
        title=dict(x=0.5, font=dict(size=18, color = "#ffffff")),
        xaxis=dict(showgrid=True, gridcolor="#3a3a3a", zeroline=False, linecolor = "#555",
                   tickfont = dict(color="#e8e8e8"), title_font=dict(color="#e8e8e8")),
        yaxis=dict(showgrid=True, gridcolor="#3a3a3a", zeroline=False, linecolor = "#555",
                   tickfont=dict(color="#e8e8e8"), title_font=dict(color="#e8e8e8")),
        margin=dict(l=60, r=30, t=60, b=60),
    )
)

pio.templates.default = 'salary_theme'

def load_model(filename='models/salary_predictor.pkl'):
    if not os.path.exists(filename):
        print("Model file not found. Training model...")
        model = train.apply_model_fit()
    else:
        model = joblib.load(filename)
    return model

def load_metadata(filename = 'models/metadata.json'):
    if os.path.exists(filename):
        with open(filename) as f:
            return json.load(f)
    
    return {'trained_at': 'unknown', 'job_count': 0}

seniority_mapping = {
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

def get_skill_buckets(skills):
    has_cloud = int(any(s in skills for s in ['aws', 'azure', 'google cloud', 'gcp']))
    has_ml = int(any(s in skills for s in ['pytorch', 'tensorflow', 'scikit-learn', 'keras']))
    has_bigdata = int(any(s in skills for s in ['spark', 'hadoop', 'hive', 'kafka']))
    has_viz = int(any(s in skills for s in ['tableau', 'power bi', 'plotly']))
    has_db = int(any(s in skills for s in ['postgresql', 'mysql', 'mongodb', 'snowflake']))
    return (has_cloud, has_ml, has_bigdata, has_viz, has_db)

def build_input_df(title, location, seniority, skills_input, degree_choice, industry, company_size, years_exp):
    seniority_score = seniority_mapping[seniority]
    degree_score = degree_mapping[degree_choice]
    has_cloud, has_ml, has_bigdata, has_viz, has_db = get_skill_buckets(skills_input)

    return pd.DataFrame([{
        'searched_title': title,
        'searched_location': location,
        'skills': skills_input,
        'has_cloud': has_cloud,
        'has_ml': has_ml,
        'has_bigdata': has_bigdata,
        'has_viz': has_viz,
        'has_db': has_db,
        'skill_count': len(skills_input),
        'seniority_score': float(seniority_score),
        'years_exp': 10.0 if years_exp == '10+' else float(years_exp),
        'senior_cloud': float(seniority_score * has_cloud),
        'senior_ml': float(seniority_score * has_ml),
        'industry': industry,
        'company_size': company_size,
        'degree_score': degree_score
    }])

model = load_model()
metadata = load_metadata()

def get_tree_preds(pipeline, X_transformed):
    estimator = pipeline.named_steps['model']
    if isinstance(estimator, XGBRegressor):
        booster = estimator.get_booster()
        dmatrix = xgb.DMatrix(X_transformed)
        n = estimator.n_estimators
        return [booster.predict(dmatrix, iteration_range=(0, i + 1))[0] for i in range(n)]
    else:
        return [est.predict(X_transformed)[0] for est in estimator.estimators_]

def predict(title, location, seniority, prog_skills, ml_skills, cloud_skills,
            bigdata_skills, viz_skills, db_skills, degree_choice, industry, company_size, years_exp):
    skills_input = prog_skills + ml_skills + cloud_skills + bigdata_skills + viz_skills + db_skills
    input_df = build_input_df(title, location, seniority, skills_input, degree_choice, industry, company_size, years_exp)

    X_transformed = model.named_steps['preprocess'].transform(input_df)
    tree_preds = get_tree_preds(model, X_transformed)
    low, _, high = np.percentile(tree_preds, [25, 50, 75])
    prediction = model.predict(input_df)[0]

    gaps = skill_gap_analysis(input_df, prediction, ['has_cloud', 'has_ml', 'has_bigdata', 'has_viz', 'has_db'])

    return (
        f"Predicted Salary: ${prediction:,.2f}\n50% Confidence Interval: (${low:,.2f}, ${high:,.2f})",
        tree_distribution(tree_preds, prediction),
        skill_gap_plot(gaps)
    )

def tree_distribution(tree_preds, prediction):
    low, high = np.percentile(tree_preds, [25, 75])

    x_min, x_max = min(tree_preds), max(tree_preds)
    x_padding = (x_max - x_min) * 0.1

    fig = px.histogram(tree_preds, nbins=20, title='Distribution of Tree Predictions', labels={'value': 'Predicted Salary ($)',
                                                                                               'count': 'Model Votes'})
    fig.update_traces(marker_line_width = 0)
    fig.add_vrect(x0 = low, x1 = high,
                  fillcolor = "#6FA8DC", 
                  opacity = 0.15, 
                  line_width = 0,)
    
    fig.add_vline(x=prediction, 
                  line_dash="dash", 
                  line_color="#E06666", 
                  line_width = 2,
                  annotation_text=f"Best Estimate: ${prediction:,.0f}", 
                  annotation_position="top")
    
    fig.update_layout(width=900, 
                      margin=dict(l=80, r=40, t=100, b=80),
                      showlegend = False, 
                      xaxis = dict(
                          title = "Predicted Salary ($)",
                          range = [x_min - x_padding, x_max + x_padding]
                      ),
                      yaxis=dict(
                          title="How many models agree",
                          range=[0, None]
                        ),
                      bargap = 0.05)
    return fig

def skill_gap_analysis(input_df, base_prediction, skill_cols):
    skill_gaps = {}
    for skill in skill_cols:
        if input_df[skill].iloc[0] == 0:
            modified_df = input_df.copy()
            modified_df[skill] = 1
            skill_gaps[skill] = model.predict(modified_df)[0] - base_prediction
    return dict(sorted(skill_gaps.items(), key=lambda item: item[1], reverse=True))

def skill_gap_plot(skill_gaps):
    values = list(skill_gaps.values())
    padding = (max(values) - min(values)) * 0.25 if values else 200
    fig = px.bar(
        x=values,
        y=list(skill_gaps.keys()),
        orientation='h',
        title='Estimated Salary Increase from Adding Each Missing Skill',
        labels={'x': 'Estimated Salary Increase ($)', 'y': ''}
        )
    fig.update_traces(text = [f"${v:,.0f}" for v in values], 
                      textposition = 'outside',
                      marker_color = "#6FA8DC")
    fig.update_layout(width=900, 
                      margin=dict(l=100, r=100),
                      showlegend = False, 
                      yaxis = dict(autorange = 'reversed'),
                      xaxis = dict(
                          title = 'Estimated Salary Increase ($)',
                          range = [min(values) - padding, max(values) + padding]
                      )) 
    return fig

with gr.Blocks() as demo:
    gr.Markdown(f"""
    # Salary Predictor
                
    Estimate salary for data-related roles based on job title, location, seniority,
    required skills, and other job-posting attributes. The model is trained on real
    job postings scraped from major job boards.
                
    **How it works:** An XGBoost model trained on job postings predicts your salary
    range, then shows you which missing skills would have the biggest impact on your
    estimate.
                
    *Model last trained: {metadata['trained_at']} on {metadata['job_count']:,}
    postings*
    """)
    
    with gr.Row():
        location = gr.Dropdown(choices=[
            'New York, NY',
            'San Francisco, CA',
            'Chicago, IL',
            'Seattle, WA',
            'Austin, TX',
            'Boston, MA',
            'Los Angeles, CA',
            'Denver, CO',
            'Atlanta, GA',
            'Washington, DC'
        ], label="Location")

        title = gr.Dropdown(
            choices=[
                ('Data Scientist', 'data scientist'),
                ('Data Engineer', 'data engineer'),
                ('Data Analyst', 'data analyst'),
                ('Machine Learning Engineer', 'machine learning engineer'),
                ('AI Engineer', 'AI engineer'),
                ('Business Intelligence Analyst', 'business intelligence analyst'),
                ('Analytics Engineer', 'analytics engineer'),
            ],
            label="Job Title"
        )

        industry = gr.Dropdown(
            choices=[
                ('Tech', 'tech'), ('Finance', 'finance'), ('Healthcare', 'healthcare'),
                ('Government', 'government'), ('Retail', 'retail')
            ],
            label="Industry", value='tech'
        )

        company_size = gr.Dropdown(
            choices=[('Large', 'large'), ('Startup', 'startup'), ('Small', 'small')],
            label='Company Size', value='large'
        )

        seniority = gr.Dropdown(
            choices=['Junior', 'Mid-level', 'Senior', 'Lead/Principal', 'Manager/Director'],
            label="Seniority Level")
        
        years_exp = gr.Dropdown(
                choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, '10+'],
                label='Years of Experience',
                value=3
            )

        degree_choice = gr.Dropdown(
                choices=['None', "Bachelor's", "Master's", 'PhD'],
                label="Degree Requirement",
                value="Bachelor's"
            )

    with gr.Accordion('Skills', open=True):
        with gr.Row():
            prog_skills = gr.CheckboxGroup(
                choices=['python', 'r', 'sql', 'java', 'c++', 'scala', 'julia'],
                label="Programming Languages"
            )
            ml_skills = gr.CheckboxGroup(
                choices=['scikit-learn', 'tensorflow', 'pytorch', 'keras'],
                label="Machine Learning"
            )
        with gr.Row():
            cloud_skills = gr.CheckboxGroup(
                choices=['aws', 'azure', 'google cloud', 'gcp'],
                label="Cloud Platforms"
            )
            bigdata_skills = gr.CheckboxGroup(
                choices=['hadoop', 'spark', 'hive', 'pig', 'kafka'],
                label="Big Data Tools"
            )
        with gr.Row():
            viz_skills = gr.CheckboxGroup(
                choices=['matplotlib', 'seaborn', 'ggplot2', 'plotly', 'tableau', 'power bi'],
                label="Visualization"
            )
            db_skills = gr.CheckboxGroup(
                choices=['mysql', 'postgresql', 'mongodb', 'sqlite', 'snowflake'],
                label="Databases"
            )
    
    gr.Examples(
        examples = [
            ['data scientist', 'San Francisco, CA', 'Senior',
             ['python', 'sql'], ['pytorch'], [], [], [], [],
             "Master's", 'tech', 'large', 5], 
             ['data analyst', 'Chicago, IL', 'Junior',
              ['sql'], [], [], [], ['tableau'], ['mysql'],
              "Bachelor's", 'retail', 'small', 1]
        ],
        inputs = [title, location, seniority, prog_skills, ml_skills, cloud_skills,
                bigdata_skills, viz_skills, db_skills, degree_choice, industry,
                company_size, years_exp],
        label = 'Try an example'
    )

    analyze_but = gr.Button('Analyze', variant = 'primary')

    with gr.Row():
        salary_output = gr.Textbox(label="Prediction", scale=1)

    with gr.Row():
        dist_plot = gr.Plot(label = "Prediction Confidence")
        gap_plot = gr.Plot(label = 'Skill Gap Analysis')

    analyze_but.click(
        fn=predict,
        inputs=[title, location, seniority, prog_skills, ml_skills, cloud_skills,
                bigdata_skills, viz_skills, db_skills, degree_choice, industry, company_size, years_exp],
        outputs=[salary_output, dist_plot, gap_plot],
        show_progress = 'full'
    )

demo.launch(theme = "gradio/monochrome")
