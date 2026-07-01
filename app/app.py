import sys
import os
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBRegressor

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

def load_model(filename=None):
    if filename is None:
        filename = _PROJECT_ROOT / 'models' / 'salary_predictor.pkl'
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Model file not found at {filename}. "
            "Run `python scripts/retrain.py` to scrape data and train the model."
        )
    return joblib.load(filename)

def load_metadata(filename=None):
    if filename is None:
        filename = _PROJECT_ROOT / 'models' / 'metadata.json'
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

LOCATIONS = [
    'New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Seattle, WA',
    'Austin, TX', 'Boston, MA', 'Los Angeles, CA', 'Denver, CO',
    'Atlanta, GA', 'Washington, DC'
]

TITLES = [
    ('Data Scientist', 'data scientist'),
    ('Data Engineer', 'data engineer'),
    ('Data Analyst', 'data analyst'),
    ('Machine Learning Engineer', 'machine learning engineer'),
    ('AI Engineer', 'AI engineer'),
    ('Business Intelligence Analyst', 'business intelligence analyst'),
    ('Analytics Engineer', 'analytics engineer'),
]

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
        salary_gauge(prediction, low, high, tree_preds),
        tree_distribution(tree_preds, prediction),
        skill_gap_plot(gaps),
        location_comparison_plot(title, seniority, skills_input, degree_choice, industry, company_size, years_exp, location),
        title_comparison_plot(location, seniority, skills_input, degree_choice, industry, company_size, years_exp, title)
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

def salary_gauge(prediction, low, high, tree_preds):
    p10, p90 = np.percentile(tree_preds, [10, 90])
    padding = (p90 - p10) * 0.12

    fig = go.Figure()

    fig.add_shape(type='rect', x0=p10, x1=p90, y0=0.35, y1=0.65,
                  fillcolor='#2a2a2a', line_width=0, layer='below')
    fig.add_shape(type='rect', x0=low, x1=high, y0=0.2, y1=0.8,
                  fillcolor='#6FA8DC', opacity=0.3, line_width=0)
    fig.add_shape(type='line', x0=prediction, x1=prediction, y0=0.05, y1=0.95,
                  line=dict(color='#E06666', width=5), opacity= 0.2)

    fig.add_annotation(x=prediction, y = 1.25, yref='paper',
                       text=f"<b>${prediction:,.0f}</b>",
                       showarrow=False, font=dict(size=26, color='#E06666'))
    fig.add_annotation(x=(low + high) / 2, y=0.04, yref='paper',
                       text=f"50% CI: ${low:,.0f} – ${high:,.0f}",
                       showarrow=False, font=dict(size=13, color='#6FA8DC'))

    fig.update_layout(
        title=dict(text='Salary Estimate', font=dict(size=16)),
        height=300,
        paper_bgcolor='#1f1f1f',
        plot_bgcolor='#1f1f1f',
        xaxis=dict(
            range=[p10 - padding, p90 + padding],
            tickformat='$,.0f',
            tickfont=dict(color='#e8e8e8'),
            showgrid=False,
            zeroline=False,
            linecolor='#555'
        ),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=80, r=80, t=70, b=70),
        showlegend=False
    )
    return fig

def location_comparison_plot(title, seniority, skills_input, degree_choice, industry, company_size, years_exp, selected_location):
    preds = []

    for loc in LOCATIONS:
        df = build_input_df(title, loc, seniority, skills_input, degree_choice, industry, company_size, years_exp)
        preds.append(model.predict(df)[0])

    colors = ["#E06666" if loc == selected_location else "#6FA8DC" for loc in LOCATIONS]
    fig = px.bar(x=preds, y=LOCATIONS, orientation='h',
                 title='Predicted Salary by Location',
                 labels={'x': 'Predicted Salary ($)', 'y': ''})
    fig.update_traces(text=[f"${p:,.0f}" for p in preds], textposition='outside', marker_color=colors)
    fig.update_layout(width=900, margin=dict(l=150, r=120), showlegend=False,
                      yaxis=dict(autorange='reversed'),
                      xaxis=dict(title='Predicted Salary ($)',
                                 range = [-10000, max(preds) + 50000]))
    return fig

def title_comparison_plot(location, seniority, skills_input, degree_choice, industry, company_size, years_exp, selected_title):
    preds = []
    for _, val in TITLES:
        df = build_input_df(val, location, seniority, skills_input, degree_choice, industry, company_size, years_exp)
        preds.append(model.predict(df)[0])

    labels = [label for label, _ in TITLES]
    colors = ["#E06666" if val == selected_title else "#6FA8DC" for _, val in TITLES]
    fig = px.bar(x=preds, y=labels, orientation='h',
                 title='Predicted Salary by Job Title',
                 labels={'x': 'Predicted Salary ($)', 'y': ''})
    fig.update_traces(text=[f"${p:,.0f}" for p in preds], textposition='outside', marker_color=colors)
    fig.update_layout(width=900, margin=dict(l=210, r=120), showlegend=False,
                      yaxis=dict(autorange='reversed'),
                      xaxis=dict(title='Predicted Salary ($)',
                                 range = [-10000, max(preds) + 50000]))
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
        location = gr.Dropdown(choices=LOCATIONS, label="Location", value='San Francisco, CA')

        title = gr.Dropdown(choices=TITLES, label="Job Title", value='data scientist')

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
            label="Seniority Level", value='Mid-level')
        
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

    # Created with render=False so gr.Examples can reference them before the
    # accordion renders them below
    prog_skills = gr.CheckboxGroup(
        choices=['python', 'r', 'sql', 'java', 'c++', 'scala', 'julia'],
        label="Programming Languages", render=False
    )
    ml_skills = gr.CheckboxGroup(
        choices=['scikit-learn', 'tensorflow', 'pytorch', 'keras'],
        label="Machine Learning", render=False
    )
    cloud_skills = gr.CheckboxGroup(
        choices=['aws', 'azure', 'google cloud', 'gcp'],
        label="Cloud Platforms", render=False
    )
    bigdata_skills = gr.CheckboxGroup(
        choices=['hadoop', 'spark', 'hive', 'pig', 'kafka'],
        label="Big Data Tools", render=False
    )
    viz_skills = gr.CheckboxGroup(
        choices=['matplotlib', 'seaborn', 'ggplot2', 'plotly', 'tableau', 'power bi'],
        label="Visualization", render=False
    )
    db_skills = gr.CheckboxGroup(
        choices=['mysql', 'postgresql', 'mongodb', 'sqlite', 'snowflake'],
        label="Databases", render=False
    )

    gr.Examples(
        examples=[
            ['data scientist', 'San Francisco, CA', 'Senior',
             ['python', 'sql'], ['pytorch'], [], [], [], [],
             "Master's", 'tech', 'large', 5],
            ['data analyst', 'Chicago, IL', 'Junior',
             ['sql'], [], [], [], ['tableau'], ['mysql'],
             "Bachelor's", 'retail', 'small', 1],
            ['machine learning engineer', 'Seattle, WA', 'Senior',
             ['python', 'scala'], ['pytorch', 'tensorflow'], ['aws'], ['spark'], [], ['postgresql'],
             "Master's", 'tech', 'large', 6],
            ['data engineer', 'New York, NY', 'Mid-level',
             ['python', 'sql', 'scala'], [], ['aws', 'gcp'], ['spark', 'kafka'], [], ['snowflake', 'mongodb'],
             "Bachelor's", 'finance', 'startup', 3],
            ['analytics engineer', 'Austin, TX', 'Mid-level',
             ['sql', 'python'], [], [], [], ['tableau', 'power bi'], ['postgresql', 'mysql'],
             "Bachelor's", 'finance', 'large', 4],
        ],
        inputs=[title, location, seniority, prog_skills, ml_skills, cloud_skills,
                bigdata_skills, viz_skills, db_skills, degree_choice, industry,
                company_size, years_exp],
        label='Try an example'
    )

    with gr.Accordion('Skills', open=True):
        with gr.Row():
            prog_skills.render()
            ml_skills.render()
        with gr.Row():
            cloud_skills.render()
            bigdata_skills.render()
        with gr.Row():
            viz_skills.render()
            db_skills.render()

    analyze_but = gr.Button('Analyze', variant='primary')

    with gr.Row():
        salary_output = gr.Plot(label="Salary Estimate")

    gr.Markdown(
        "*The red line is the final salary estimate. The blue band is the 50% confidence interval, "
        "the range where half of the model's 200 individual tree predictions fell. The final estimate "
        "can occasionally sit outside this band when later trees push the prediction beyond where the "
        "earlier trees clustered.*"
    )

    with gr.Row():
        dist_plot = gr.Plot(label = "Prediction Confidence")
        gap_plot = gr.Plot(label = 'Skill Gap Analysis')

    with gr.Row():
        loc_plot = gr.Plot(label = "Salary by Location")
        title_plot = gr.Plot(label = "Salary by Job Title")

    analyze_but.click(
        fn=predict,
        inputs=[title, location, seniority, prog_skills, ml_skills, cloud_skills,
                bigdata_skills, viz_skills, db_skills, degree_choice, industry, company_size, years_exp],
        outputs=[salary_output, dist_plot, gap_plot, loc_plot, title_plot],
        show_progress = 'full'
    )

if __name__ == '__main__':
    demo.launch(theme="gradio/monochrome", share = True)
