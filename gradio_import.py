import os
import gradio as gr
import src.model.train as train
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import joblib

def load_model(filename='models/salary_predictor.pkl'):
    if not os.path.exists(filename):
        print("Model file not found. Training model...")
        model = train.apply_model_fit()
    else:
        model = joblib.load(filename)
    return model

# Maps chosen seniority to artificial nominal variable
seniority_mapping = {
    'Junior': 1,
    'Mid-level': 1.5,
    'Senior': 3,
    'Lead/Principal': 3.5,
    'Manager/Director': 4
}

# Maps chosen degree to artificial nominal variable
degree_mapping = {
    'Bachelor\'s': 1,
    'Master\'s': 2,
    'PhD': 3,
    'None': 0
}

# Creates skill buckets recongnized by model fit
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
    senior_ml = seniority_score * has_ml
    senior_cloud = seniority_score * has_cloud

    input_df = pd.DataFrame([{
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
        'senior_cloud': float(senior_cloud),
        'senior_ml': float(senior_ml),
        'industry': industry,
        'company_size': company_size,
        'degree_score': degree_score
    }])
    return input_df

model = load_model()

def predict(title, location, seniority, skills_input, degree_choice, industry, company_size, years_exp):
    input_df = build_input_df(title, location, seniority, skills_input, degree_choice, industry, company_size, years_exp)

    X_transformed = model.named_steps['preprocess'].transform(input_df)
    tree_preds = [est.predict(X_transformed)[0] for est in model.named_steps['model'].estimators_]
    low, _, high = np.percentile(tree_preds, [25, 50, 75])
    prediction = model.predict(input_df)[0]

    gaps = skill_gap_analysis(input_df, prediction, ['has_cloud', 'has_ml', 'has_bigdata', 'has_viz', 'has_db'])

    return (f"Predicted Salary: ${round(prediction, 2)}\n50% Confidence Interval: (${round(low, 2)}, ${round(high, 2)})",
            tree_distribution(tree_preds, prediction),
            skill_gap_plot(gaps))


def tree_distribution(tree_preds, prediction):
    fig = px.histogram(tree_preds, nbins=20, title='Distribution of Tree Predictions', labels={'value': 'Predicted Salary'})
    fig.add_vline(x=prediction, line_dash="dash", line_color="red", annotation_text=f"Estimate: ${round(prediction, 2)}", annotation_position="top right")
    return fig
    # fig, ax = plt.subplots()
    # ax.hist(tree_preds, bins = 20, alpha = 0.7, color = 'b')
    # ax.axvline(prediction, color = 'r', linestyle = '--', label = f"Estimate: ${round(prediction, 2)}")
    # ax.set_xlabel('Predicted Salary')
    # ax.set_ylabel("Number of Trees")
    # ax.set_title('Distribution of Tree Predictions')
    # ax.legend()
    # fig.tight_layout()
    # return fig

def skill_gap_analysis(input_df, base_prediction, skill_cols):
    skill_gaps = {}
    for skill in skill_cols:
        if input_df[skill].iloc[0] == 0:
            modified_df = input_df.copy()
            modified_df[skill] = 1
            modified_prediction = model.predict(modified_df)[0]
            skill_gaps[skill] = modified_prediction - base_prediction
    return dict(sorted(skill_gaps.items(), key=lambda item: item[1], reverse=True))

def skill_gap_plot(skill_gaps):
    fig = px.bar(
        x = list(skill_gaps.keys()), 
        y = list(skill_gaps.values()),
        title='Estimated Salary Increase from Adding Each Missing Skill',
        labels={'x': 'Skill', 'y': 'Estimated Salary Increase'})
    fig.update_layout(xaxis_title='Skill', yaxis_title='Estimated Salary Increase', title_x=0.5, showlegend=False)
    return fig

with gr.Blocks() as demo:
    gr.Markdown("# Welcome to the Salary Prediction")
    with gr.Row():
        location = gr.Dropdown(choices = [
            'New York, NY',
            'San Francisco, CA',
            'Chicago, IL',
            'Seattle, WA',
            'Austin, TX'
        ], label= "Chose Location")

        title = gr.Dropdown(choices = [
            'data scientist',
            'data engineer',
            'data analyst',
            'machine learning engineer',
            'AI engineer',
            'business intelligence analyst',
            'analytics engineer'
        ], label = "Chose Job Title")

        seniority = gr.Dropdown(
            choices=['Junior', 'Mid-level', 'Senior', 'Lead/Principal', 'Manager/Director'],
            label="Seniority Level")
        
        with gr.Accordion('Advanced Options', open = False):
            degree_choice = gr.Dropdown(
                choices=['None', "Bachelor's", "Master's", 'PhD'],
                label="Degree Requirement",
                value="Bachelor's"
            )

            industry = gr.Dropdown(
                choices=['tech', 'finance', 'healthcare', 'government', 'retail'],
                label="Industry",
                value='tech'
            )
            years_exp = gr.Dropdown(
                choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, '10+'],
                label = 'Years of Experience',
                value = 'unknown'
            )

            company_size = gr.Dropdown(
                choices = ['large', 'startup', 'small'],
                label = 'Company Size',
                value = 'unknown'
            )

    with gr.Row():
        skills_input = gr.CheckboxGroup(
            choices = [
            # programming languages
            'python', 'r', 'sql', 'java', 'c++', 'scala', 'julia',
            # data manipulation
            'pandas', 'numpy', 'dplyr', 'data.table',
            # machine learning
            'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            # data visualization
            'matplotlib', 'seaborn', 'ggplot2', 'plotly',
            # big data tools
            'hadoop', 'spark', 'hive', 'pig', 'git',
            # cloud platforms
            'aws', 'azure', 'google cloud',
            # other tools
            'tableau', 'power bi', 'excel', 'jupyter',
            # databases
            'mysql', 'postgresql', 'mongodb', 'sqlite'
        ], label = "Skill Choices")

    analyze_but = gr.Button('Analyze')

    with gr.Row():
        salary_output = gr.Textbox(label = "Prediction", scale = 1)
    
    with gr.Row():
        dist_plot = gr.Plot()
        gap_plot = gr.Plot()



    analyze_but.click(
        fn = predict,
        inputs = [title, location, seniority, skills_input, degree_choice, industry, company_size, years_exp],
        outputs = [salary_output, dist_plot, gap_plot]
    )


demo.launch()