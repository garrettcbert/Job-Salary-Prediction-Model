---
title: Salary-Predictor
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.8.0"
app_file: app/app.py
pinned: false
---

# Salary Predictor

Predicts salaries for data-related roles (Data Scientist, Analyst, ML Engineer, etc.) based on job title, location, seniority, skills, and other attributes. Model trained on real job postings scraped from major job boards.

**[Try the live demo](#)** *(add deployment link)*

## Features

- Salary estimate with a visual confidence gauge (built from 200 individual XGBoost tree predictions)
- Skill gap analysis — shows which missing skills would increase your predicted salary the most
- Location and job title comparison charts
- Interactive Gradio UI with categorized skill selection

## How It Works

1. **Scraping** — `scraper.py` pulls job postings (title, location, salary, description) using `jobspy`
2. **Preprocessing** — Extracts structured features from raw postings: seniority level, required skills, degree requirements, years of experience, industry, and company size (parsed from free-text descriptions via keyword matching)
3. **Feature Engineering** — Builds interaction terms (e.g. seniority × has_cloud) to capture how skill value scales with experience
4. **Model Training** — XGBoost pipeline with one-hot encoding for categorical features and standard scaling for numeric features
5. **Prediction** — The Gradio app loads the trained pipeline and returns a salary estimate, confidence range, skill gap recommendations, and cross-location/title comparisons

## Project Structure

```
Job_Scraping_Project/
├── data/
│   └── Jobs.db                    # scraped job postings (SQLite)
├── src/
│   ├── database.py                # DB schema and connection
│   ├── queries.py                 # SQL helpers
│   ├── scraping/
│   │   └── scraper.py             # job scraping logic
│   ├── preprocessing/
│   │   └── preprocess.py          # feature extraction
│   └── model/
│       ├── train.py               # pipeline building and training
│       ├── predict.py             # prediction helpers
│       └── evaluate.py            # cross-validation
├── models/
│   ├── salary_predictor.pkl       # trained model
│   └── metadata.json              # training run metadata
├── app/
│   └── app.py                     # Gradio UI
├── scripts/
│   └── retrain.py                 # scrape + retrain pipeline
├── launch.py                      # run the app from project root
└── requirements.txt
```

## Setup (Local)

```bash
git clone https://github.com/garrettcbert/Job-Salary-Prediction-Model.git
cd Job_Scraping_Project
pip install -r requirements.txt

# Scrape job postings and train the model (required before first run)
python scripts/retrain.py

# Launch the app
python launch.py
```

To retrain on fresh data at any time:
```bash
python scripts/retrain.py            # scrape + retrain
python scripts/retrain.py --skip-scrape  # retrain on existing data only
```

## Tech Stack

- **Scraping:** jobspy
- **Data:** SQLite + SQLAlchemy
- **ML:** scikit-learn, XGBoost
- **UI:** Gradio
- **Visualization:** Plotly

## Limitations

- Skill/industry/seniority extraction relies on keyword matching from job descriptions, which is an approximation and may misclassify some postings
- Salary data quality varies by source (some ranges are estimates rather than exact figures)
- Limited to U.S. job postings in select metro areas
