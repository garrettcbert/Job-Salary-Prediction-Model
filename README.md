# Salary Predictor

Predicts salaries for data-related roles (Data Scientist, Analyst, etc.) based on job title, seniority, skills, and other attributes. Model trained on real job postings scraped from major job boards.

**[Try the live demo](#)** *(add deployment link)*

![App Screenshot]

## Features

 - Predicts salary with a confidence interval (using individual tree predictions)
 - Skill gap analysis - shows which missing skills would increase your predicted salary 
 - Interactive Gradio UI with categorized skill selection
 - XGBoost model trained on scraped job postings

## How It Works

1. **Scraping** — `scraper.py` pulls job postings (title, location, salary, description) using `jobspy`
2. **Preprocessing** — Extracts structured features from raw postings: seniority level,
   required skills, degree requirements, years of experience, industry, and company size
   (parsed from free-text descriptions using keyword matching)
3. **Feature Engineering** — Builds interaction terms (e.g. seniority × has_cloud) to
   capture how skill value scales with experience level
4. **Model Training** — An XGBoost/Random Forest pipeline with one-hot encoding for
   categorical features and standard scaling for numeric features
5. **Prediction** — The Gradio app loads the trained pipeline and returns a salary
   estimate, confidence range, and skill recommendations

## Project Structure

```
job_salary_predictor/
├── data/
│   └── jobs.db                 # scraped job postings (SQLite)
├── src/
│   ├── database.py             # DB models and queries
│   ├── scraping/
│   │   └── scraper.py          # job scraping logic
│   ├── processing/
│   │   └── preprocess.py       # feature extraction
│   └── model/
│       └── train.py            # pipeline building and training
├── models/
│   └── salary_predictor.pkl    # trained model (generated)
├── app/
│   └── gradio_app.py           # Gradio UI
└── requirements.txt
```

## Setup

```bash
# clone and install dependencies
git clone <your-repo-url>
cd job_salary_predictor
pip install -r requirements.txt

# scrape job postings (populates data/jobs.db)
python src/scraping/scraper.py

# train the model (saves models/salary_predictor.pkl)
python src/model/train.py

# launch the app
python app/gradio_app.py
```

If `models/salary_predictor.pkl` doesn't exist, the app will train automatically on launch.

## Tech Stack

- **Scraping:** jobspy
- **Data:** SQLite + SQLAlchemy
- **ML:** scikit-learn, XGBoost
- **UI:** Gradio
- **Visualization:** Plotly

## Limitations

- Skill/industry/seniority extraction relies on keyword matching from job descriptions,
  which is an approximation and may misclassify some postings
- Salary data quality varies by source/posting (some ranges are estimates rather than
  exact figures)
- Limited to U.S. job postings in select metro areas