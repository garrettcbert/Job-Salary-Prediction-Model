import pandas as pd
import re

skills = [
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
    'aws', 'azure', 'google cloud', 'gcp',
    # other tools
    'tableau', 'power bi', 'excel', 'jupyter',
    # databases
    'mysql', 'postgresql', 'mongodb', 'sqlite'
]

# Normalizing salary to a yearly measure
def normalize_salary(row):
    salary_min = row['min_amount']
    salary_max = row['max_amount']
    interval = row['interval'] # Interval is pay interval from scraped data
    if pd.isna(salary_min) and pd.isna(salary_max):
        return None
    if interval == 'monthly':
        if not pd.isna(salary_min):
            salary_min *= 12
        if not pd.isna(salary_max):
            salary_max *= 12
    elif interval == 'weekly':
        if not pd.isna(salary_min):
            salary_min *= 52
        if not pd.isna(salary_max):
            salary_max *= 52
    elif interval == 'hourly':
        if not pd.isna(salary_min):
            salary_min *= 40 * 52
        if not pd.isna(salary_max):
            salary_max *= 40 * 52
    
    if salary_min is not None and salary_max is not None:
        salary_avg = (salary_min + salary_max) / 2
    elif salary_min is not None:
        salary_avg = salary_min
    elif salary_max is not None:
        salary_avg = salary_max
    else:
        salary_avg = None

    return salary_avg

# Extract raw skills from description
def extract_skills(row):
    description = row['description']
    
    description = description.lower()
    found_skills = []
    for skill in skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, description):
            found_skills.append(skill)
    
    return found_skills

# Group skills based on type
def get_skills(found_skills):
    skills = found_skills
    return pd.Series({
        'has_cloud':int(any(s in skills for s in ['aws', 'azure', 'google cloud', 'gcp'])),
        'has_ml': int(any(s in skills for s in ['pytorch', 'tensorflow', 'scikit-learn', 'keras'])),
        'has_bigdata': int(any(s in skills for s in ['spark', 'hadoop', 'hive', 'kafka'])),
        'has_viz': int(any(s in skills for s in ['tableau', 'power bi', 'plotly'])),
        'has_db': int(any(s in skills for s in ['postgresql', 'mysql', 'mongodb', 'snowflake']))
    })

# Extract the role type/seniority from title
def get_seniority(row):
    title = row['title']
    title = title.lower()
    
    if any(word in title for word in ['intern', 'internship', 'co-op']):
        return 0
    elif any(word in title for word in ['jr', 'junior', 'entry', 'associate']):
        return 1
    elif any(word in title for word in ['senior', 'sr']):
        return 3
    elif any(word in title for word in ['lead', 'principal']):
        return 3.5
    elif any(word in title for word in ['manager', 'director', 'head', 'vp']):
        return 4
    else:
        return 1.5 # Default is assumed to be between junior and senior level

# Extract the approximate hiring company's size from description
def get_company_size_proxy(row):
    description = str(row['description']).lower()
    if any(word in description for word in ['fortune 500', 'global', 'worldwide', 'multinational']):
        return 'large'
    elif any(word in description for word in ['series a', 'series b', 'startup', 'seed']):
        return 'startup'
    elif any(word in description for word in ['small team', 'growing team', 'early stage']):
        return 'small'
    else:
        return 'unknown'

# Extract type of industry the job is in from description
def get_industry(row):
    description = str(row['description']).lower()
    if any(word in description for word in ['healthcare', 'medical', 'clinical', 'pharma']):
        return 'healthcare'
    elif any(word in description for word in ['finance', 'banking', 'investment', 'trading']):
        return 'finance'
    elif any(word in description for word in ['retail', 'ecommerce', 'consumer']):
        return 'retail'
    elif any(word in description for word in ['defense', 'government', 'federal']):
        return 'government'
    else:
        return 'tech'

# Extract required or suggested degree for role from description
def get_degree_requirements(row):
    description = str(row['description']).lower()
    if any(word in description for word in ['phd', 'ph.d', 'doctorate', 'doctoral']):
        return 3
    elif any(word in description for word in ['masters', 'master\'s', 'mba', 'm.s.', 'mse', 'msc', 'graduate degree']):
        return 2
    elif any(word in description for word in ['bachelor\'s', 'bachelors', 'b.s.', 'b.a.', 'undergraduate', 'college degree']):
        return 1
    else:
        return 0

# Extract the required years experience
# ~ 80% are Not Found
def get_years_experience(row):
    description = str(row['description']).lower()
    if any(word in description for word in ['10 years', '10+ years']):
        return 10
    elif any(word in description for word in ['7 years', '7+ years']):
        return 7
    elif any(word in description for word in ['5 years', '5+ years']):
        return 5
    elif any(word in description for word in ['3 years', '3+ years']):
        return 3
    elif any(word in description for word in ['1 year', '1+ years']):
        return 1
    else:
        return None # Reported as the median in apply_preprocess function
    
# Removes outliers with standard formula
def remove_outliers(df):
    Q1 = df['salary_avg'].quantile(0.25)
    Q3 = df['salary_avg'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['salary_avg'] >= Q1 - 1.5 * IQR) & (df['salary_avg'] <= Q3 + 1.5 * IQR)]
    return df

# Apply all preprocessing functions
def apply_preprocess(df):
    df = df.copy()
    df['salary_avg'] = df.apply(normalize_salary, axis = 1)
    df = df.dropna(subset = ['salary_avg']) # Drop rows with missing salaries

    df['skills'] = df.apply(extract_skills, axis = 1)

    df['seniority_score'] = df.apply(get_seniority, axis = 1) * 3
    df = df[df['seniority_score'] > 0] # Drop internship roles since they are not comparable to other roles

    df['years_exp'] = df.apply(get_years_experience, axis = 1)
    df['years_exp'] = df['years_exp'].fillna(df['years_exp'].median())

    df['skill_count'] = df['skills'].apply(len)
    df[['has_cloud', 'has_ml', 'has_bigdata', 'has_viz', 'has_db']] = df['skills'].apply(get_skills)
    df['senior_cloud'] = df['seniority_score'] * df['has_cloud'] # Interaction variable
    df['senior_ml'] = df['seniority_score'] * df['has_ml'] # Interaction variable

    df['industry'] = df.apply(get_industry, axis = 1)
    df['company_size'] = df.apply(get_company_size_proxy, axis = 1)

    df['degree_score'] = df.apply(get_degree_requirements, axis = 1)

    df = remove_outliers(df)

    return df[['title', 'location', 'searched_title', 'searched_location', 'skills', 'has_cloud', 'has_ml', \
               'has_bigdata', 'has_viz', 'has_db', 'skill_count', 'seniority_score', 'years_exp', 'senior_cloud', \
                'senior_ml', 'industry', 'company_size', 'degree_score', 'salary_avg']]