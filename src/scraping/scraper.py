from jobspy import scrape_jobs
from pathlib import Path
import sys 

sys.path.append(str(Path(__file__).parent.parent))

from database import save_jobs, init_db
import pandas as pd

TITLES = [
    'data scientist',
    'data engineer',
    'data analyst',
    'machine learning engineer',
    'AI engineer',
    'business intelligence analyst',
    'analytics engineer'
]

LOCATIONS = [
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
]

def scrape_all():
    init_db()
    all_jobs = []

    for title in TITLES:
        for location in LOCATIONS:
            print(f'Scraping for {title} in {location}...')
            jobs = scrape_jobs(
                site_name = ['indeed'],
                search_term = title,
                location = location,
                results_wanted = 100
            )
            jobs['searched_title'] = title
            jobs['searched_location'] = location

            all_jobs.append(jobs)
    
    combined = pd.concat(all_jobs, ignore_index = True)
    combined = combined.drop_duplicates(subset = ['title', 'company', 'location'])
    save_jobs(combined)
    prop_without_salary = combined['min_amount'].isna().mean()
    
    print(f'Scraped and saved {len(combined)} unique job listings.')
    print(f'Proportion of listings without salary info: {prop_without_salary:.2%})')

scrape_all()