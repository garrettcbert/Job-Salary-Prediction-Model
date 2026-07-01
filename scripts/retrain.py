#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.scraping.scraper import scrape_all
from src.model.train import apply_model_fit


def main():
    parser = argparse.ArgumentParser(
        description='Scrape job postings and retrain the salary prediction model.'
    )
    parser.add_argument(
        '--skip-scrape',
        action='store_true',
        help='Skip scraping and retrain on existing database data only'
    )
    args = parser.parse_args()

    if not args.skip_scrape:
        print('Scraping jobs...')
        scrape_all()

    print('Training model...')
    apply_model_fit()
    print('Done. Model saved to models/salary_predictor.pkl')


if __name__ == '__main__':
    main()
