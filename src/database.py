from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, UniqueConstraint
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
DB_PATH = Path(__file__).parent.parent / 'data' / 'jobs.db'

class Job(Base):
    __tablename__ = 'Jobs'

    id = Column(Integer, primary_key=True)
    searched_title = Column(String)
    searched_location = Column(String)
    title = Column(String)
    company = Column(String)
    location = Column(String)
    date_posted = Column(DateTime)
    description = Column(Text)
    site = Column(String)
    min_amount = Column(Float)
    max_amount = Column(Float)
    interval = Column(String)

    __table_args__ = (
        UniqueConstraint('title', 'company', 'searched_title', 'searched_location', name='uq_job'),
    )

def get_engine():
    DB_PATH.parent.mkdir(parents = True, exist_ok = True)
    return create_engine(f'sqlite:///{DB_PATH}')

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine

def save_jobs(jobs):
    engine = get_engine()
    Session = sessionmaker(bind = engine)
    session = Session()

    try:
        for _, job in jobs.iterrows():
            job_entry = insert(Job).values(
                searched_title = job['searched_title'],
                searched_location = job['searched_location'],
                title = job['title'],
                company = job['company'],
                location = job['location'],
                date_posted = job['date_posted'],
                description = job['description'],
                site = job['site'],
                min_amount = job["min_amount"],
                max_amount = job["max_amount"],
                interval = job['interval']
            ).prefix_with('OR IGNORE')
            session.execute(job_entry)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
