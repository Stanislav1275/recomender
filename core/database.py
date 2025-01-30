import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
DB_HOST = os.getenv("RE_DB_HOST")
DB_NAME = os.getenv("RE_DB")
DB_USER = os.getenv("RE_DB_USER")
DB_PASSWORD = os.getenv("RE_DB_PASSWORD")
DB_PORT = os.getenv("RE_DB_PORT")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()