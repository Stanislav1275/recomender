import os
from mailbox import ExternalClashError

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()
DB_HOST = os.getenv("RE_DB_HOST")
DB_NAME = os.getenv("RE_DB")
DB_USER = os.getenv("RE_DB_USER")
DB_PASSWORD = os.getenv("RE_DB_PASSWORD")
DB_PORT = os.getenv("RE_DB_PORT")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
external_engine = create_engine(DATABASE_URL)
ExternalClashErrorSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=external_engine)
ExternalBase = declarative_base()

@contextmanager
def get_external_session():
    """Контекстный менеджер для работы с сессией внешней БД"""
    db = ExternalClashErrorSessionLocal()
    try:
        yield db
    finally:
        db.close()