import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
DB_HOST = os.getenv("RE_DB_HOST")
DB_NAME = os.getenv("RE_DB")
DB_USER = os.getenv("RE_DB_USER")
DB_PASSWORD = os.getenv("RE_DB_PASSWORD")
DB_PORT = os.getenv("RE_DB_PORT", "3306")

# Формирование URL для подключения к базе данных (только для чтения)
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Создание движка SQLAlchemy с параметрами только для чтения
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    connect_args={"read_only": True},  # Подключение только для чтения
)

# Создание фабрики сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db_session():
    """
    Создает сессию для работы с базой данных и гарантирует ее закрытие.
    Используется как контекстный менеджер или в FastAPI Depends.
    """
    db = SessionLocal()
    logger.debug("Создана новая сессия базы данных (только для чтения)")
    try:
        yield db
    finally:
        db.close()
        logger.debug("Сессия базы данных закрыта")