import os
from contextlib import contextmanager
from typing import Optional, Any, Dict, List
import logging
from motor.motor_asyncio import AsyncIOMotorClient

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Глобальные переменные для хранения клиента и базы данных
_mongo_client: Optional[AsyncIOMotorClient] = None
_db_name = os.getenv("MONGO_DB_NAME", "recommender_db")

async def get_mongo_client() -> AsyncIOMotorClient:
    """Получить или создать клиент MongoDB"""
    global _mongo_client
    
    if _mongo_client is None:
        host = os.getenv("MONGO_HOST", "localhost")
        port = int(os.getenv("MONGO_PORT", "27017"))
        user = os.getenv("MONGO_USER", "admin")
        password = os.getenv("MONGO_PASSWORD", "password")
        
        connection_string = f"mongodb://{user}:{password}@{host}:{port}"
        logger.info(f"MongoDB connection string: {connection_string}")
        
        _mongo_client = AsyncIOMotorClient(connection_string)
    
    return _mongo_client

async def get_database():
    """Получить базу данных"""
    client = await get_mongo_client()
    return client[_db_name]

async def close_mongo_connection():
    """Закрыть соединение с MongoDB"""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None

class MongoAdapter:
    def __init__(self):
        self.host = os.getenv("MONGO_HOST", "localhost")
        self.port = int(os.getenv("MONGO_PORT", "27017"))
        self.user = os.getenv("MONGO_USER", "admin")
        self.password = os.getenv("MONGO_PASSWORD", "password")
        self.db_name = os.getenv("MONGO_DB_NAME", "recommender_db")
        
        connection_string = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}"
        logger.info(f"MongoDB connection string: {connection_string}")
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[self.db_name]
    
    async def get_collection(self, collection_name: str):
        return self.db[collection_name]
    
    async def close(self):
        await self.client.close()

class MongoSession:
    def __init__(self, adapter: MongoAdapter):
        self.adapter = adapter
        self.db = adapter.db
        self._operations = []
    
    def add(self, collection_name: str, document: Dict[str, Any]) -> Any:
        self._operations.append(('insert', collection_name, document))
        return document
    
    def query(self, collection_name: str) -> 'MongoQuery':
        return MongoQuery(self.db[collection_name])
    
    def commit(self):
        for operation, collection_name, data in self._operations:
            if operation == 'insert':
                self.db[collection_name].insert_one(data)
            elif operation == 'update':
                filter_data, update_data = data
                self.db[collection_name].update_one(filter_data, update_data)
            elif operation == 'delete':
                self.db[collection_name].delete_one(data)
        self._operations.clear()
    
    def rollback(self):
        self._operations.clear()
    
    def close(self):
        pass

class MongoQuery:
    def __init__(self, collection):
        self.collection = collection
        self._filter = {}
        self._limit_count = None
        self._sort_params = None
    
    def filter(self, **kwargs) -> 'MongoQuery':
        self._filter.update(kwargs)
        return self
    
    def filter_by(self, **kwargs) -> 'MongoQuery':
        self._filter.update(kwargs)
        return self
    
    def limit(self, count: int) -> 'MongoQuery':
        self._limit_count = count
        return self
    
    def order_by(self, field: str, direction: int = 1) -> 'MongoQuery':
        self._sort_params = (field, direction)
        return self
    
    def first(self) -> Optional[Dict[str, Any]]:
        return self.collection.find_one(self._filter)
    
    def all(self) -> List[Dict[str, Any]]:
        cursor = self.collection.find(self._filter)
        if self._sort_params:
            cursor = cursor.sort(self._sort_params[0], self._sort_params[1])
        if self._limit_count:
            cursor = cursor.limit(self._limit_count)
        return list(cursor)
    
    def count(self) -> int:
        return self.collection.count_documents(self._filter)

# Создание глобального адаптера и фабрики сессий
mongo_adapter = MongoAdapter()

@contextmanager
async def get_mongo_session():
    session = MongoSession(mongo_adapter)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
