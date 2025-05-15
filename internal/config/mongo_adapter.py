import os
from contextlib import contextmanager
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional, Any, Dict, List

load_dotenv()

class MongoAdapter:
    def __init__(self):
        self.host = os.getenv("MONGO_HOST", "localhost")
        self.port = int(os.getenv("MONGO_PORT", "27017"))
        self.user = os.getenv("MONGO_USER", "admin")
        self.password = os.getenv("MONGO_PASSWORD", "password")
        self.db_name = os.getenv("MONGO_DB_NAME", "recommender_db")
        
        self.client = MongoClient(f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/")
        self.db = self.client[self.db_name]
    
    def get_collection(self, collection_name: str):
        return self.db[collection_name]
    
    def close(self):
        self.client.close()

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
def get_mongo_session():
    session = MongoSession(mongo_adapter)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Для совместимости с SQLAlchemy стилем
