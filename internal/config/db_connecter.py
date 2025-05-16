from internal_db import mongo_adapter


MongoSessionLocal = lambda: mongo_adapter.MongoSession(mongo_adapter) 
