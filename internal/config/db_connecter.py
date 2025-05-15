from internal.config.mongo_adapter import MongoSession, mongo_adapter

MongoSessionLocal = lambda: MongoSession(mongo_adapter)
