from internal.config.mongo_adapter import mongo_adapter, MongoSession

MongoSessionLocal = lambda: MongoSession(mongo_adapter)
