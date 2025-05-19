// Создаем пользователя для внешнего доступа
db = db.getSiblingDB('admin');

// Создаем пользователя с полными правами для внешнего доступа
db.createUser({
  user: 'recommender_user',
  pwd: 'recommender_password',
  roles: [
    {
      role: 'readWriteAnyDatabase',
      db: 'admin'
    },
    {
      role: 'dbAdminAnyDatabase', 
      db: 'admin'
    }
  ]
});

// Переключаемся на нашу базу
db = db.getSiblingDB('recommender_db');

// === КОНФИГУРАЦИИ (Configuration) ===
db.createCollection('configurations');
// Индексы не указаны в мета, добавляем базовые
db.configurations.createIndex({ name: 1 }, { unique: true });
db.configurations.createIndex({ site_id: 1 });
db.configurations.createIndex({ created_at: -1 });

// === РЕКОМЕНДАТЕЛЬНЫЕ КОНФИГУРАЦИИ (RecommendationConfig) ===
db.createCollection('recommendation_configs', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["name", "title_field_filters", "is_active", "created_at", "updated_at"],
            properties: {
                name: {
                    bsonType: "string",
                    description: "Название конфигурации"
                },
                description: {
                    bsonType: ["string", "null"],
                    description: "Описание конфигурации"
                },
                title_field_filters: {
                    bsonType: "array",
                    items: {
                        bsonType: "object",
                        required: ["field_name", "operator", "values", "is_active"],
                        properties: {
                            field_name: {
                                bsonType: "string",
                                description: "Имя поля для фильтрации"
                            },
                            operator: {
                                bsonType: "string",
                                description: "Оператор сравнения"
                            },
                            values: {
                                bsonType: "array",
                                description: "Значения для сравнения"
                            },
                            is_active: {
                                bsonType: "bool",
                                description: "Активен ли фильтр"
                            }
                        }
                    }
                },
                is_active: {
                    bsonType: "bool",
                    description: "Активна ли конфигурация"
                },
                created_at: {
                    bsonType: "date",
                    description: "Дата создания"
                },
                updated_at: {
                    bsonType: "date",
                    description: "Дата обновления"
                }
            }
        }
    }
});

// Индексы из meta модели
db.recommendation_configs.createIndex({ name: 1 }, { unique: true });
db.recommendation_configs.createIndex({ is_active: 1 });
db.recommendation_configs.createIndex({ created_at: -1 });
db.recommendation_configs.createIndex({ 'schedules_dates.next_run': 1 });

// === ЛОГИ ВЫПОЛНЕНИЯ КОНФИГУРАЦИЙ (ConfigExecutionLog) ===
db.createCollection('config_execution_logs');
// Индексы из meta модели
db.config_execution_logs.createIndex({ config_id: 1 });
db.config_execution_logs.createIndex({ executed_at: -1 });
db.config_execution_logs.createIndex({ status: 1 });

// === МОДЕЛИ ===
db.createCollection('models');
db.models.createIndex({ config_id: 1, created_at: -1 });
db.models.createIndex({ config_id: 1, is_latest: 1 });

// === ЗАПУСКИ ОБУЧЕНИЯ ===
db.createCollection('runs');
db.runs.createIndex({ config_id: 1, start_time: -1 });
db.runs.createIndex({ status: 1 });

// === МЕТРИКИ ===
db.createCollection('metrics');
db.metrics.createIndex({ model_id: 1 });
db.metrics.createIndex({ run_id: 1 });

// === ОТСЛЕЖИВАЕМЫЕ ПОЛЬЗОВАТЕЛИ ===
db.createCollection('observable_users');
db.observable_users.createIndex({ config_id: 1 });
db.observable_users.createIndex({ user_id: 1 });

// === САЙТЫ ===
db.sites.createIndex({ external_id: 1 }, { unique: true });
db.sites.createIndex({ name: 1 });

db.sites.insertMany([
  {
    external_id: 1,
    name: "main_platform",
    created_at: new Date()
  }
]);

db.recommendation_configs.insertOne({
  name: "default_recommendation_config",
  description: "Базовая конфигурация рекомендательной системы - Remanga",
  is_active: true,
  title_field_filters: [
    {
      field_name: "is_erotic",
      operator: "not_equals",
      values: [false],
      is_active: true
    },
    {
      field_name: "is_legal",
      operator: "equals", 
      values: [true],
      is_active: true
    },
    {
      field_name: "site_id",
      operator: "in",
      values: [1, 3, 6],
      is_active: true
    }
  ],
  schedules_dates: [
    {
      type: "once_day",
      date_like: "04:00",
      is_active: true,
      next_run: null
    }
  ],
  created_at: new Date(),
  updated_at: new Date()
});

db.recommendation_configs.insertOne({
  name: "default_recommendation_config_novels",
  description: "Базовая конфигурация рекомендательной системы - Книги",
  is_active: true,
  title_field_filters: [
    {
      field_name: "is_erotic",
      operator: "not_in",
      values: [true],
      is_active: true
    },
    {
      field_name: "is_legal",
      operator: "in", 
      values: [true],
      is_active: true
    },
    {
      field_name: "site_id",
      operator: "in",
      values: [2],
      is_active: true
    }
  ],
  schedules_dates: [
    {
      type: "once_day",
      date_like: "04:00",
      is_active: true,
      next_run: null
    }
  ],
  created_at: new Date(),
  updated_at: new Date()
});

print();