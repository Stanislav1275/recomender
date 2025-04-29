# Сервис рекомендаций

Сервис рекомендаций предоставляет API для получения персонализированных рекомендаций и похожих элементов на основе модели LightFM.

## Особенности

- gRPC API для получения рекомендаций и управления моделями
- Аутентификация и авторизация с использованием API-токенов
- Ограничение доступа по IP-адресам
- Кэширование результатов для быстрой выдачи рекомендаций
- Режим "только чтение" для базы данных (безопасная работа)
- Планировщик заданий для автоматического обучения моделей
- Поддержка различных метрик для оценки качества моделей

## Требования

- Python 3.9+
- PostgreSQL
- Redis для кэширования
- [rectools](https://rectools.readthedocs.io/) и LightFM

## Установка

1. Клонировать репозиторий
2. Установить зависимости:

```bash
pip install -r recommendations/requirements.txt
```

3. Скомпилировать proto-файлы:

```bash
cd recommendations
python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/recommendation_service.proto
```

4. Настроить переменные окружения (см. раздел "Настройка")

## Запуск

### Локально

```bash
python -m recommendations.main
```

### Docker

```bash
docker build -t rec-service -f recommendations/Dockerfile .
docker run -p 50051:50051 -e REC_DB_URL="postgresql+asyncpg://user:password@db:5432/rec_db" rec-service
```

## Настройка

Сервис настраивается через переменные окружения:

### Основные настройки

- `REC_DB_URL` - URL подключения к PostgreSQL (формат: `postgresql+asyncpg://user:password@host:port/db_name`)
- `REC_GRPC_HOST` - Хост для сервера gRPC (по умолчанию: "0.0.0.0")
- `REC_GRPC_PORT` - Порт для сервера gRPC (по умолчанию: 50051)
- `REC_CACHE_URL` - URL подключения к Redis (формат: `redis://host:port/db`)

### Настройки безопасности

- `REC_ADMIN_TOKEN` - Токен для административного доступа
- `REC_USER_TOKEN` - Токен для пользовательского доступа
- `REC_ALLOWED_IPS` - Список разрешенных IP-адресов (через запятую, поддерживаются CIDR-нотации, например: "192.168.1.0/24,10.0.0.1")

## API gRPC

Сервис предоставляет следующие методы:

### Пользовательские методы (токен пользователя)

- `GetRecommendations` - Получение персональных рекомендаций для пользователя
- `GetSimilarItems` - Получение похожих элементов для заданного элемента
- `GetUserRecentInteractions` - Получение последних взаимодействий пользователя

### Административные методы (токен администратора)

- `TrainModel` - Запуск обучения модели
- `GetModelInfo` - Получение информации о модели
- `ListModels` - Получение списка моделей
- `SetActiveModel` - Установка активной модели

## Использование клиента

Пример кода для клиента:

```python
import grpc
from recommendations.proto import recommendation_service_pb2, recommendation_service_pb2_grpc

# Установка соединения
channel = grpc.insecure_channel('localhost:50051')
stub = recommendation_service_pb2_grpc.RecommendationServiceStub(channel)

# Добавление метаданных для авторизации
metadata = [('authorization', 'YOUR_TOKEN')]

# Получение рекомендаций
request = recommendation_service_pb2.UserRequest(user_id=123, limit=10, filter_viewed=True)
response = stub.GetRecommendations(request, metadata=metadata)

print("Рекомендованные элементы:", response.item_ids)
```

## Описание модели

Сервис использует LightFM с оберткой из rectools для обучения модели. Модель учитывает:

- Взаимодействия пользователей (просмотры, лайки)
- Характеристики пользователей (возраст, пол, предпочтения)
- Характеристики элементов (жанры, категории, рейтинги)

## Безопасность

- Вся работа с базой данных осуществляется в режиме "только чтение"
- Административные методы API защищены отдельным токеном
- Доступ может быть ограничен по IP-адресам

## Лицензия

[MIT License](LICENSE) 