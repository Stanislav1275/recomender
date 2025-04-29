# Рекомендательный сервис на базе LightFM

Этот сервис предоставляет персонализированные рекомендации контента на основе данных пользователей, используя алгоритм LightFM и набор различных типов взаимодействий (рейтинги, просмотры, закладки и т.д.).

## Особенности

- Использование LightFM для создания гибридной рекомендательной системы с учетом пользовательских и контентных характеристик
- Кеширование рекомендаций в Redis для ускорения доступа
- История обучения моделей с возможностью переключения между версиями
- Планирование заданий на обучение
- Административные методы для управления моделями
- Автоматическое ежедневное обучение модели (по умолчанию в 03:00)

## Архитектура сервиса

Сервис предоставляет два интерфейса:
- **REST API** (FastAPI) для удобного доступа к рекомендациям и административных функций
- **gRPC** интерфейс для высокоэффективного доступа из других сервисов

### Основные компоненты:

1. **ModelManager** - менеджер модели LightFM, отвечающий за:
   - Инициализацию и обучение модели
   - Сохранение и загрузку моделей
   - Частичное дообучение на новых данных
   - Планирование заданий на обучение

2. **DataPrepareService** - сервис подготовки данных, отвечающий за:
   - Извлечение данных из MySQL
   - Предобработку и агрегацию данных
   - Фильтрацию неподходящих данных

3. **RecService** - сервис рекомендаций, предоставляющий методы:
   - Получение персональных рекомендаций для пользователя
   - Получение релевантных тайтлов для конкретного тайтла
   - Обновление модели через API

4. **BlacklistManager** - менеджер черного списка, исключающий недопустимый контент из рекомендаций

5. **RecommendationCache** - сервис кеширования рекомендаций в Redis

6. **ModelRegistry** - реестр моделей и истории обучения в SQLite

## API Endpoints

### REST API (FastAPI)

#### Рекомендации
- `GET /titles/recommendations?user_id={id}` - получение персональных рекомендаций для пользователя
- `GET /titles/relavant?title_id={id}` - получение похожих тайтлов
- `GET /rec/hot/interact?user_id={id}&title_id={id}&int_type={type}&raw_score={score}` - добавление нового взаимодействия
- `GET /health` - проверка состояния сервиса

#### Администрирование
- `GET /admin/train` - запуск обучения модели с указанными параметрами
- `GET /admin/schedule-training` - планирование обучения модели
- `GET /admin/models` - получение списка моделей
- `GET /admin/models/{model_id}` - получение информации о модели
- `POST /admin/models/{model_id}/activate` - активация модели
- `GET /admin/jobs` - получение списка заданий на обучение
- `GET /admin/jobs/{job_id}` - получение информации о задании
- `POST /admin/cache/invalidate` - инвалидация кеша рекомендаций

### gRPC API

- `GetUserRecommendations` - получение персональных рекомендаций
- `GetTitleRelevant` - получение похожих тайтлов
- `TrainModel` - запуск переобучения модели

## Хранение данных

1. **Redis** - используется для кеширования рекомендаций с TTL (время жизни) 1 час
2. **SQLite** - используется для хранения метаданных моделей и заданий на обучение 
3. **Файловая система** - используется для хранения файлов моделей и датасетов
4. **MySQL** (только для чтения) - основная база данных с пользовательскими данными и контентом

## Запуск сервиса

### Через Docker Compose

```bash
docker-compose up -d
```

### Локально

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Установите Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS (с использованием Homebrew)
brew install redis
```

3. Запустите Redis:
```bash
redis-server
```

4. Запустите сервис:
```bash
python main.py
```

## Структура проекта

- `main.py` - основной файл с FastAPI сервером
- `grpc_server.py` - gRPC сервер
- `models.py` - модели данных SQLAlchemy
- `recommendations/` - директория с основным кодом рекомендательной системы
  - `data_preparer.py` - сервис подготовки данных
  - `rec_service.py` - сервис рекомендаций
  - `rec_server.py` - реализация gRPC сервера
  - `cache_service.py` - сервис кеширования рекомендаций
  - `model_registry.py` - реестр моделей и истории обучения
  - `protos/` - protobuf файлы для gRPC
  - `config.py` - конфигурация системы рекомендаций
- `data/` - директория для хранения моделей и датасетов
  - `models/` - директория с файлами моделей
  - `model_registry.db` - база данных SQLite с метаданными моделей

## Параметры LightFM

При обучении модели можно указать следующие параметры:

- `no_components` - размерность векторов представления (факторов), по умолчанию 100
- `loss` - функция потерь, доступные варианты:
  - `bpr` (Bayesian Personalized Ranking) - оптимизирует ранжирование, хорошо работает для неявных взаимодействий
  - `warp` (Weighted Approximate-Rank Pairwise) - оптимизирует точность@k, лучше для ранжирования
  - `logistic` - логистическая функция потерь, для явных оценок
  - `warp-kos` (k-OS WARP) - вариант WARP с использованием k-го порядка статистики
- `epochs` - количество эпох обучения, по умолчанию 30
- `item_alpha` - L2 регуляризация для элементов, по умолчанию 0.0
- `user_alpha` - L2 регуляризация для пользователей, по умолчанию 0.0

## Использование

### Пример получения рекомендаций:

```python
import requests

# REST API
response = requests.get("http://localhost:8000/titles/recommendations?user_id=123")
recommendations = response.json()

# gRPC
import grpc
from recommendations.protos import recommendations_pb2_grpc, recommendations_pb2

channel = grpc.insecure_channel('localhost:50051')
stub = recommendations_pb2_grpc.RecommenderStub(channel)
response = stub.GetUserRecommendations(recommendations_pb2.UserRequest(user_id=123))
recommendations = response.item_ids
```

### Пример запуска обучения модели с кастомными параметрами:

```python
import requests

response = requests.get(
    "http://localhost:8000/admin/train",
    params={
        "no_components": 150,
        "loss": "warp",
        "epochs": 50,
        "item_alpha": 0.001,
        "user_alpha": 0.001
    }
)
result = response.json()
print(f"Модель обучена, ID: {result['model_id']}")
```

### Пример планирования обучения:

```python
import requests
from datetime import datetime, timedelta

scheduled_time = (datetime.now() + timedelta(hours=1)).isoformat()

response = requests.get(
    "http://localhost:8000/admin/schedule-training",
    params={
        "scheduled_at": scheduled_time,
        "no_components": 120,
        "loss": "bpr"
    }
)
result = response.json()
print(f"Обучение запланировано, ID задания: {result['job_id']}")
```
