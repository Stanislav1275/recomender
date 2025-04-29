FROM python:3.9-slim

WORKDIR /app

# Устанавливаем зависимости для компиляции
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файл с зависимостями сначала, чтобы использовать кеширование Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Компилируем proto-файлы
RUN python -m grpc_tools.protoc -I./proto --python_out=./proto --grpc_python_out=./proto ./proto/recommendation_service.proto
RUN touch ./proto/__init__.py
RUN touch ./proto/recommendation_service_pb2_grpc.py

# Переменные окружения по умолчанию
ENV REC_DB_URL="postgresql+asyncpg://user:password@db:5432/rec_db"
ENV REC_GRPC_HOST="0.0.0.0"
ENV REC_GRPC_PORT="50051"
ENV REC_CACHE_URL="redis://redis:6379/0"
ENV REC_ADMIN_TOKEN=""
ENV REC_USER_TOKEN=""
ENV REC_ALLOWED_IPS=""

# Открываем порт для gRPC
EXPOSE 50051

# Запускаем сервис
CMD ["python", "-m", "recommendations.main"] 