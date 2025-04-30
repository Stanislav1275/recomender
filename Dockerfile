FROM python:3.9-slim

# Установка системных зависимостей для компиляции LightFM и других пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libomp-dev \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка переменной окружения для OpenMP
ENV OMP_NUM_THREADS=1

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Компиляция proto-файлов
RUN python -m grpc_tools.protoc -I./recommendations/protos --python_out=./recommendations/protos --grpc_python_out=./recommendations/protos ./recommendations/protos/recommendations.proto
RUN touch ./recommendations/protos/__init__.py
RUN touch ./recommendations/protos/recommendations_pb2_grpc.py

# Переменные окружения по умолчанию
ENV REC_DB_URL="postgresql://user:password@db:5432/rec_db"
ENV REC_GRPC_HOST="0.0.0.0"
ENV REC_GRPC_PORT="50051"
ENV REC_CACHE_URL="redis://redis:6379/0"
ENV REC_KAFKA_SERVERS="kafka:9092"
ENV REC_KAFKA_TOPIC="user-interactions"

# Порт для gRPC
EXPOSE 50051

# Запуск сервиса
CMD ["python", "-m", "recommendations.main"]