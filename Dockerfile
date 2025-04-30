FROM python:3.9-slim AS builder

# Установка необходимых зависимостей для компиляции
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка базовых инструментов перед другими зависимостями
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

# Копирование виртуального окружения из этапа сборки
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Настройка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1

# Создание рабочего каталога
WORKDIR /app

# Копирование кода приложения
COPY . /app/

# Создание директорий для моделей
RUN mkdir -p /app/data/cur /app/data/prev && \
    chmod -R 777 /app/data

# Создание непривилегированного пользователя
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

# Переключение на непривилегированного пользователя
USER appuser

# Запуск приложения
CMD ["python", "main.py"]