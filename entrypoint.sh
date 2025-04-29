#!/bin/bash
set -e

# Ждем доступности Redis
echo "Waiting for Redis..."
until nc -z redis 6379; do
  echo "Redis is unavailable - sleeping"
  sleep 1
done
echo "Redis is up - starting recommender service"

# Запускаем FastAPI сервис
python main.py 