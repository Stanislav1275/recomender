FROM python:3.12

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install rectools[lightfm]
RUN pip install pymysql

# Копирование только необходимых файлов
COPY core/ ./core/
COPY models.py models.py
COPY recommendations/ ./recommendations/
COPY grpc_server.py .

# Запуск сервера
CMD ["python", "grpc_server.py"]