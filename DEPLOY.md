# Руководство по деплою

Данное руководство описывает процесс деплоя рекомендательной системы на production сервер.

## Предварительные требования

- Сервер с установленным Docker и Docker Compose
- Доступ к MySQL базе данных с данными о пользователях и контенте

## Подготовка сервера

1. Обновите пакеты и установите необходимые зависимости:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose git
```

2. Настройте Docker для запуска без sudo (опционально):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Развертывание приложения

### Шаг 1: Загрузка кода

```bash
git clone https://github.com/your-username/recomender.git
cd recomender
```

### Шаг 2: Настройка переменных окружения

Создайте файл `.env` и настройте переменные окружения:

```bash
cp .env.example .env
nano .env
```

Отредактируйте файл, указав параметры подключения к вашей БД:

```
# Параметры подключения к базе данных
RE_DB_HOST=your-db-host
RE_DB=your-db-name
RE_DB_USER=your-db-username
RE_DB_PASSWORD=your-db-password
RE_DB_PORT=3306

# Параметры API сервера
API_PORT=8000
DEBUG=false

# Параметры gRPC сервера
GRPC_PORT=50051
GRPC_MAX_WORKERS=10
```

### Шаг 3: Сборка и запуск Docker контейнеров

```bash
docker-compose build
docker-compose up -d
```

### Шаг 4: Проверка работоспособности

Проверьте, что сервисы успешно запущены:

```bash
docker-compose ps
```

Проверьте логи:

```bash
docker-compose logs -f
```

Проверьте доступность API:

```bash
curl http://localhost:8000/health
```

## Настройка производительности

### Шаг 1: Тюнинг параметров модели

Параметры модели можно настроить, отредактировав файл `recommendations/config.py`:

```python
MODEL_CONFIG: Dict[str, Any] = {
    "no_components": 100,  # Размерность эмбеддингов (увеличьте для большей точности)
    "loss": "bpr",         # bpr, warp, logistic
    "num_threads": 3,      # Количество потоков для обучения
    "epochs": 30,          # Количество эпох обучения
}
```

### Шаг 2: Оптимизация Docker контейнеров

Если требуется ограничить ресурсы, выделяемые контейнерам, добавьте в `docker-compose.yaml`:

```yaml
services:
  recommendation_api:
    # ... существующие настройки ...
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Настройка NGINX (опционально)

Для использования сервиса за NGINX, создайте конфигурационный файл:

```bash
sudo nano /etc/nginx/sites-available/recommendation
```

Добавьте следующую конфигурацию:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Активируйте конфигурацию и перезапустите NGINX:

```bash
sudo ln -s /etc/nginx/sites-available/recommendation /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Автоматическое обновление данных

Модель по умолчанию обновляется каждый день в 4:00 UTC. Вы можете изменить расписание, отредактировав файл `recommendations/rec_service.py`.

Для ручного запуска обновления:

```bash
curl -X GET http://localhost:8000/train
```

## Резервное копирование и восстановление

### Резервное копирование данных моделей

```bash
tar -czf backup_models_$(date +%Y%m%d).tar.gz data/
```

### Восстановление из резервной копии

```bash
tar -xzf backup_models_YYYYMMDD.tar.gz -C /path/to/application/
```

## Устранение неполадок

### Проблема: Сервис не запускается

1. Проверьте логи:
```bash
docker-compose logs
```

2. Убедитесь, что БД доступна:
```bash
docker exec -it recommendation_api python -c "from core.database import engine; print(engine.connect())"
```

### Проблема: Модель не обучается

1. Проверьте доступ к базе данных
2. Проверьте логи обучения:
```bash
docker-compose logs recommendation_api | grep "train"
```

### Проблема: gRPC сервер не работает

1. Проверьте, запущен ли сервис:
```bash
docker-compose ps recommendation_grpc
```

2. Проверьте, открыт ли порт:
```bash
nc -zv localhost 50051
```

## Мониторинг

Для базового мониторинга используйте:

```bash
docker stats $(docker-compose ps -q)
```

## Обновление системы

Для обновления системы до новой версии:

```bash
git pull
docker-compose down
docker-compose build
docker-compose up -d
``` 