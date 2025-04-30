.PHONY: setup venv install run_api run_grpc run clean env init_model test docker docker_build docker_up docker_down help

VENV = ./.venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help: ## Показать эту справку
	@echo "Использование: make [цель]"
	@echo ""
	@echo "Цели:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: venv install env ## Полная настройка проекта (venv + install + env)

venv: ## Создать виртуальное окружение
	python3 -m venv $(VENV)

install: ## Установить зависимости
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

env: ## Создать файл .env из примера
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo ".env создан из примера. Отредактируйте его!"; \
		else \
			echo "Файл .env.example не найден!"; \
			exit 1; \
		fi; \
	else \
		echo "Файл .env уже существует."; \
	fi

init_model: ## Инициализировать структуру директорий и обучить модель
	$(PYTHON) setup.py --full

run_api: ## Запустить API сервер
	$(PYTHON) main.py

run_grpc: ## Запустить gRPC сервер
	$(PYTHON) grpc_server.py

run: ## Запустить API и gRPC серверы (через Docker)
	docker-compose up -d

clean: ## Очистить временные файлы
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

test: ## Запустить тесты
	pytest -xvs

docker_build: ## Собрать Docker образы
	docker-compose build

docker_up: ## Запустить контейнеры
	docker-compose up -d

docker_down: ## Остановить контейнеры
	docker-compose down

logs: ## Показать логи контейнеров
	docker-compose logs -f

train: ## Запустить переобучение модели
	curl -X GET http://localhost:8000/train 