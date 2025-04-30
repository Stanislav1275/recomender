#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для настройки и инициализации рекомендательной системы
"""

import os
import sys
import argparse
import logging
import shutil
import asyncio
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup")

# Проверяем, запущен ли скрипт в корне проекта
if not os.path.exists('main.py'):
    logger.error("Скрипт должен запускаться из корня проекта!")
    sys.exit(1)

async def prepare_directories():
    """Подготовка необходимых директорий"""
    logger.info("Подготовка директорий для данных и моделей...")
    
    # Создаем структуру директорий
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/cur",
        "data/prev",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Создана директория: {directory}")
    
    return True

def check_env_file():
    """Проверка наличия .env файла и создание из примера при необходимости"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            logger.warning("Файл .env не найден. Создаю из .env.example")
            shutil.copy2('.env.example', '.env')
            logger.info("Создан файл .env из примера. Пожалуйста, отредактируйте его!")
        else:
            logger.error("Не найдены файлы .env и .env.example!")
            return False
    return True

async def init_train_model():
    """Инициализация и первичное обучение модели"""
    try:
        # Импортируем необходимые модули
        logger.info("Импортирую модули для обучения модели...")
        from recommendations.rec_service import ModelManager
        
        # Подключаемся к базе данных и обучаем модель
        logger.info("Запускаю инициализацию и обучение модели...")
        manager = ModelManager()
        await manager.initialize()
        
        logger.info("Обучение модели завершено успешно!")
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {e}")
        return False

def parse_arguments():
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Настройка рекомендательной системы")
    
    parser.add_argument(
        "--prepare", 
        action="store_true", 
        help="Подготовить директории для данных и логов"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Запустить обучение модели"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Выполнить полную настройку (директории + обучение)"
    )
    
    return parser.parse_args()

async def main():
    """Основная функция настройки"""
    args = parse_arguments()
    
    if not args.prepare and not args.train and not args.full:
        logger.error("Не указаны действия. Используйте --help для просмотра опций.")
        return False
    
    # Проверяем и создаем .env файл
    if not check_env_file():
        return False
    
    success = True
    
    # Подготовка директорий
    if args.prepare or args.full:
        if not await prepare_directories():
            success = False
    
    # Обучение модели
    if args.train or args.full:
        if not await init_train_model():
            success = False
    
    if success:
        logger.info("Настройка системы завершена успешно!")
        return True
    else:
        logger.error("Настройка системы завершена с ошибками.")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1) 