#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска отдельного gRPC сервера рекомендаций.
"""

import asyncio
import logging
from recommendations.rec_server import serve
from recommendations.config import GRPC_CONFIG

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Основная функция для запуска gRPC сервера.
    """
    try:
        logger.info("Запуск отдельного gRPC сервера...")
        await serve(
            host=f"{GRPC_CONFIG['host']}:{GRPC_CONFIG['port']}",
            max_workers=GRPC_CONFIG['max_workers']
        )
    except KeyboardInterrupt:
        logger.info("Сервер остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")


if __name__ == '__main__':
    logger.info("Инициализация gRPC сервера рекомендаций")
    asyncio.run(main())