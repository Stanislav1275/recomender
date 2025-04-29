import json
import logging
import asyncio
from typing import List, Dict, Any, Callable, Awaitable
from datetime import datetime
from recommendations.core.messaging.message import InteractionMessage
from recommendations.workers.batch_updater import BatchUpdater

class MessageConsumer:
    def __init__(self, bootstrap_servers, batch_updater: BatchUpdater, 
                topic="user-interactions", group_id="rec-updater"):
        self.topic = topic
        self.group_id = group_id
        self.logger = logging.getLogger(__name__)
        self.bootstrap_servers = bootstrap_servers
        self.batch_updater = batch_updater
        self.consumer = None
        self.running = False
        
    async def start(self):
        """Запускает потребителя сообщений"""
        self.running = True
        
        # Проверяем наличие Kafka
        try:
            from kafka import KafkaConsumer
            # Запускаем потребителя в отдельном потоке
            loop = asyncio.get_event_loop()
            asyncio.create_task(loop.run_in_executor(None, self._consume_messages))
            self.logger.info(f"Started Kafka consumer for topic {self.topic}")
        except ImportError:
            self.logger.warning("Kafka library not installed, running in mock mode")
            # Запускаем эмуляцию потребителя
            asyncio.create_task(self._mock_consumer())
        
    async def _mock_consumer(self):
        """Эмуляция работы потребителя сообщений (для отладки)"""
        self.logger.info("Started mock message consumer")
        while self.running:
            await asyncio.sleep(10)
        
    def _consume_messages(self):
        """Основной цикл потребления сообщений"""
        try:
            from kafka import KafkaConsumer
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='latest',
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            self.logger.info(f"Started consuming from topic {self.topic}")
            
            for message in self.consumer:
                if not self.running:
                    break
                    
                try:
                    # Преобразуем сообщение в модель
                    value = message.value
                    # Преобразуем строку в datetime
                    if "timestamp" in value and isinstance(value["timestamp"], str):
                        value["timestamp"] = datetime.fromisoformat(value["timestamp"].replace('Z', '+00:00'))
                        
                    interaction = InteractionMessage(**value)
                    
                    # Добавляем взаимодействие в обработчик пакетов
                    asyncio.run(self.batch_updater.add_interaction(
                        user_id=interaction.user_id,
                        item_id=interaction.item_id,
                        interaction_type=interaction.interaction_type,
                        weight=interaction.weight
                    ))
                        
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in Kafka consumer: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
                
    async def stop(self):
        """Останавливает потребителя сообщений"""
        self.running = False
        if self.consumer:
            self.consumer.close() 