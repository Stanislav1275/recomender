import json
import logging
from typing import Dict, Any
import asyncio
from recommendations.core.messaging.message import InteractionMessage

class MessageProducer:
    def __init__(self, bootstrap_servers, topic="user-interactions"):
        self.topic = topic
        self.logger = logging.getLogger(__name__)
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        
        # Проверяем наличие Kafka
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.logger.info(f"Kafka producer initialized: {bootstrap_servers}")
        except ImportError:
            self.logger.warning("Kafka library not installed, messages will be logged only")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
    
    async def send_interaction(self, message: InteractionMessage) -> bool:
        """
        Отправляет сообщение о взаимодействии в очередь
        
        Args:
            message: Сообщение о взаимодействии
            
        Returns:
            True если сообщение успешно отправлено
        """
        try:
            # Сериализуем модель в словарь
            message_dict = message.dict()
            # Преобразуем datetime в строку
            message_dict["timestamp"] = message_dict["timestamp"].isoformat()
            
            # Если Kafka не доступна, просто логируем
            if not self.producer:
                self.logger.info(f"[MOCK KAFKA] Message for {self.topic}: {message_dict}")
                return True
                
            # Отправляем сообщение в отдельном потоке
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_message,
                message_dict,
                str(message.user_id).encode('utf-8')
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
            
    def _send_message(self, message_dict, key):
        """Синхронная отправка сообщения в Kafka"""
        try:
            future = self.producer.send(
                self.topic, 
                value=message_dict,
                key=key
            )
            # Ждем подтверждения отправки
            future.get(timeout=10)
            self.logger.debug(f"Message sent to topic {self.topic}: {message_dict}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending message to Kafka: {e}")
            return False 