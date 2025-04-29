import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional

class BatchUpdater:
    """
    Периодический обработчик накопленных взаимодействий
    """
    
    def __init__(self, model_manager, config=None):
        self.model_manager = model_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Накопленные взаимодействия
        self.interactions_buffer = []
        
        # Интервал обновления в секундах
        self.update_interval = self.config.get("batch_update_interval", 300)  # 5 минут по умолчанию
        
    async def start(self):
        """Запускает планировщик обновлений"""
        # Запускаем фоновую задачу для периодического обновления модели
        asyncio.create_task(self._periodic_update())
        
        self.logger.info(f"Batch updater started with interval {self.update_interval} seconds")
        
    async def add_interaction(self, user_id: int, item_id: int, 
                             interaction_type: str = "watch", weight: float = 1.0):
        """
        Добавляет взаимодействие в буфер
        
        Args:
            user_id: ID пользователя
            item_id: ID элемента
            interaction_type: Тип взаимодействия
            weight: Вес взаимодействия
        """
        try:
            # Добавляем взаимодействие в буфер
            self.interactions_buffer.append({
                'user_id': user_id,
                'item_id': item_id,
                'weight': weight,
                'interaction_type': interaction_type,
                'timestamp': datetime.now()
            })
            
            self.logger.debug(f"Added interaction to buffer: user={user_id}, item={item_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding interaction to buffer: {e}")
    
    async def _periodic_update(self):
        """Фоновая задача для периодического обновления модели"""
        while True:
            await asyncio.sleep(self.update_interval)
            
            # Если буфер пуст, пропускаем обновление
            if not self.interactions_buffer:
                self.logger.debug("No interactions to process, skipping update")
                continue
                
            try:
                # Копируем текущий буфер и очищаем его
                interactions = self.interactions_buffer.copy()
                self.interactions_buffer = []
                
                # Преобразуем в DataFrame
                interactions_df = pd.DataFrame(interactions)
                
                # Логируем статистику
                self.logger.info(f"Processing {len(interactions_df)} interactions from {interactions_df['user_id'].nunique()} users")
                
                # Обновляем модель
                await self._update_model(interactions_df)
                
            except Exception as e:
                self.logger.error(f"Error in periodic update: {e}")
    
    async def _update_model(self, interactions_df: pd.DataFrame):
        """Обновляет модель с новыми взаимодействиями"""
        try:
            # Получаем текущий датасет
            dataset = self.model_manager._dataset
            if dataset is None:
                self.logger.warning("No dataset available for model update")
                return False
            
            # Создаем новый датасет с добавленными взаимодействиями
            from rectools import Columns
            
            # Преобразуем колонки для соответствия RecTools
            interactions_df = interactions_df.rename(columns={
                'user_id': Columns.User,
                'item_id': Columns.Item,
                'weight': Columns.Weight,
                'timestamp': Columns.Datetime
            })
            
            # Обновляем датасет
            new_dataset = dataset.update_interactions(interactions_df)
            
            # Обучаем модель на обновленном датасете
            if hasattr(self.model_manager, 'fit') and callable(self.model_manager.fit):
                success = await self.model_manager.fit(new_dataset)
                if success:
                    self.logger.info("Model successfully updated with new interactions")
                else:
                    self.logger.warning("Failed to update model with new interactions")
                return success
            else:
                self.logger.warning("Model manager does not support fit method")
                return False
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False 