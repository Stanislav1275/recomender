import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FastRanker:
    """
    Сервис быстрого реранжирования рекомендаций на основе свежих взаимодействий
    """
    
    def __init__(self, model_manager, data_service=None, cache_service=None, config=None):
        self.model_manager = model_manager
        self.data_service = data_service
        self.cache_service = cache_service
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Хранилище свежих взаимодействий
        self.recent_interactions = {}
        # Хранилище векторных представлений элементов
        self.item_vectors = {}
        # Хранилище векторных представлений пользователей
        self.user_vectors = {}
        # Множество новых пользователей (не было на этапе обучения)
        self.new_users = set()
        
        # Максимальный возраст взаимодействий для хранения
        self.max_interaction_age = timedelta(minutes=self.config.get("interaction_ttl_minutes", 30))
        
    async def initialize(self):
        """Инициализирует быстрый рекомендатель"""
        try:
            # Получаем модель и датасет
            model = self.model_manager._model
            
            if model and hasattr(model, 'model') and hasattr(model.model, 'item_embeddings'):
                # Извлекаем векторные представления элементов из LightFM модели в rectools
                self.item_vectors = {
                    item_id: model.model.item_embeddings[int_id] 
                    for item_id, int_id in model._item_id_map.external_to_internal.items()
                }
                self.logger.info(f"Loaded {len(self.item_vectors)} item vectors")
                
                # Извлекаем векторные представления пользователей
                if hasattr(model.model, 'user_embeddings'):
                    self.user_vectors = {
                        user_id: model.model.user_embeddings[int_id]
                        for user_id, int_id in model._user_id_map.external_to_internal.items()
                    }
                    self.logger.info(f"Loaded {len(self.user_vectors)} user vectors")
            else:
                self.logger.warning("Model does not support embeddings extraction")
                
            # Запускаем фоновую задачу для очистки устаревших взаимодействий
            asyncio.create_task(self._cleanup_old_interactions())
            
        except Exception as e:
            self.logger.error(f"Error initializing FastRanker: {e}")
    
    async def hot_interact(self, user_id: int, item_id: int, interaction_type: str = "watch", weight: float = 1.0):
        """
        Обрабатывает новое взаимодействие пользователя
        
        Args:
            user_id: ID пользователя
            item_id: ID элемента
            interaction_type: Тип взаимодействия
            weight: Вес взаимодействия
            
        Returns:
            True если взаимодействие успешно обработано
        """
        try:
            # Проверяем, является ли пользователь новым
            if user_id not in self.user_vectors:
                self.new_users.add(user_id)
                # Создаем случайный вектор для нового пользователя
                user_vector_size = next(iter(self.user_vectors.values())).shape[0] if self.user_vectors else 64
                self.user_vectors[user_id] = np.random.normal(0, 0.1, user_vector_size)
                
            # Добавляем взаимодействие в хранилище
            if user_id not in self.recent_interactions:
                self.recent_interactions[user_id] = []
                
            # Добавляем взаимодействие с текущим временем
            self.recent_interactions[user_id].append({
                'item_id': item_id,
                'interaction_type': interaction_type,
                'weight': weight,
                'timestamp': datetime.now()
            })
            
            # Инвалидируем кэш рекомендаций для этого пользователя
            if self.cache_service:
                await self.cache_service.invalidate_user_recommendations(user_id)
            
            self.logger.info(f"Hot interaction processed: user={user_id}, item={item_id}, type={interaction_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing hot interaction: {e}")
            return False
    
    async def recommend(self, user_id: int, k: int = 10, filter_viewed: bool = True):
        """
        Быстрое получение рекомендаций с учетом свежих взаимодействий
        
        Args:
            user_id: ID пользователя
            k: Количество рекомендаций
            filter_viewed: Фильтровать ли просмотренные элементы
            
        Returns:
            Список ID рекомендованных элементов
        """
        # Проверяем кэш
        cached_recos = None
        if self.cache_service:
            cached_recos = await self.cache_service.get_user_recommendations(user_id)
        if cached_recos:
            self.logger.debug(f"Returning cached recommendations for user {user_id}")
            return cached_recos[:k]
            
        try:
            # Если пользователь не известен системе и у него нет взаимодействий,
            # возвращаем популярные элементы
            if user_id not in self.user_vectors and user_id not in self.recent_interactions:
                self.logger.info(f"User {user_id} is unknown, returning popular items")
                popular_items = await self._get_popular_items(k * 2)
                return popular_items[:k]
                
            # Собираем просмотренные элементы, если нужно фильтровать
            viewed_items = set()
            if filter_viewed and user_id in self.recent_interactions:
                for interaction in self.recent_interactions[user_id]:
                    viewed_items.add(interaction['item_id'])
                    
            # Получаем рекомендации от основной модели
            base_recos = []
            if user_id in self.user_vectors:
                base_recos = await self._get_base_recommendations(user_id, k * 2)
                
            # Если пользователь новый, но у него есть взаимодействия,
            # получаем рекомендации на основе похожих элементов
            similar_recos = []
            if user_id in self.recent_interactions:
                similar_recos = await self._get_similar_item_recommendations(user_id, k * 2)
                
            # Объединяем и пересортируем рекомендации
            combined_recos = await self._combine_and_rerank(
                user_id,
                base_recos,
                similar_recos,
                viewed_items,
                k
            )
            
            # Кэшируем результат
            if self.cache_service:
                await self.cache_service.cache_user_recommendations(user_id, combined_recos)
            
            return combined_recos[:k]
            
        except Exception as e:
            self.logger.error(f"Error getting fast recommendations: {e}")
            # В случае ошибки возвращаем базовые рекомендации
            return await self._fallback_recommendations(user_id, k)
    
    async def _get_base_recommendations(self, user_id: int, k: int):
        """Получает базовые рекомендации от основной модели"""
        try:
            model = self.model_manager._model
            dataset = self.model_manager._dataset
            
            if model is None or dataset is None:
                return []
                
            recos = model.recommend(
                users=[user_id], 
                dataset=dataset,
                k=k,
                filter_viewed=True
            )
            
            if recos.empty:
                return []
                
            return recos[recos['user_id'] == user_id]['item_id'].tolist()
            
        except Exception as e:
            self.logger.error(f"Error getting base recommendations: {e}")
            return []
    
    async def _get_similar_item_recommendations(self, user_id: int, k: int):
        """Получает рекомендации на основе похожих элементов для новых пользователей"""
        if user_id not in self.recent_interactions:
            return []
            
        try:
            # Получаем элементы, с которыми взаимодействовал пользователь
            interacted_items = [
                interaction['item_id'] 
                for interaction in self.recent_interactions[user_id]
            ]
            
            if not interacted_items:
                return []
                
            # Получаем похожие элементы для каждого
            similar_items = []
            for item_id in interacted_items:
                items = await self._get_similar_items(item_id, k // len(interacted_items) + 1)
                similar_items.extend(items)
                
            # Удаляем дубликаты
            unique_items = list(dict.fromkeys(similar_items))
            
            return unique_items[:k]
            
        except Exception as e:
            self.logger.error(f"Error getting similar item recommendations: {e}")
            return []
    
    async def _get_similar_items(self, item_id: int, k: int):
        """Находит похожие элементы на основе векторных представлений"""
        if not self.item_vectors or item_id not in self.item_vectors:
            return []
            
        try:
            # Вычисляем косинусное сходство с другими элементами
            item_vector = self.item_vectors[item_id]
            
            # Нормализуем вектор
            item_norm = np.linalg.norm(item_vector)
            if item_norm == 0:
                return []
                
            normalized_vector = item_vector / item_norm
            
            # Вычисляем скоры для всех элементов
            scores = {}
            for other_id, other_vector in self.item_vectors.items():
                if other_id == item_id:
                    continue
                    
                other_norm = np.linalg.norm(other_vector)
                if other_norm == 0:
                    continue
                    
                normalized_other = other_vector / other_norm
                
                # Косинусное сходство
                score = np.dot(normalized_vector, normalized_other)
                scores[other_id] = score
                
            # Сортируем элементы по убыванию скора
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return [item_id for item_id, _ in sorted_items[:k]]
            
        except Exception as e:
            self.logger.error(f"Error getting similar items: {e}")
            return []
    
    async def _combine_and_rerank(self, user_id: int, base_recos: List[int], 
                                 similar_recos: List[int], viewed_items: Set[int], k: int):
        """Объединяет и пересортирует рекомендации"""
        # Фильтруем просмотренные элементы
        filtered_base = [item for item in base_recos if item not in viewed_items]
        filtered_similar = [item for item in similar_recos if item not in viewed_items]
        
        # Объединяем списки рекомендаций
        combined = []
        
        # Чередуем элементы из обоих списков
        base_idx = 0
        similar_idx = 0
        
        while len(combined) < k * 2 and (base_idx < len(filtered_base) or similar_idx < len(filtered_similar)):
            # Добавляем элемент из базовых рекомендаций
            if base_idx < len(filtered_base):
                item = filtered_base[base_idx]
                if item not in combined:
                    combined.append(item)
                base_idx += 1
                
            # Добавляем элемент из похожих рекомендаций
            if similar_idx < len(filtered_similar):
                item = filtered_similar[similar_idx]
                if item not in combined:
                    combined.append(item)
                similar_idx += 1
        
        # Если недостаточно рекомендаций, добавляем популярные элементы
        if len(combined) < k:
            popular = await self._get_popular_items(k)
            for item in popular:
                if item not in combined and item not in viewed_items:
                    combined.append(item)
                    if len(combined) >= k:
                        break
        
        return combined[:k]
    
    async def _get_popular_items(self, k: int):
        """Получает список популярных элементов"""
        try:
            # В реальном приложении здесь должен быть запрос к базе данных
            # или кэшу с популярными элементами
            # Для упрощения возвращаем список первых k элементов
            if self.item_vectors:
                return list(self.item_vectors.keys())[:k]
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting popular items: {e}")
            return []
    
    async def _fallback_recommendations(self, user_id: int, k: int):
        """Возвращает запасные рекомендации в случае ошибки"""
        try:
            # Сначала пытаемся получить рекомендации из кэша
            cached = None
            if self.cache_service:
                cached = await self.cache_service.get_user_recommendations(user_id)
            if cached:
                return cached[:k]
                
            # Иначе возвращаем популярные элементы
            return await self._get_popular_items(k)
            
        except Exception as e:
            self.logger.error(f"Error getting fallback recommendations: {e}")
            return []
    
    async def _cleanup_old_interactions(self):
        """Фоновая задача для очистки устаревших взаимодействий"""
        while True:
            try:
                now = datetime.now()
                to_remove = []
                
                for user_id, interactions in self.recent_interactions.items():
                    # Фильтруем актуальные взаимодействия
                    filtered = [
                        interaction for interaction in interactions
                        if now - interaction['timestamp'] <= self.max_interaction_age
                    ]
                    
                    if filtered:
                        self.recent_interactions[user_id] = filtered
                    else:
                        # Если все взаимодействия устарели, удаляем пользователя из хранилища
                        to_remove.append(user_id)
                        
                # Удаляем пользователей без актуальных взаимодействий
                for user_id in to_remove:
                    del self.recent_interactions[user_id]
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up old interactions: {e}")
                
            # Ждем 5 минут перед следующей очисткой
            await asyncio.sleep(300) 