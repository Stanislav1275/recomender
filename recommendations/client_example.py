"""
Пример клиентского кода для работы с сервисом рекомендаций.
"""
import grpc
from recommendations.protos import recommendations_pb2
from recommendations.protos import recommendations_pb2_grpc
import time
from typing import List, Optional


class RecommendationClient:
    """
    Клиент для работы с gRPC сервисом рекомендаций
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051, token: Optional[str] = None):
        """
        Инициализирует клиент для работы с сервисом рекомендаций
        
        Args:
            host: Хост сервиса рекомендаций
            port: Порт сервиса рекомендаций
            token: Токен авторизации (если требуется)
        """
        # Создаем gRPC канал
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        
        # Создаем клиент для сервиса рекомендаций
        self.stub = recommendations_pb2_grpc.RecommenderStub(self.channel)
        
        # Сохраняем токен авторизации
        self.token = token
        
        # Метаданные для авторизации
        self.metadata = []
        if token:
            self.metadata.append(("authorization", f"Bearer {token}"))
    
    def get_recommendations(self, user_id: int, limit: int = 10, filter_viewed: bool = True) -> List[int]:
        """
        Получает рекомендации для пользователя
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество рекомендаций
            filter_viewed: Фильтровать ли просмотренные элементы
            
        Returns:
            Список ID рекомендованных элементов
        """
        # Создаем запрос
        request = recommendations_pb2.UserRequest(
            user_id=user_id,
            limit=limit,
            filter_viewed=filter_viewed
        )
        
        # Выполняем запрос
        try:
            response = self.stub.GetUserRecommendations(request, metadata=self.metadata)
            return list(response.item_ids)
        except grpc.RpcError as e:
            print(f"Ошибка при получении рекомендаций: {e.details()}")
            return []
    
    def get_similar_items(self, item_id: int, limit: int = 10) -> List[int]:
        """
        Получает похожие элементы для указанного элемента
        
        Args:
            item_id: ID элемента
            limit: Максимальное количество похожих элементов
            
        Returns:
            Список ID похожих элементов
        """
        # Создаем запрос
        request = recommendations_pb2.TitleRequest(
            title_id=item_id
        )
        
        # Выполняем запрос
        try:
            response = self.stub.GetTitleRelevant(request, metadata=self.metadata)
            return list(response.item_ids)
        except grpc.RpcError as e:
            print(f"Ошибка при получении похожих элементов: {e.details()}")
            return []
    
    def register_interaction(self, user_id: int, item_id: int, 
                          interaction_type: str = "watch", weight: float = 1.0) -> bool:
        """
        Регистрирует новое взаимодействие пользователя с элементом
        
        Args:
            user_id: ID пользователя
            item_id: ID элемента
            interaction_type: Тип взаимодействия (watch, like и т.д.)
            weight: Вес взаимодействия
            
        Returns:
            True если взаимодействие успешно зарегистрировано
        """
        # Создаем запрос
        request = recommendations_pb2.InteractionRequest(
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type,
            weight=weight
        )
        
        # Выполняем запрос
        try:
            response = self.stub.RegisterInteraction(request, metadata=self.metadata)
            return response.success
        except grpc.RpcError as e:
            print(f"Ошибка при регистрации взаимодействия: {e.details()}")
            return False
    
    def train_model(self, force_retrain: bool = False) -> bool:
        """
        Запускает обучение модели
        
        Args:
            force_retrain: Принудительное переобучение модели
            
        Returns:
            True если обучение запущено успешно
        """
        # Создаем запрос
        request = recommendations_pb2.TrainRequest(
            force_retrain=force_retrain
        )
        
        # Выполняем запрос
        try:
            response = self.stub.TrainModel(request, metadata=self.metadata)
            return response.success
        except grpc.RpcError as e:
            print(f"Ошибка при запуске обучения модели: {e.details()}")
            return False
    
    def close(self):
        """Закрывает соединение с сервером"""
        self.channel.close()


def demo():
    """Демонстрирует использование клиента для работы с сервисом рекомендаций"""
    # Создаем клиент
    client = RecommendationClient(host="localhost", port=50051)
    
    try:
        # Получаем рекомендации для пользователя с ID = 1
        print("Получение рекомендаций для пользователя с ID = 1:")
        recommendations = client.get_recommendations(user_id=1, limit=5)
        print(f"Рекомендации: {recommendations}")
        
        # Имитируем взаимодействие пользователя с элементом
        item_id = 100 if not recommendations else recommendations[0]
        print(f"\nРегистрация взаимодействия: user_id=1, item_id={item_id}")
        success = client.register_interaction(user_id=1, item_id=item_id, interaction_type="watch")
        print(f"Успешно: {success}")
        
        # Получаем обновленные рекомендации
        print("\nПолучение обновленных рекомендаций:")
        updated_recommendations = client.get_recommendations(user_id=1, limit=5)
        print(f"Обновленные рекомендации: {updated_recommendations}")
        
        # Получаем похожие элементы
        print(f"\nПолучение похожих элементов для item_id={item_id}:")
        similar_items = client.get_similar_items(item_id=item_id, limit=5)
        print(f"Похожие элементы: {similar_items}")
        
    finally:
        # Закрываем соединение
        client.close()


if __name__ == "__main__":
    demo() 