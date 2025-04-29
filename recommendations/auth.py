import os
import logging
import ipaddress
from typing import Optional, List, Dict, Set, Tuple
from functools import wraps
import grpc
from grpc.aio import ServicerContext

logger = logging.getLogger(__name__)

# Константы для типов токенов
TOKEN_TYPE_ADMIN = "admin"
TOKEN_TYPE_USER = "user"

class AuthService:
    """
    Сервис аутентификации и авторизации для API рекомендаций
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Инициализирует сервис аутентификации, загружая токены и разрешенные IP-адреса 
        из переменных окружения
        """
        # Загружаем API токены из переменных окружения
        self.admin_token = os.environ.get("REC_ADMIN_TOKEN", "")
        self.user_token = os.environ.get("REC_USER_TOKEN", "")
        
        # Если токены не заданы, генерируем предупреждение
        if not self.admin_token:
            logger.warning("Административный токен API не задан! Доступ к административным методам отключен.")
        
        if not self.user_token:
            logger.warning("Пользовательский токен API не задан! Доступ к API рекомендаций будет открыт для всех.")
        
        # Загружаем разрешенные IP-адреса из переменных окружения
        allowed_ips_str = os.environ.get("REC_ALLOWED_IPS", "")
        self.allowed_ips: Set[ipaddress.IPv4Network] = set()
        
        if allowed_ips_str:
            for ip_str in allowed_ips_str.split(","):
                try:
                    # Если адрес содержит маску подсети
                    if "/" in ip_str:
                        self.allowed_ips.add(ipaddress.IPv4Network(ip_str.strip(), strict=False))
                    else:
                        # Если это одиночный IP, добавляем его как сеть с маской /32
                        self.allowed_ips.add(ipaddress.IPv4Network(f"{ip_str.strip()}/32", strict=False))
                except ValueError as e:
                    logger.error(f"Ошибка при разборе IP-адреса {ip_str}: {e}")
        
        logger.info(f"Настроены разрешенные IP-адреса: {self.allowed_ips}")
        
        # Словарь с разрешенными методами для каждого типа токена
        self.allowed_methods: Dict[str, Set[str]] = {
            TOKEN_TYPE_ADMIN: {
                "train", "list_models", "get_model_info", "set_active_model", 
                "schedule_training", "rec", "relevant", "fit_partial"
            },
            TOKEN_TYPE_USER: {
                "rec", "relevant", "get_user_recent_interactions"
            }
        }
    
    def verify_token(self, token: str) -> Tuple[bool, str]:
        """
        Проверяет валидность токена и возвращает его тип
        
        Args:
            token: Токен для проверки
            
        Returns:
            Кортеж (is_valid, token_type)
        """
        if token == self.admin_token and self.admin_token:
            return True, TOKEN_TYPE_ADMIN
        elif token == self.user_token and self.user_token:
            return True, TOKEN_TYPE_USER
        return False, ""
    
    def check_ip_allowed(self, ip_address: str) -> bool:
        """
        Проверяет, разрешен ли доступ с данного IP-адреса
        
        Args:
            ip_address: IP-адрес для проверки
            
        Returns:
            True, если IP-адрес разрешен, иначе False
        """
        # Если список разрешенных IP пуст, разрешаем доступ всем
        if not self.allowed_ips:
            return True
            
        try:
            client_ip = ipaddress.IPv4Address(ip_address)
            
            # Проверяем вхождение IP в разрешенные сети
            for network in self.allowed_ips:
                if client_ip in network:
                    return True
                    
            return False
            
        except ValueError:
            logger.error(f"Некорректный формат IP-адреса: {ip_address}")
            return False
    
    def is_method_allowed(self, token_type: str, method_name: str) -> bool:
        """
        Проверяет, разрешен ли доступ к методу для данного типа токена
        
        Args:
            token_type: Тип токена (admin или user)
            method_name: Имя метода API
            
        Returns:
            True, если метод разрешен для данного типа токена, иначе False
        """
        allowed_methods = self.allowed_methods.get(token_type, set())
        return method_name in allowed_methods


def auth_required(method_name: str):
    """
    Декоратор для проверки аутентификации и авторизации для методов API
    
    Args:
        method_name: Имя метода API
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Находим контекст gRPC в аргументах
            context = None
            for arg in args:
                if isinstance(arg, ServicerContext):
                    context = arg
                    break
                    
            if not context and "context" in kwargs:
                context = kwargs["context"]
                
            if not context:
                # Если контекст не найден, предполагаем, что это внутренний вызов
                return await func(self, *args, **kwargs)
                
            # Получаем метаданные запроса
            metadata = dict(context.invocation_metadata())
            token = metadata.get("authorization", "")
            
            # Удаляем префикс "Bearer " из токена, если он есть
            if token.startswith("Bearer "):
                token = token[7:]
                
            # Получаем IP клиента
            peer_info = context.peer()
            client_ip = peer_info.split(":")[-2].replace("[", "").replace("]", "")
            
            auth_service = AuthService()
            
            # Проверяем IP-адрес
            if not auth_service.check_ip_allowed(client_ip):
                logger.warning(f"Доступ запрещен для IP-адреса {client_ip}")
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "Доступ запрещен для вашего IP-адреса")
                return
                
            # Проверяем токен
            is_valid, token_type = auth_service.verify_token(token)
            
            if not is_valid:
                logger.warning(f"Недействительный токен: {token}")
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Недействительный токен авторизации")
                return
                
            # Проверяем разрешения
            if not auth_service.is_method_allowed(token_type, method_name):
                logger.warning(f"Метод {method_name} не разрешен для токена типа {token_type}")
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "Недостаточно прав для выполнения этой операции")
                return
                
            logger.debug(f"Доступ разрешен для метода {method_name} с токеном типа {token_type}")
            return await func(self, *args, **kwargs)
            
        return wrapper
    return decorator 