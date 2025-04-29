import json
import os
import datetime
import uuid
import pathlib
import shutil
import sqlite3
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple

from rectools.models.base import ModelBase
from rectools.dataset import Dataset

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Сервис для управления моделями и хранения истории обучения"""
    _instance = None
    _db_path = "data/model_registry.db"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_db()
        return cls._instance
    
    def _initialize_db(self):
        """Инициализирует базу данных для хранения информации о моделях"""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу для моделей
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            parameters TEXT NOT NULL,
            metrics TEXT,
            status TEXT NOT NULL,
            file_path TEXT,
            dataset_path TEXT,
            is_active BOOLEAN DEFAULT 0
        )
        ''')
        
        # Создаем таблицу для заданий на обучение
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_jobs (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            scheduled_at TIMESTAMP NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT NOT NULL,
            parameters TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_model(self, name: str, parameters: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Регистрирует новую модель в реестре
        
        Args:
            name: Имя модели
            parameters: Параметры модели
            metrics: Метрики качества модели
            
        Returns:
            Идентификатор модели
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        # Определяем версию модели
        cursor.execute("SELECT MAX(version) FROM models WHERE name = ?", (name,))
        max_version = cursor.fetchone()[0] or 0
        new_version = max_version + 1
        
        # Генерируем уникальный ID модели
        model_id = str(uuid.uuid4())
        
        # Сохраняем информацию о модели
        cursor.execute(
            "INSERT INTO models (id, name, version, created_at, parameters, metrics, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                model_id,
                name,
                new_version,
                datetime.datetime.now().isoformat(),
                json.dumps(parameters),
                json.dumps(metrics) if metrics else None,
                "created"
            )
        )
        
        conn.commit()
        conn.close()
        
        return model_id
    
    def update_model_status(self, model_id: str, status: str, metrics: Optional[Dict[str, Any]] = None,
                           file_path: Optional[str] = None, dataset_path: Optional[str] = None):
        """
        Обновляет статус и метаданные модели
        
        Args:
            model_id: Идентификатор модели
            status: Новый статус модели (created, training, trained, failed)
            metrics: Метрики качества модели
            file_path: Путь к файлу модели
            dataset_path: Путь к файлу датасета
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        update_fields = ["status = ?"]
        params = [status]
        
        if metrics is not None:
            update_fields.append("metrics = ?")
            params.append(json.dumps(metrics))
        
        if file_path is not None:
            update_fields.append("file_path = ?")
            params.append(file_path)
            
        if dataset_path is not None:
            update_fields.append("dataset_path = ?")
            params.append(dataset_path)
        
        params.append(model_id)
        
        cursor.execute(
            f"UPDATE models SET {', '.join(update_fields)} WHERE id = ?",
            params
        )
        
        conn.commit()
        conn.close()
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Получает информацию о модели
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            Словарь с информацией о модели
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        model_row = cursor.fetchone()
        
        conn.close()
        
        if not model_row:
            return {}
            
        model_info = dict(model_row)
        model_info['parameters'] = json.loads(model_info['parameters'])
        
        if model_info.get('metrics'):
            model_info['metrics'] = json.loads(model_info['metrics'])
            
        return model_info
    
    def get_active_model(self) -> Dict[str, Any]:
        """
        Получает активную модель
        
        Returns:
            Словарь с информацией об активной модели
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models WHERE is_active = 1")
        model_row = cursor.fetchone()
        
        conn.close()
        
        if not model_row:
            return {}
            
        model_info = dict(model_row)
        model_info['parameters'] = json.loads(model_info['parameters'])
        
        if model_info.get('metrics'):
            model_info['metrics'] = json.loads(model_info['metrics'])
            
        return model_info
    
    def set_active_model(self, model_id: str) -> bool:
        """
        Устанавливает модель как активную
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            True, если модель успешно установлена как активная
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        # Проверяем существование модели
        cursor.execute("SELECT id FROM models WHERE id = ?", (model_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Сначала деактивируем все модели
        cursor.execute("UPDATE models SET is_active = 0")
        
        # Затем активируем указанную модель
        cursor.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def list_models(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Получает список моделей
        
        Args:
            limit: Максимальное количество моделей
            offset: Смещение для пагинации
            
        Returns:
            Список словарей с информацией о моделях
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM models ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        models = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        for model in models:
            model['parameters'] = json.loads(model['parameters'])
            if model.get('metrics'):
                model['metrics'] = json.loads(model['metrics'])
        
        return models
    
    def save_model_files(self, model_id: str, model: ModelBase, dataset: Dataset) -> Tuple[str, str]:
        """
        Сохраняет файлы модели и датасета
        
        Args:
            model_id: Идентификатор модели
            model: Объект модели
            dataset: Объект датасета
            
        Returns:
            Кортеж с путями к файлам модели и датасета
        """
        # Создаем директорию для хранения моделей
        model_dir = pathlib.Path(f"data/models/{model_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Пути к файлам
        model_path = str(model_dir / "model.pkl")
        dataset_path = str(model_dir / "dataset.pkl")
        
        # Сохраняем модель
        model.save(model_path)
        
        # Сохраняем датасет
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Обновляем информацию о модели
        self.update_model_status(
            model_id=model_id,
            status="trained",
            file_path=model_path,
            dataset_path=dataset_path
        )
        
        return model_path, dataset_path
    
    def load_model_files(self, model_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Получает пути к файлам модели и датасета
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            Кортеж с путями к файлам модели и датасета
        """
        model_info = self.get_model_info(model_id)
        
        if not model_info:
            return None, None
            
        return model_info.get('file_path'), model_info.get('dataset_path')
    
    def schedule_training_job(self, parameters: Dict[str, Any], scheduled_at: datetime.datetime) -> str:
        """
        Планирует задание на обучение модели
        
        Args:
            parameters: Параметры обучения
            scheduled_at: Время запуска задания
            
        Returns:
            Идентификатор задания
        """
        job_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO training_jobs (id, scheduled_at, status, parameters) VALUES (?, ?, ?, ?)",
            (
                job_id,
                scheduled_at.isoformat(),
                "scheduled",
                json.dumps(parameters)
            )
        )
        
        conn.commit()
        conn.close()
        
        return job_id
    
    def update_training_job(self, job_id: str, status: str, model_id: Optional[str] = None,
                           started_at: Optional[datetime.datetime] = None,
                           completed_at: Optional[datetime.datetime] = None):
        """
        Обновляет статус задания на обучение
        
        Args:
            job_id: Идентификатор задания
            status: Новый статус задания (scheduled, running, completed, failed)
            model_id: Идентификатор созданной модели
            started_at: Время начала выполнения
            completed_at: Время завершения выполнения
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        update_fields = ["status = ?"]
        params = [status]
        
        if model_id is not None:
            update_fields.append("model_id = ?")
            params.append(model_id)
        
        if started_at is not None:
            update_fields.append("started_at = ?")
            params.append(started_at.isoformat())
            
        if completed_at is not None:
            update_fields.append("completed_at = ?")
            params.append(completed_at.isoformat())
        
        params.append(job_id)
        
        cursor.execute(
            f"UPDATE training_jobs SET {', '.join(update_fields)} WHERE id = ?",
            params
        )
        
        conn.commit()
        conn.close()
    
    def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """
        Получает информацию о задании на обучение
        
        Args:
            job_id: Идентификатор задания
            
        Returns:
            Словарь с информацией о задании
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
        job_row = cursor.fetchone()
        
        conn.close()
        
        if not job_row:
            return {}
            
        job_info = dict(job_row)
        job_info['parameters'] = json.loads(job_info['parameters'])
            
        return job_info
    
    def list_training_jobs(self, status: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Получает список заданий на обучение
        
        Args:
            status: Фильтр по статусу
            limit: Максимальное количество заданий
            offset: Смещение для пагинации
            
        Returns:
            Список словарей с информацией о заданиях
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status:
            cursor.execute(
                "SELECT * FROM training_jobs WHERE status = ? ORDER BY scheduled_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            )
        else:
            cursor.execute(
                "SELECT * FROM training_jobs ORDER BY scheduled_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            
        jobs = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        for job in jobs:
            job['parameters'] = json.loads(job['parameters'])
        
        return jobs
    
    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """
        Получает список заданий, которые должны быть выполнены
        
        Returns:
            Список словарей с информацией о заданиях
        """
        now = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM training_jobs WHERE status = 'scheduled' AND scheduled_at <= ? ORDER BY scheduled_at",
            (now,)
        )
        jobs = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        for job in jobs:
            job['parameters'] = json.loads(job['parameters'])
        
        return jobs 