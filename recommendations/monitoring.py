import asyncio

import pandas as pd

from recommendations.data_preparer import DataPrepareService
from recommendations.rec_service import ModelManager, logger


class MonitoringService:  # NEW
    def __init__(self):
        self.performance_history = []

    async def track_performance(self):
        model, dataset = await ModelManager().get_model()
        _, test = await DataPrepareService.get_interactions()

        metrics = await ModelManager().calculate_metrics(model, dataset, test)
        self.performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'version': ModelManager().current_version,
            'metrics': metrics
        })

        self._check_metrics_degradation(metrics)

    def _check_metrics_degradation(self, metrics: pd.DataFrame, threshold: float = 0.1):
        if len(self.performance_history) < 2:
            return

        prev = self.performance_history[-2]['metrics']
        current = metrics

        degradation = {
            metric: (prev[metric] - current[metric]) / prev[metric]
            for metric in prev.columns
        }

        if any(d > threshold for d in degradation.values()):
            logger.warning(f"Деградация метрик: {degradation}")
            self.trigger_retrain()

    @staticmethod
    def trigger_retrain(self):
        logger.info("Запуск экстренного переобучения модели...")
        asyncio.create_task(ModelManager().train())