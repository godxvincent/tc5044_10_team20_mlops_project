"""
Monitor de performance del modelo y comparación con baseline.
"""

from typing import Dict

from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from mlops.base.logger import BaseLogger


class PerformanceMonitor(BaseLogger):
    """
    Monitorea performance del modelo y compara con baseline.

    Esta clase evalúa el modelo en datos con drift y compara
    las métricas con las métricas de baseline.
    """

    def __init__(self, model: Pipeline, baseline_metrics: Dict[str, float]):
        """
        Inicializa el monitor de performance.

        Args:
            model: Modelo entrenado (Pipeline de sklearn)
            baseline_metrics: Diccionario con métricas de baseline
        """
        super().__init__(self.__class__.__name__)
        self.model = model
        self.baseline_metrics = baseline_metrics.copy()
        self.logger.info("PerformanceMonitor inicializado")

    def evaluate(self, X: DataFrame, y: Series) -> Dict[str, float]:
        """
        Evalúa el modelo y retorna métricas.

        Args:
            X: Features de entrada
            y: Valores reales (target)

        Returns:
            Diccionario con métricas de evaluación
        """
        # Si hay NaN en los datos, imputarlos antes de la predicción
        # El pipeline debería manejar NaN, pero PCA falla si hay NaN antes de que el imputer se ejecute
        # Por lo tanto, imputamos manualmente usando la mediana (mismo método que en el entrenamiento)
        if X.isna().any().any():
            self.logger.debug("Datos contienen NaN, imputando con mediana antes de la predicción")
            X_imputed = X.copy()
            # Imputar NaN con la mediana de cada columna (mismo método usado en el entrenamiento)
            numeric_cols = X_imputed.select_dtypes(include=["float64", "int64"]).columns
            X_imputed[numeric_cols] = X_imputed[numeric_cols].fillna(X_imputed[numeric_cols].median())
            y_pred = self.model.predict(X_imputed)
        else:
            # No hay NaN, predicción normal
            y_pred = self.model.predict(X)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        }

        self.logger.info(f"Métricas calculadas: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        return metrics

    def compare_with_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Compara métricas actuales con baseline.

        Args:
            current_metrics: Métricas del modelo en datos con drift

        Returns:
            Diccionario con comparación de métricas
        """
        comparison = {}

        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            if metric_name not in self.baseline_metrics or metric_name not in current_metrics:
                continue

            baseline_value = self.baseline_metrics[metric_name]
            current_value = current_metrics[metric_name]

            drop = baseline_value - current_value
            drop_percent = (drop / baseline_value * 100) if baseline_value > 0 else 0.0

            comparison[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "drop": drop,
                "drop_percent": drop_percent,
            }

        self.logger.info("Comparación con baseline completada")
        return comparison
