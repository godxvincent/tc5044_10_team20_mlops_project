"""
Evaluador principal de data drift que orquesta todo el proceso.
"""

from typing import Any, Dict

from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline

from mlops.base.logger import BaseLogger
from mlops.monitoring.data_synthesizer import DataSynthesizer
from mlops.monitoring.drift_detector import DriftDetector
from mlops.monitoring.performance_monitor import PerformanceMonitor


class DriftEvaluator(BaseLogger):
    """
    Evalúa data drift y su impacto en el modelo.

    Esta clase orquesta todo el proceso:
    1. Genera datos sintéticos con drift
    2. Detecta drift estadístico
    3. Evalúa performance del modelo
    4. Compara con baseline
    """

    def __init__(
        self,
        model: Pipeline,
        reference_data: DataFrame,
        reference_target: Series,
        baseline_metrics: Dict[str, float],
    ):
        """
        Inicializa el evaluador de drift.

        Args:
            model: Modelo entrenado
            reference_data: Dataset de referencia (X)
            reference_target: Target de referencia (y)
            baseline_metrics: Métricas de baseline del modelo
        """
        super().__init__(self.__class__.__name__)
        self.model = model
        self.reference_data = reference_data.copy()
        self.reference_target = reference_target.copy()
        self.baseline_metrics = baseline_metrics.copy()

        # Inicializar componentes
        self.synthesizer = DataSynthesizer(reference_data)
        self.detector = DriftDetector(reference_data)
        self.monitor = PerformanceMonitor(model, baseline_metrics)

        self.logger.info("DriftEvaluator inicializado")

    def evaluate_drift_scenario(self, drift_type: str, drift_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa un escenario de drift específico.

        Args:
            drift_type: Tipo de drift ('covariate_shift', etc.)
            drift_params: Parámetros del drift (features, mean_shift, etc.)

        Returns:
            Diccionario con resultados completos de la evaluación
        """
        self.logger.info(f"Evaluando escenario: {drift_type}")

        # 1. Generar datos con drift
        if drift_type == "covariate_shift":
            features = drift_params.get("features", [])
            mean_shift = drift_params.get("mean_shift", 0.0)
            variance_factor = drift_params.get("variance_factor", 1.0)

            drifted_data = self.synthesizer.apply_covariate_shift(features, mean_shift, variance_factor)
        elif drift_type == "missing_data":
            missing_rate = drift_params.get("missing_rate", 0.1)
            columns = drift_params.get("columns", None)

            drifted_data = self.synthesizer.apply_missing_data_drift(missing_rate, columns)
        else:
            raise ValueError(f"Tipo de drift no soportado: {drift_type}")

        # 2. Detectar drift estadístico
        drift_metrics = self.detector.detect_drift_all_features(drifted_data)

        # 3. Evaluar performance del modelo
        performance_metrics = self.monitor.evaluate(drifted_data, self.reference_target)

        # 4. Comparar con baseline
        comparison = self.monitor.compare_with_baseline(performance_metrics)

        # 5. Preparar resultados
        results = {
            "drift_type": drift_type,
            "drift_params": drift_params,
            "drift_metrics": drift_metrics,
            "performance_metrics": performance_metrics,
            "comparison": comparison,
        }

        self.logger.info("Evaluación de drift completada")
        return results
