"""
Detector de data drift usando alibi-detect con Kolmogorov-Smirnov (KS).

Esta implementación usa alibi-detect para detectar drift de forma robusta
y probada, reemplazando la implementación manual de PSI.
"""

from typing import Any, Dict, Optional

import numpy as np
from alibi_detect.cd import KSDrift
from pandas import DataFrame

from mlops.base.logger import BaseLogger


class DriftDetector(BaseLogger):
    """
    Detecta data drift usando Kolmogorov-Smirnov (KS) de alibi-detect.

    KS es un test estadístico univariado que compara las funciones de distribución
    acumulada (CDF) de dos muestras. Se aplica a cada feature individualmente,
    permitiendo identificar qué features específicas tienen drift.
    """

    def __init__(self, reference_data: DataFrame, p_val: float = 0.05):
        """
        Inicializa el detector de drift usando KS de alibi-detect.

        Args:
            reference_data: Dataset de referencia (baseline) sin drift
            p_val: Valor p para determinar drift (default: 0.05)
        """
        super().__init__(self.__class__.__name__)
        self.reference_data = reference_data.copy()
        self.p_val = p_val

        # Preparar datos de referencia (solo numéricos para KS)
        numeric_data = reference_data.select_dtypes(include=[np.number])
        self.x_ref = numeric_data.values
        self.feature_names = numeric_data.columns.tolist()

        if len(self.feature_names) == 0:
            raise ValueError("No se encontraron features numéricas en reference_data")

        # Crear detector KS
        try:
            self.detector = KSDrift(self.x_ref, p_val=p_val, correction="bonferroni")
            self.logger.info(
                f"DriftDetector (KS) inicializado con {len(reference_data)} filas "
                f"y {len(self.feature_names)} features numéricas"
            )
        except Exception as e:
            self.logger.error(f"Error inicializando detector KS: {e}")
            raise

    def detect_drift_all_features(self, current_data: DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detecta drift usando KS (univariado por feature).

        KS aplica el test de Kolmogorov-Smirnov a cada feature individualmente,
        permitiendo identificar qué features específicas tienen drift.

        Args:
            current_data: Dataset actual a comparar

        Returns:
            Diccionario con métricas de drift por feature.
            Formato:
            {
                'feature_name': {
                    'p_value': float,
                    'distance': float,
                    'drift_detected': bool,
                    'severity': str
                },
                ...
                '_all_features': {
                    'p_value': float (promedio),
                    'distance': float (promedio),
                    'drift_detected': bool (si alguna feature tiene drift),
                    'severity': str
                }
            }
        """
        # Preparar datos actuales (solo numéricos)
        numeric_data = current_data.select_dtypes(include=[np.number])
        x = numeric_data.values

        # Verificar que tenemos las mismas features
        if x.shape[1] != self.x_ref.shape[1]:
            self.logger.warning(
                f"Número de features diferente: " f"referencia={self.x_ref.shape[1]}, actual={x.shape[1]}"
            )
            return {}

        # Detectar drift usando KS
        try:
            preds = self.detector.predict(x)
        except Exception as e:
            self.logger.error(f"Error detectando drift con KS: {e}")
            return {}

        # Extraer resultados
        # KS retorna:
        # - is_drift: escalar (0 o 1) indicando si hay drift después de corrección Bonferroni
        # - p_val: array con p-values por feature
        # - distance: array con distancias KS por feature
        is_drift_overall = preds["data"]["is_drift"]
        p_val_array = preds["data"].get("p_val", None)
        distance_array = preds["data"].get("distance", None)

        # Asegurar que p_val y distance sean arrays numpy
        if p_val_array is None or not isinstance(p_val_array, np.ndarray):
            p_val_array = np.array([p_val_array] if p_val_array is not None else [np.nan] * len(self.feature_names))

        if distance_array is None or not isinstance(distance_array, np.ndarray):
            distance_array = np.array(
                [distance_array] if distance_array is not None else [np.nan] * len(self.feature_names)
            )

        # Calcular p_val ajustado con Bonferroni para determinar drift por feature
        # Bonferroni ajusta: p_adj = p_val / n_features
        p_val_adjusted = self.p_val / len(self.feature_names)

        # Crear resultados por feature
        results = {}
        p_values = []
        distances = []

        for i, feature in enumerate(self.feature_names):
            p_val_feature = float(p_val_array[i]) if i < len(p_val_array) else np.nan
            distance_feature = float(distance_array[i]) if i < len(distance_array) else np.nan

            # Determinar si esta feature tiene drift (p-value < p_val ajustado)
            is_drift_feature = p_val_feature < p_val_adjusted if not np.isnan(p_val_feature) else False

            results[feature] = {
                "p_value": p_val_feature,
                "distance": distance_feature,
                "drift_detected": is_drift_feature,
                "severity": self._classify_severity(p_val_feature),
            }

            # Acumular para resultado agregado
            if not np.isnan(p_val_feature):
                p_values.append(p_val_feature)
            if not np.isnan(distance_feature):
                distances.append(distance_feature)

        # Resultado agregado
        # is_drift_overall ya considera corrección Bonferroni
        results["_all_features"] = {
            "p_value": float(np.mean(p_values)) if p_values else np.nan,
            "distance": float(np.mean(distances)) if distances else np.nan,
            "drift_detected": bool(is_drift_overall == 1),
            "severity": self._classify_severity(np.mean(p_values) if p_values else np.nan),
        }

        drift_count = sum(
            1
            for key, r in results.items()
            if key != "_all_features" and isinstance(r, dict) and r.get("drift_detected", False)
        )
        # Formatear p_value para logging
        p_val_avg = results["_all_features"]["p_value"]
        p_val_str = f"{p_val_avg:.4f}" if not np.isnan(p_val_avg) else "N/A"
        self.logger.info(
            f"Drift detectado usando KS: {drift_count}/{len(self.feature_names)} features con drift, "
            f"p_value_promedio={p_val_str}"
        )
        return results

    def _classify_severity(self, p_val: Optional[float]) -> str:
        """
        Clasifica la severidad del drift basada en el p-value.

        Args:
            p_val: P-value del test estadístico

        Returns:
            'none', 'moderate', 'significant', o 'error'
        """
        if p_val is None or np.isnan(p_val):
            return "error"
        elif p_val >= 0.1:
            return "none"  # No hay evidencia de drift
        elif p_val >= 0.05:
            return "moderate"  # Drift moderado
        else:
            return "significant"  # Drift significativo
