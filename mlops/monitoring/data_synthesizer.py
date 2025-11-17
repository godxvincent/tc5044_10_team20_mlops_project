"""
Generador de datos sintéticos con diferentes tipos de data drift.
"""

import numpy as np
from pandas import DataFrame

from mlops.base.logger import BaseLogger


class DataSynthesizer(BaseLogger):
    """
    Genera datasets sintéticos con diferentes tipos de data drift.

    Esta clase permite simular cambios en la distribución de los datos
    para evaluar el impacto en el modelo.
    """

    def __init__(self, reference_data: DataFrame):
        """
        Inicializa el generador de datos sintéticos.

        Args:
            reference_data: Dataset de referencia (baseline) sin drift
        """
        super().__init__(self.__class__.__name__)
        self.reference_data = reference_data.copy()
        self.logger.info(f"DataSynthesizer inicializado con {len(reference_data)} filas")

    def apply_covariate_shift(
        self, features: list[str], mean_shift: float = 0.0, variance_factor: float = 1.0
    ) -> DataFrame:
        """
        Aplica covariate shift a features específicas.

        Covariate shift es un cambio en la distribución de las variables
        de entrada (X) manteniendo la relación P(Y|X).

        Args:
            features: Lista de nombres de features a modificar
            mean_shift: Desplazamiento de la media (en unidades de desviación estándar)
            variance_factor: Factor de multiplicación de la varianza

        Returns:
            DataFrame con los datos modificados (con drift)
        """
        drifted_data = self.reference_data.copy()

        for feature in features:
            if feature not in drifted_data.columns:
                self.logger.warning(f"Feature '{feature}' no encontrada, omitiendo")
                continue

            # Calcular desviación estándar de la feature
            std = drifted_data[feature].std()

            # Aplicar desplazamiento de media
            if mean_shift != 0.0:
                shift_amount = mean_shift * std
                drifted_data[feature] = drifted_data[feature] + shift_amount
                self.logger.debug(f"Aplicado shift de {shift_amount:.4f} a {feature}")

            # Aplicar cambio de varianza
            if variance_factor != 1.0:
                mean_value = drifted_data[feature].mean()
                drifted_data[feature] = (drifted_data[feature] - mean_value) * variance_factor + mean_value
                self.logger.debug(f"Aplicado factor de varianza {variance_factor} a {feature}")

        self.logger.info(
            f"Covariate shift aplicado a {len(features)} features: "
            f"mean_shift={mean_shift}, variance_factor={variance_factor}"
        )

        return drifted_data

    def apply_missing_data_drift(self, missing_rate: float = 0.1, columns: list[str] | None = None) -> DataFrame:
        """
        Aplica missing data drift introduciendo valores faltantes.

        Missing data drift simula la aparición de valores faltantes
        en features que antes no los tenían.

        Args:
            missing_rate: Proporción de valores a convertir en NaN (0.0-1.0)
            columns: Lista de columnas a afectar. Si es None, afecta todas las numéricas.

        Returns:
            DataFrame con valores faltantes introducidos
        """
        drifted_data = self.reference_data.copy()

        # Si no se especifican columnas, usar todas las numéricas
        if columns is None:
            columns = list(drifted_data.select_dtypes(include=[np.number]).columns)

        # Filtrar columnas que existen
        valid_columns = [col for col in columns if col in drifted_data.columns]
        if not valid_columns:
            self.logger.warning("No se encontraron columnas válidas para aplicar missing data drift")
            return drifted_data

        # Introducir valores faltantes aleatoriamente usando máscara
        np.random.seed(42)  # Para reproducibilidad
        for col in valid_columns:
            mask = np.random.random(len(drifted_data)) < missing_rate
            drifted_data.loc[mask, col] = np.nan

        self.logger.info(
            f"Missing data drift aplicado: {missing_rate * 100:.1f}% de valores faltantes "
            f"en {len(valid_columns)} columnas"
        )

        return drifted_data
