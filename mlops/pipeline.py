from typing import Any, Dict, Self, Tuple

from pandas import DataFrame

from mlops.base.steps import MLPipelineBase
from mlops.dataset import DataLoader
from mlops.features import FeatureEngineProcessor
from mlops.modeling.train import ModelTrainer


class MLPipeline(MLPipelineBase):
    def __init__(self):
        super().__init__()
        self.__data_loader = DataLoader()
        self.__feature_engine_processor = FeatureEngineProcessor()

    def load_data_step(self, file_name: str) -> Self:
        try:
            self.__data_loader.load_file(file_name)
            return self
        except Exception as e:
            self.logger.error(f"Error al cargar el archivo: {e}")
            raise e

    def clean_up_data_step(self) -> Self:
        try:
            self.__data_loader.run_cleaning_up()
            return self
        except Exception as e:
            self.logger.error(f"Error al limpiar los datos: {e}")
            raise e

    def feature_engineering_step(self) -> Self:
        try:
            self.__model_trainer = ModelTrainer(
                self.__feature_engine_processor.createPipeline(),
                self.__data_loader.get_train_test_dataset(),
            )
            return self
        except Exception as e:
            self.logger.error(f"Error al intentar crear el pipeline de feature engineering: {e}")
            raise e

    def train_step(self) -> Self:
        try:
            self.__model_trainer.train()
            return self
        except Exception as e:
            self.logger.error(f"Error al intentar entrenar el modelo: {e}")
            raise e

    def evaluate_step(self) -> Self:
        try:
            self.__model_trainer.evaluate()
            return self
        except Exception as e:
            self.logger.error(f"Error al intentar evaluar el performance del modelo: {e}")
            raise e

    def get_dataframe_statistics(self) -> DataFrame:
        try:
            return self.__data_loader.get_statistics()
        except Exception as e:
            self.logger.error(f"Error al intentar recuperar estadísticas: {e}")
            raise e

    def get_dataframe_shape(self) -> Tuple:
        try:
            return self.__data_loader.get_shape()
        except Exception as e:
            self.logger.error(f"Error al recuperar las dimensiones: {e}")
            raise e

    def get_original_dataframe_statistics(self) -> DataFrame:
        try:
            return self.__data_loader.get_original_statistics()
        except Exception as e:
            self.logger.error(f"Error al intentar recuperar estadísticas: {e}")
            raise e

    def get_original_dataframe_shape(self) -> Tuple:
        try:
            return self.__data_loader.get_original_shape()
        except Exception as e:
            self.logger.error(f"Error al recuperar las dimensiones: {e}")
            raise e

    def get_model_params(self) -> Dict[str, Any]:
        """
        Recupera los parámetros y la importancia de las características del modelo entrenado.
        """
        try:
            return self.__model_trainer.get_model_attributes()
        except AttributeError as e:
            self.logger.error(f"Error al recuperar los parámetros: Modelo no configurado o entrenado. {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error al recuperar los parámetros del modelo: {e}")
            raise e

    def get_model_metrics(self) -> Dict[str, float]:
        """
        Recupera las métricas de evaluación del modelo (Accuracy, Precision, Recall, F1).
        """
        try:
            return self.__model_trainer.get_performance_metrics()
        except AttributeError as e:
            self.logger.error(f"Error al recuperar las métricas: Modelo no configurado o evaluado. {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error al recuperar las métricas del modelo: {e}")
            raise e

    def drift_monitoring_step(self, drift_scenarios=None):
        """
        Ejecuta evaluación de data drift.

        Args:
            drift_scenarios: Escenarios personalizados de drift.
                           Si es None, usa los 3 escenarios predefinidos:
                           - covariate_shift_mild
                           - covariate_shift_moderate
                           - missing_data_10pct

        Returns:
            Diccionario con resultados de la evaluación de drift por escenario
        """
        try:
            # Si no se proporcionan escenarios, usar los 3 escenarios predefinidos
            if drift_scenarios is None:
                drift_scenarios = [
                    {
                        "name": "covariate_shift_mild",
                        "type": "covariate_shift",
                        "params": {
                            "features": ["_Tempo_Mean", "_RMSenergy_Mean"],
                            "mean_shift": 0.1,  # 10% de la desviación estándar
                            "variance_factor": 1.05,
                        },
                    },
                    {
                        "name": "covariate_shift_moderate",
                        "type": "covariate_shift",
                        "params": {
                            "features": ["_Tempo_Mean", "_RMSenergy_Mean", "_Brightness_Mean"],
                            "mean_shift": 0.5,  # 50% de la desviación estándar
                            "variance_factor": 1.2,
                        },
                    },
                    {
                        "name": "missing_data_10pct",
                        "type": "missing_data",
                        "params": {
                            "missing_rate": 0.1,  # 10% de valores faltantes
                            "columns": None,  # Todas las columnas numéricas
                        },
                    },
                ]

            results = self.__model_trainer.evaluate_drift(drift_scenarios)
            return results
        except Exception as e:
            self.logger.error(f"Error en evaluación de drift: {e}")
            raise e
