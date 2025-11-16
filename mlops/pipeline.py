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
        # self.__model_trainer = ModelTrainer()

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
            # ⭐ AJUSTE: Asumimos que ModelTrainer.evaluate() almacena las métricas
            # y tiene un método público para devolverlas (usaremos get_metrics).
            return self.__model_trainer._calculate_metrics()
        except AttributeError as e:
            self.logger.error(f"Error al recuperar las métricas: Modelo no configurado o evaluado. {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error al recuperar las métricas del modelo: {e}")
            raise e


if __name__ == "__main__":
    mlpipeline = MLPipeline()
    mlpipeline.load_data_step("turkish_music_emotion_modified.csv")
    mlpipeline.clean_up_data_step()
    mlpipeline.feature_engineering_step()
    mlpipeline.train_step()
    mlpipeline.evaluate_step()
