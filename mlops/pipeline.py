import os
import pathlib
import sys
from typing import Self

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
            self.__feature_engine_processor.createPipeline()
            self.__model_trainer = ModelTrainer(
                self.__feature_engine_processor.createPipeline(), self.__data_loader.get_train_test_dataset()
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


if __name__ == "__main__":

    sys.path.append(str(pathlib.Path(__name__).parent.parent.absolute().parent))
    dynaconf_env = os.getenv("ENV_FOR_DYNACONF", None)
    if not dynaconf_env:
        os.environ["ENV_FOR_DYNACONF"] = "local"

    mlpipeline = MLPipeline()
    mlpipeline.load_data_step("turkish_music_emotion_modified.csv")
    mlpipeline.clean_up_data_step()
    mlpipeline.feature_engineering_step()
    mlpipeline.train_step()
    mlpipeline.evaluate_step()
