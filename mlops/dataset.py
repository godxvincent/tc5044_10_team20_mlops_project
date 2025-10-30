from dataclasses import dataclass
from typing import Dict, Tuple

from pandas import DataFrame, read_csv

from mlops.base.steps import DataLoaderBase
from mlops.config import BaseDataClassModel, MLConfigLoader


@dataclass
class DataLoaderConfig(BaseDataClassModel):
    test_data_percentage: str

    def __init__(self, **kwargs):
        """
        Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)


class DataLoader(DataLoaderBase):

    def __init__(self):
        self.data_loader_config = MLConfigLoader().getParameter("data_loader", DataLoaderConfig())
        super().__init__()

    def loadFile(self, file_name: str) -> Dict:
        try:
            self.df = read_csv(file_name)
        except FileNotFoundError as e:
            self.logger.debug(f"File {file_name} not found")
            raise e

    def getShape(self) -> Tuple[int, int]:
        return self.df.shape

    def getStatistics(self) -> DataFrame:
        return self.df.describe()

    # def getTrainTestDataSet(self) -> Dict[str, DataFrame]:
    # test_size = self.data_loader_config.test_data_percentage
    # Deje esta variable solo comentada, si no va a ser usada, por favor, borrarla.

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    # )


if __name__ == "__main__":
    dataLoader = DataLoader()
    dataLoader.loadFile("data/external/turkish_music_emotion_original.csv")
    print(dataLoader.getShape())
    print(dataLoader.getStatistics())
