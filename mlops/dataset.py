from mlops.base.steps import DataLoaderBase
from typing import Dict, Tuple
from pandas import DataFrame

class DataLoader(DataLoaderBase):

    def __init__(self):
        super().__init__()

    def loadFile(self, file_name:str) -> Dict:
        pass

    def getShape(self) -> Tuple[int, int]:
        pass

    def getStatistics(self) -> DataFrame:
        pass    