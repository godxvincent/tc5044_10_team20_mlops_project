from abc import ABC, abstractmethod
from typing import Dict, Self
from pandas import DataFrame

class MLPipelineBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loadData(self, file_name):
        pass

    @abstractmethod
    def cleanUpData(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass




class DataLoaderBase(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loadFile(self, file_name:str) -> Dict:
        """
            La implementación concreta deberia 
            * Manejar correctamente los errores en caso de que el archivo no exista.
            * Retornar la información relevante del dataset cargado
                * Shape
                * Información estadistica base.
        """
        pass

    @abstractmethod
    def getShape(self):
        pass

    @abstractmethod
    def getStatistics(self) -> DataFrame:
        #TODO: Definir funcion comun para poder retornar la misma estructura en el DataLoader como en el DataCleaner dado un dataset.
        pass 



class DataCleanerBase(ABC):

    def __init__(self, dataframe:DataFrame):
        self.dataframe = dataframe
        super().__init__()

    @abstractmethod
    def __imputeValues(self) -> Self:
        """Este metodo será privado ya que no es necesario que se exponga al consumidor de la clase"""
        pass

    @abstractmethod
    def __removeOutliers(self) -> Self:
        """Este metodo será privado ya que no es necesario que se exponga al consumidor de la clase"""
        pass

    @abstractmethod
    def __estandardizeValues(self) -> Self:
        """Este metodo será privado ya que no es necesario que se exponga al consumidor de la clase"""
        pass 

    def execute(self) -> DataFrame:
        """
            Este metodo debe ejecutar cada uno de los metodos privados 
            El metodo debe retornar la misma estructura de datos que retorne el metodo 
            getStatistics de la clase DataLoaderBase para que sean equiparables.
            TODO: Verificar este comentario :up:
        """
        # TODO: Revisar si este es el correcto orden en el que se deben implementar las limpiezas
        self.__removeOutliers().__estandardizeValues().__imputeValues()
        # TODO: Pendiente definir en este punto si esta es la estructura adecuada para devolver los datos.
        return self.dataframe.describe()
