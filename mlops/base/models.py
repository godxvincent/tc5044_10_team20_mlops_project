from abc import ABC, abstractmethod
from typing import Dict, Optional
from pandas import DataFrame
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


def defaultFunctionTest(a:int, b:int) -> int:
    return a+b

class MLPipelineBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loadData(self, file_name:str):
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


class FeatureEngineProcessorBase(ABC):

    """
        Esta clase define los metodos minimos que deberia tener cualquier clase encargada de hacer limpieza de datos, así como ingenieria de 
        caracteristicas. Esta clase al final deberia retornar un Pipeline con los pasos
        Por default se espera que esta clase tenga como minimo el dataset al cual se le aplicará la limpieza de datos.
    """
    def __init__(self):
        self.__PCAfeaturesScaled = False
        super().__init__()



    def createPipeline(self) -> Pipeline:

        steps = []
        imputer = self.__createImputer()
        if imputer:
            steps.append(('imputer', imputer))

        standardizer = self.__createStandardizer()
        if standardizer: 
            steps.append(('standardizer', standardizer))

        outlierProcessor = self.__createOutlierProcessor()
        if outlierProcessor: 
            steps.append(('outlierProcessor', outlierProcessor))
        
        scaler = self.__createScaler()
        if scaler: 
            steps.append(('scaler', scaler))

        featureReducer = self.__createFeatureReducer()
        if featureReducer and self.__PCAfeaturesScaled:
            steps.append(('PCA', featureReducer))

        return Pipeline(steps=steps)
        



    @abstractmethod
    def __createImputer(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de imputar los valores faltantes.
        """
        pass

    @abstractmethod
    def __createStandardizer(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de estandarizar los valores como la clase.
        """
        pass

    @abstractmethod
    def __createScaler(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de escalar los valores de un conjunto de features.
        """
        pass

    @abstractmethod
    def __createOutlierProcessor(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de lidiar con los valores atipicos.
        """
        pass


    def __createFeatureReducer(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de definir la estrategía para reducir el número de features necesarios para el modelo.
            Por ejemplo hacer uso de PCA.
            No olvidar que para aplicar PCA se requiere hacer scaling the las columnas que se van a reducir. Para garantizar 
            esto se ha añadido el atributo __PCAfeaturesScaled que debe ser cambiado a True si este metodo si hizo el scaler de 
            los feature para PCA.
        """
        pass

