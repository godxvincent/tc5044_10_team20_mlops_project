from abc import ABC, abstractmethod
from typing import Dict, Optional

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from mlops.base.logger import BaseLogger


class MLPipelineBase(ABC, BaseLogger):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    @abstractmethod
    def loadData(self, file_name: str):
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


class DataLoaderBase(ABC, BaseLogger):

    def __init__(self):
        super().__init__(self.__class__.__name__)

    @abstractmethod
    def loadFile(self, file_name: str) -> Dict:
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
        # TODO: Definir funcion comun para poder retornar la misma estructura en el DataLoader como en el DataCleaner dado un dataset.
        pass

    @abstractmethod
    def getTrainTestDataSet(self) -> Dict[str, DataFrame]:
        pass


class FeatureEngineProcessorBase(ABC, BaseLogger):
    """
    Esta clase define los metodos minimos que deberia tener cualquier clase encargada de hacer limpieza de datos, así como ingenieria de
    caracteristicas. Esta clase al final deberia retornar un Pipeline con los pasos
    Por default se espera que esta clase tenga como minimo el dataset al cual se le aplicará la limpieza de datos.
    """

    def __init__(self):
        self.__PCAfeaturesScaled = False
        super().__init__(self.__class__.__name__)

    def createPipeline(self) -> Pipeline:

        steps = []
        imputer = self._createImputer()
        if imputer:
            steps.append(("imputer", imputer))

        standardizer = self._createStandardizer()
        if standardizer:
            steps.append(("standardizer", standardizer))

        outlierProcessor = self._createOutlierProcessor()
        if outlierProcessor:
            steps.append(("outlierProcessor", outlierProcessor))

        scaler = self._createScaler()
        if scaler:
            steps.append(("scaler", scaler))

        featureReducer = self._createFeatureReducer()
        if featureReducer and self.__PCAfeaturesScaled:
            steps.append(("PCA", featureReducer))

        return Pipeline(steps=steps)

    @abstractmethod
    def _createImputer(self) -> Optional[ColumnTransformer]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de imputar los valores faltantes.
        """
        pass

    @abstractmethod
    def _createStandardizer(self) -> Optional[ColumnTransformer]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de estandarizar los valores como la clase.
        """
        pass

    @abstractmethod
    def _createScaler(self) -> Optional[ColumnTransformer]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de escalar los valores de un conjunto de features.
        """
        pass

    @abstractmethod
    def _createOutlierProcessor(self) -> Optional[ColumnTransformer]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de lidiar con los valores atipicos.
        """
        pass

    @abstractmethod
    def _createFeatureReducer(self) -> Optional[ColumnTransformer]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que defina la estrategia para reducir el número de features necesarios (p. ej., usando PCA).
        Para aplicar PCA se requiere hacer *scaling* a las columnas que se van a reducir. Para garantizarlo,
        se añadió el atributo `__PCAfeaturesScaled`, que debe cambiarse a `True` si este método aplicó el
        *scaler* a los features usados por PCA.
        """
        pass


class ModelTrainerBase(ABC, BaseLogger):

    def __init__(self, pipeline: Pipeline, datasets: Dict[str, DataFrame]):
        """
        Pipeline: Este parametro corresponde al pipeline que se encarga de transformar los datos, deberia
        corresponder al tipo de datos que develve la clase FeatureEngineeringProcessorBase en su metodo
        createPipeline(self) -> Pipeline.
        datasets: Este parametro corresponde a un diccionario de la forma
        {
            "trainX": trainXDataFrame,
            "trainY": trainYDataFrame,
            "testX": testXDataFrame,
            "testY": testYDataFrame,
        }
        Donde trainX, trainY, corresponden a los features de entrada y la variable de respuesta del conjunto
        de datos de entrenamiento mientras que testX y testY corresponden a los features y las variables de
        respuesta del conjunto de datos de prueba.

        """
        self.pipeline = pipeline
        self.datasets = datasets
        super().__init__(self.__class__.__name__)

    @abstractmethod
    def _createModel(self) -> Pipeline:
        """
        La implementación concreta de este metodo debería contener todos los pasos para crear la instancia del modelo y
        añadirlo al pipeline.
        """
        pass

    @abstractmethod
    def train(self):
        """
        La implementación concreta de esta función debera ejecutar el entrenamiento del modelo, con los datos
        de entrenamiento.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        La implementación concreta de esta función debera ejecutar la evaluación del modelo con los datos de test.
        Y retornar los valores de las metricas evaluadas del modelo.
        """
        pass

    @abstractmethod
    def predict(self) -> DataFrame:
        """
        La implementación concreta de esta función debe ejecutar el modelo contra un conjunto de pruebas y devolver la predicción.
        """
        pass

    @abstractmethod
    def get_model_attributes(self, **kwargs):
        """
        No estoy 100% seguro aun como deberia funcionar esta función.
        """
        pass
