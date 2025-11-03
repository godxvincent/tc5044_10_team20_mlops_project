from abc import ABC, abstractmethod
from typing import Dict, Optional, Self, Any

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from mlops.base.logger import BaseLogger


class MLPipelineBase(ABC, BaseLogger):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    @abstractmethod
    def load_data_step(self, file_name: str) -> Self:
        pass

    @abstractmethod
    def clean_up_data_step(self) -> Self:
        pass

    @abstractmethod
    def feature_engineering_step(self) -> Self:
        pass

    @abstractmethod
    def train_step(self) -> Self:
        pass

    @abstractmethod
    def evaluate_step(self) -> Self:
        pass


class DataLoaderBase(ABC, BaseLogger):

    def __init__(self):
        super().__init__(self.__class__.__name__)

    @abstractmethod
    def load_file(self, file_name: str) -> Dict:
        """
        La implementación concreta deberia
        * Manejar correctamente los errores en caso de que el archivo no exista.
        * Retornar la información relevante del dataset cargado
            * Shape
            * Información estadistica base.
        """
        pass

    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def get_statistics(self) -> DataFrame:
        # TODO: Definir funcion comun para poder retornar la misma estructura en el DataLoader como en el DataCleaner dado un dataset.
        pass

    @abstractmethod
    def get_train_test_dataset(self) -> Dict[str, DataFrame]:
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

        return Pipeline(steps=steps).set_output(transform="pandas")

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
        self.best_model = None
        self.best_params_ = None
        self.mlflow_tracker = None
        super().__init__(self.__class__.__name__)

    def _initialize_mlflow_tracker(self, model_name: str) -> None:
        """
        Inicializa MLflowTracker si está disponible.

        Args:
            model_name: Nombre del modelo para el experimento de MLflow
        """
        try:
            from mlops.base.tracking import MLflowTracker
            self.mlflow_tracker = MLflowTracker()
            if self.mlflow_tracker.config.enabled:
                self.mlflow_tracker.initialize(model_name=model_name)
                self.logger.info("MLflowTracker inicializado correctamente")
            else:
                self.logger.debug("MLflowTracker deshabilitado en configuración")
        except Exception as e:
            self.logger.warning(f"No se pudo inicializar MLflowTracker: {e}. Continuando sin MLflow.")
            self.mlflow_tracker = None

    def _start_mlflow_run(self) -> None:
        """Inicia un run de MLflow si está disponible."""
        if self.mlflow_tracker and self.mlflow_tracker.config.enabled:
            self.mlflow_tracker.start_run()

    def _end_mlflow_run(self) -> None:
        """Finaliza el run activo de MLflow si existe."""
        if self.mlflow_tracker and self.mlflow_tracker.config.enabled:
            try:
                self.mlflow_tracker.end_run()
            except Exception as e:
                self.logger.error(f"Error finalizando run de MLflow: {e}")

    def _log_mlflow_params(self, params: Dict[str, Any]) -> None:
        """
        Loggea parámetros a MLflow.

        Args:
            params: Diccionario con los parámetros a loggear
        """
        if not (self.mlflow_tracker and self.mlflow_tracker.config.enabled):
            return

        try:
            params_str = {k: str(v) for k, v in params.items()}
            self.mlflow_tracker.log_params(params_str)
            self.logger.debug(f"Parámetros loggeados a MLflow: {list(params_str.keys())}")
        except Exception as e:
            self.logger.error(f"Error loggeando parámetros a MLflow: {e}")

    def _log_mlflow_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Loggea métricas a MLflow. Reutiliza el run activo del entrenamiento o crea uno nuevo.

        Args:
            metrics: Diccionario con las métricas a loggear
        """
        if not (self.mlflow_tracker and self.mlflow_tracker.config.enabled):
            return

        try:
            # Verificar si hay un run activo del entrenamiento
            if self.mlflow_tracker._active_run is None:
                self.logger.warning("No hay run activo de entrenamiento, iniciando nuevo run para evaluación")
                #self.mlflow_tracker.start_run(run_name="evaluation_only")

            self.mlflow_tracker.log_metrics(metrics)
            self.logger.info(f"Métricas loggeadas a MLflow: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Error loggeando métricas a MLflow: {e}")

    def _log_mlflow_model(self, model: Pipeline, artifact_path: str, input_example: Any = None) -> None:
        """
        Loggea el modelo entrenado a MLflow.

        Args:
            model: Modelo entrenado (Pipeline)
            artifact_path: Ruta del artifact en MLflow
            input_example: Ejemplo de entrada para el modelo (opcional)
        """
        if not (self.mlflow_tracker and self.mlflow_tracker.config.enabled):
            return

        try:
            if input_example is not None:
                self.mlflow_tracker.log_model(model, artifact_path=artifact_path, input_example=input_example)
            else:
                self.mlflow_tracker.log_model(model, artifact_path=artifact_path)
            self.logger.debug(f"Modelo loggeado a MLflow en: {artifact_path}")
        except Exception as e:
            self.logger.error(f"Error loggeando modelo a MLflow: {e}")

    def _log_mlflow_text(self, text: str, artifact_file: str) -> None:
        """
        Loggea un texto como artifact a MLflow.

        Args:
            text: Contenido de texto a loggear
            artifact_file: Nombre del archivo donde guardar el texto
        """
        if not (self.mlflow_tracker and self.mlflow_tracker.config.enabled):
            return

        try:
            self.mlflow_tracker.log_text(text, artifact_file=artifact_file)
            self.logger.debug(f"Texto loggeado a MLflow: {artifact_file}")
        except Exception as e:
            self.logger.error(f"Error loggeando texto a MLflow: {e}")

    def _log_mlflow_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Loggea un archivo o directorio como artifact a MLflow.

        Args:
            local_path: Ruta local del archivo o directorio a subir
            artifact_path: Ruta donde guardar el artifact en MLflow (opcional)
        """
        if not (self.mlflow_tracker and self.mlflow_tracker.config.enabled):
            return

        try:
            self.mlflow_tracker.log_artifact(local_path, artifact_path=artifact_path)
            self.logger.debug(f"Artifact loggeado a MLflow: {local_path}")
        except Exception as e:
            self.logger.error(f"Error loggeando artifact a MLflow: {e}")

    def _ensure_model_trained(self) -> None:
        """Valida que el modelo haya sido entrenado antes de evaluar o predecir."""
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")

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
