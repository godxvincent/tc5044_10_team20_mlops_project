from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Self, Tuple

import numpy as np
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

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
        # TODO: Definir funcion comun para retornar la misma estructura
        # en el DataLoader como en el DataCleaner dado un dataset.
        pass

    @abstractmethod
    def get_train_test_dataset(self) -> Dict[str, DataFrame]:
        pass


class FeatureEngineProcessorBase(ABC, BaseLogger):
    """
    Esta clase define los metodos minimos que deberia tener cualquier clase
    encargada de hacer limpieza de datos, así como ingenieria de caracteristicas.
    Esta clase al final deberia retornar un Pipeline con los pasos.
    Por default se espera que esta clase tenga como minimo el dataset al cual
    se le aplicará la limpieza de datos.
    """

    _PCAfeaturesScaled = False

    def __init__(self):
        self._PCAfeaturesScaled = False
        super().__init__(self.__class__.__name__)

    def createPipeline(self) -> Pipeline:

        transformers = []
        imputer = self._createImputer()
        if imputer:
            transformers.append(imputer)

        standardizer = self._createStandardizer()
        if standardizer:
            transformers.append(standardizer)

        outlierProcessor = self._createOutlierProcessor()
        if outlierProcessor:
            transformers.append(outlierProcessor)

        scaler = self._createScaler()
        if scaler:
            transformers.append(scaler)

        featureReducer = self._createFeatureReducer()
        if featureReducer and self._PCAfeaturesScaled:
            transformers.append(featureReducer)

        return make_pipeline(ColumnTransformer(transformers))

    @abstractmethod
    def _createImputer(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de imputar los valores faltantes.
        """
        pass

    @abstractmethod
    def _createStandardizer(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de estandarizar los valores como la clase.
        """
        pass

    @abstractmethod
    def _createScaler(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de escalar los valores de un conjunto de features.
        """
        pass

    @abstractmethod
    def _createOutlierProcessor(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de lidiar con los valores atipicos.
        """
        pass

    @abstractmethod
    def _createFeatureReducer(self) -> Optional[Tuple]:
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
        self._training_run_id = None  # Guardar run_id del entrenamiento para nested runs
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
            # Guardar run_id del entrenamiento si no está guardado
            if self._training_run_id is None and self.mlflow_tracker._active_run:
                self._training_run_id = self.mlflow_tracker._active_run.info.run_id
                self.logger.debug(f"Run ID de entrenamiento guardado: {self._training_run_id}")

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
                # self.mlflow_tracker.start_run(run_name="evaluation_only")

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

    def evaluate_drift(self, drift_scenarios: list[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """
        Evalúa data drift en diferentes escenarios.

        Args:
            drift_scenarios: Lista de escenarios de drift a evaluar.
                           Si es None, usa escenario por defecto (covariate_shift leve).

        Returns:
            Diccionario con resultados de todas las evaluaciones
        """
        self._ensure_model_trained()

        # Obtener métricas de baseline del test set
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        X_test = self.datasets["testX"]
        Y_test = self.datasets["testY"]
        Y_pred = self.best_model.predict(X_test)

        baseline_metrics = {
            "accuracy": float(accuracy_score(Y_test, Y_pred)),
            "precision": float(precision_score(Y_test, Y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(Y_test, Y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(Y_test, Y_pred, average="weighted", zero_division=0)),
        }

        self.logger.info("Métricas de baseline calculadas")
        self.logger.info(f"Baseline - Accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1']:.4f}")

        # Si no se proporcionan escenarios, usar el escenario por defecto
        if drift_scenarios is None:
            drift_scenarios = [
                {
                    "name": "covariate_shift_mild",
                    "type": "covariate_shift",
                    "params": {
                        "features": ["_Tempo_Mean", "_RMSenergy_Mean"],
                        "mean_shift": 0.1,
                        "variance_factor": 1.05,
                    },
                }
            ]

        from mlops.monitoring.drift_evaluator import DriftEvaluator

        evaluator = DriftEvaluator(
            model=self.best_model,
            reference_data=X_test,
            reference_target=Y_test,
            baseline_metrics=baseline_metrics,
        )

        # Evaluar cada escenario (cada uno en su propio nested run)
        all_results = {}
        for scenario in drift_scenarios:
            scenario_name = scenario.get("name", "unknown")
            drift_type = scenario.get("type")
            drift_params = scenario.get("params", {})

            self.logger.info(f"Evaluando escenario: {scenario_name}")

            # Crear nested run como child del run de entrenamiento
            # Asumimos que siempre hay un run de entrenamiento (_training_run_id)
            if self._training_run_id and self.mlflow_tracker:
                import mlflow

                run_name = f"drift_detection_{scenario_name}"
                self.mlflow_tracker.start_nested_run(parent_run_id=self._training_run_id, run_name=run_name)

                # Añadir tags al nested run
                mlflow.set_tag("run_type", "drift_detection")
                mlflow.set_tag("scenario_name", scenario_name)
                mlflow.set_tag("drift_type", drift_type)

            try:
                result = evaluator.evaluate_drift_scenario(drift_type, drift_params)
                all_results[scenario_name] = result

                # Loggear a MLflow (en el nested run del escenario)
                if self.mlflow_tracker and self.mlflow_tracker._active_run:
                    self._log_drift_results_to_mlflow(scenario_name, result)

            except Exception as e:
                self.logger.error(f"Error evaluando escenario {scenario_name}: {e}")
                all_results[scenario_name] = {"error": str(e)}
            finally:
                # Cerrar el nested run del escenario
                if self.mlflow_tracker and self.mlflow_tracker._active_run:
                    self._end_mlflow_run()

        return all_results

    def _log_drift_results_to_mlflow(self, scenario_name: str, result: Dict[str, Any]) -> None:
        """
        Loggea resultados de drift a MLflow.

        Args:
            scenario_name: Nombre del escenario
            result: Resultados de la evaluación de drift
        """
        if not (self.mlflow_tracker and self.mlflow_tracker.config.enabled):
            self.logger.debug("MLflow deshabilitado, omitiendo logging de drift")
            return

        try:
            # Loggear parámetros del drift
            drift_params = result.get("drift_params", {})
            mlflow_params = {
                "drift_scenario": scenario_name,
                "drift_type": result.get("drift_type", "unknown"),
            }
            for key, value in drift_params.items():
                mlflow_params[f"drift_{key}"] = str(value) if isinstance(value, list) else value

            self._log_mlflow_params(mlflow_params)

            # Loggear métricas de drift (usando KS de alibi-detect)
            drift_metrics = result.get("drift_metrics", {})
            if drift_metrics:
                # Extraer p_values y distances de KS
                p_values = []
                distances = []
                drift_detected_count = 0

                # Excluir '_all_features' del conteo individual
                for key, v in drift_metrics.items():
                    if key == "_all_features":
                        continue
                    p_val = v.get("p_value", np.nan)
                    dist = v.get("distance", np.nan)
                    drift_detected = v.get("drift_detected", False)

                    if isinstance(p_val, (int, float)) and not np.isnan(p_val):
                        p_values.append(float(p_val))
                    if isinstance(dist, (int, float)) and not np.isnan(dist):
                        distances.append(float(dist))
                    if drift_detected:
                        drift_detected_count += 1

                mlflow_drift_metrics = {}

                # Loggear p-values
                if p_values:
                    mlflow_drift_metrics.update(
                        {
                            "drift_p_value_mean": float(np.mean(p_values)),
                            "drift_p_value_min": float(np.min(p_values)),
                        }
                    )

                # Loggear distances (KS)
                if distances:
                    mlflow_drift_metrics.update(
                        {
                            "drift_ks_distance_mean": float(np.mean(distances)),
                            "drift_ks_distance_max": float(np.max(distances)),
                        }
                    )

                # Loggear información de drift detection
                mlflow_drift_metrics["drift_features_with_drift"] = drift_detected_count
                mlflow_drift_metrics["drift_detected"] = drift_detected_count > 0
                mlflow_drift_metrics["drift_method"] = "ks"

                if mlflow_drift_metrics:
                    self._log_mlflow_metrics(mlflow_drift_metrics)

            # Loggear métricas de performance
            performance_metrics = result.get("performance_metrics", {})
            if performance_metrics:
                mlflow_perf_metrics = {f"drift_{k}": v for k, v in performance_metrics.items()}
                self._log_mlflow_metrics(mlflow_perf_metrics)

            # Loggear comparación con baseline
            comparison = result.get("comparison", {})
            if comparison:
                mlflow_comparison = {}
                for metric_name, comp_data in comparison.items():
                    mlflow_comparison[f"drift_{metric_name}_drop"] = comp_data.get("drop", 0.0)
                    mlflow_comparison[f"drift_{metric_name}_drop_percent"] = comp_data.get("drop_percent", 0.0)

                self._log_mlflow_metrics(mlflow_comparison)

            self.logger.info(f"Resultados de drift loggeados a MLflow para escenario: {scenario_name}")

        except Exception as e:
            self.logger.error(f"Error loggeando resultados de drift a MLflow: {e}")

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
        La implementación concreta de esta función debe ejecutar el modelo
        contra un conjunto de pruebas y devolver la predicción.
        """
        pass

    @abstractmethod
    def get_model_attributes(self, **kwargs):
        """
        No estoy 100% seguro aun como deberia funcionar esta función.
        """
        pass
