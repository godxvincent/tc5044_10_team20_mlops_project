"""
Módulo simple para tracking de experimentos con MLflow.
Implementación sencilla que permite loggear parámetros, métricas y modelos.
"""

import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

from mlops.base.logger import BaseLogger
from mlops.config import BaseDataClassModel, MLConfigLoader


@dataclass
class MLflowConfig(BaseDataClassModel):
    """Configuración de MLflow"""
    server: str
    port: str
    experiment_name: str
    enabled: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def tracking_uri(self) -> str:
        """Retorna la URI completa del tracking server"""
        return f"http://{self.server}:{self.port}"


class MLflowTracker(BaseLogger):
    """
    Tracker simple para MLflow.
    Permite loggear parámetros, métricas y modelos.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.config = None
        self._active_run = None
        self._experiment_id = None
        
        # Cargar configuración
        try:
            config_loader = MLConfigLoader()
            self.config = config_loader.getParameter("mlflowconfig", MLflowConfig())
            self.logger.info(f"MLflow config cargada: enabled={self.config.enabled}, server={self.config.server}:{self.config.port}")
        except Exception as e:
            self.logger.warning(f"Error cargando configuración MLflow: {e}. MLflow deshabilitado.")
            self.config = MLflowConfig(
                server="localhost",
                port="5000",
                experiment_name="Default Experiment",
                enabled=False
            )

    def initialize(self, model_name: str) -> None:
        """
        Inicializa conexión con MLflow y configura experimento.
        El nombre del experimento incluye el nombre del modelo.
        
        Args:
            model_name: Nombre del modelo (ej: "RandomForest", "GradientBoosting")
        """
        if not self.config.enabled:
            self.logger.debug("MLflow está deshabilitado, omitiendo inicialización")
            return

        try:
            # Configurar tracking URI
            tracking_uri = self.config.tracking_uri
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"MLflow tracking URI: {tracking_uri}")

            # Crear nombre de experimento que incluya el nombre del modelo
            experiment_name = f"{self.config.experiment_name} - {model_name}"
            
            # Crear o obtener experimento
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment:
                self._experiment_id = experiment.experiment_id
                self.logger.info(f"Experimento MLflow: {experiment_name} (ID: {self._experiment_id})")
            else:
                self.logger.warning(f"Experimento {experiment_name} no encontrado")

            # Habilitar autologging de scikit-learn
            #mlflow.sklearn.autolog()
            #self.logger.info("MLflow sklearn autologging habilitado")

        except Exception as e:
            self.logger.error(f"Error inicializando MLflow: {e}")
            self.config.enabled = False
            raise e

    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Inicia un nuevo run de MLflow.
        El nombre del run incluye un timestamp como prefijo para ordenar.
        
        Args:
            run_name: Nombre del run. Si es None, se genera automáticamente con timestamp.
        """
        if not self.config.enabled:
            self.logger.debug("MLflow está deshabilitado, omitiendo start_run")
            return

        try:
            # Generar nombre de run con timestamp como prefijo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_name is None:
                run_name = f"{timestamp}_run"
            else:
                run_name = f"{timestamp}_{run_name}"

            # Iniciar run
            self._active_run = mlflow.start_run(run_name=run_name, experiment_id=self._experiment_id)
            self.logger.info(f"MLflow run iniciado: {run_name} (ID: {self._active_run.info.run_id})")

        except Exception as e:
            self.logger.error(f"Error iniciando run de MLflow: {e}")
            raise e

    def end_run(self) -> None:
        """Finaliza el run actual de MLflow"""
        if not self.config.enabled or not self._active_run:
            return

        try:
            mlflow.end_run()
            self._active_run = None
            self.logger.info("MLflow run finalizado")
        except Exception as e:
            self.logger.error(f"Error finalizando run de MLflow: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Loggea parámetros a MLflow.
        Todos los valores se convierten a string.
        
        Args:
            params: Diccionario de parámetros a loggear
        """
        if not self.config.enabled or not self._active_run:
            return

        try:
            # Convertir todos los valores a string (MLflow requiere strings)
            params_str = {k: str(v) for k, v in params.items()}
            mlflow.log_params(params_str)
            self.logger.debug(f"Parámetros loggeados: {list(params_str.keys())}")
        except Exception as e:
            self.logger.error(f"Error loggeando parámetros: {e}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Loggea métricas a MLflow.
        
        Args:
            metrics: Diccionario de métricas a loggear (valores numéricos)
        """
        if not self.config.enabled or not self._active_run:
            return

        try:
            mlflow.log_metrics(metrics)
            self.logger.debug(f"Métricas loggeadas: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Error loggeando métricas: {e}")

    def log_model(self, model: Any, artifact_path: str = "model", input_example: Any = None) -> None:
        """
        Loggea un modelo a MLflow.
        
        Args:
            model: Modelo entrenado (debe ser compatible con mlflow.sklearn)
            artifact_path: Ruta donde guardar el modelo
            input_example: Ejemplo de entrada para el modelo (opcional)
        """
        if not self.config.enabled or not self._active_run:
            return

        try:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path, input_example=input_example)
            self.logger.info(f"Modelo loggeado en: {artifact_path}")
        except Exception as e:
            self.logger.error(f"Error loggeando modelo: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Loggea un archivo o directorio como artifact a MLflow.
        
        Args:
            local_path: Ruta local del archivo o directorio a subir
            artifact_path: Ruta donde guardar el artifact en MLflow (opcional, por defecto usa el nombre del archivo)
        """
        if not self.config.enabled or not self._active_run:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            artifact_display = artifact_path if artifact_path else local_path
            self.logger.info(f"Artifact loggeado: {artifact_display}")
        except Exception as e:
            self.logger.error(f"Error loggeando artifact: {e}")

    def log_text(self, text: str, artifact_file: str) -> None:
        """
        Loggea un texto como artifact a MLflow.
        
        Args:
            text: Contenido de texto a loggear
            artifact_file: Nombre del archivo donde guardar el texto (ej: "classification_report.txt")
        """
        if not self.config.enabled or not self._active_run:
            return

        try:
            mlflow.log_text(text, artifact_file=artifact_file)
            self.logger.info(f"Texto loggeado como artifact: {artifact_file}")
        except Exception as e:
            self.logger.error(f"Error loggeando texto: {e}")

