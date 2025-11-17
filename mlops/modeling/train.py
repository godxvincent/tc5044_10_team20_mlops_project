from dataclasses import dataclass
from typing import Any, Dict

from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from mlops.base.steps import ModelTrainerBase
from mlops.config import BaseDataClassModel, MLConfigLoader

PARAM_DIST_RF = {
    "classifier__n_estimators": randint(100, 1000),
    "classifier__max_depth": [None] + list(range(5, 20)),
    "classifier__min_samples_split": randint(2, 20),
    "classifier__min_samples_leaf": randint(1, 10),
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__bootstrap": [True, False],
    "classifier__class_weight": [None, "balanced"],
}

PARAM_DIST_GB = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__max_depth": randint(3, 10),
    "classifier__subsample": uniform(0.7, 0.3),
}


@dataclass
class ModelTrainerConfig(BaseDataClassModel):
    iterations: int
    cv_folds: int
    scoring: str
    random_state: int
    verbose: int

    def __init__(self, **kwargs):
        """Clase modelo de datos para la configuración del model trainer."""
        super().__init__(**kwargs)


class ModelTrainer(ModelTrainerBase):

    def __init__(self, pipeline, datasets):
        mlconfigloader = MLConfigLoader()
        self.model_trainer_config = mlconfigloader.getParameter("model_trainer", ModelTrainerConfig())
        self.general_params = mlconfigloader.general_parameters
        super().__init__(pipeline, datasets)
        self._initialize_mlflow_tracker(model_name="RandomForest")

    def _createModel(self):
        classifier = RandomForestClassifier(random_state=self.general_params.random_state)
        return Pipeline([("preprocessing", self.pipeline), ("classifier", classifier)])

    def train(self):
        self._start_mlflow_run()

        try:
            self.pipeline = self._createModel()
            search = self._optimize_model(PARAM_DIST_RF)
            self.best_model = search.best_estimator_
            self.best_params_ = search.best_params_

            self.logger.info("\nMejores parámetros encontrados:")
            self.logger.info(self.best_params_)

            # Loggear parámetros y modelo a MLflow
            params = self._build_training_params()
            self._log_mlflow_params(params)
            self._log_mlflow_model(
                self.best_model, artifact_path="random_forest_model", input_example=self.datasets["trainX"]
            )
            # Registrar modelo en Model Registry
            if self.mlflow_tracker and self.mlflow_tracker._active_run:
                self.mlflow_tracker.register_model("turkish_music_emotion_rf", "random_forest_model")
        except Exception as e:
            self._end_mlflow_run()
            raise e

    def _build_training_params(self) -> Dict[str, Any]:
        """Construye diccionario de parámetros para logging a MLflow."""
        params = {f"best_{k}": v for k, v in self.best_params_.items()}
        params.update(
            {
                "random_state": self.general_params.random_state,
                "cv_folds": self.model_trainer_config.cv_folds,
                "scoring": self.model_trainer_config.scoring,
                "iterations": self.model_trainer_config.iterations,
                "model_type": "RandomForest",
            }
        )
        return params

    def evaluate(self):
        self._ensure_model_trained()

        X_test = self.datasets["testX"]
        Y_test = self.datasets["testY"]
        Y_pred = self.best_model.predict(X_test)

        # Generar classification report
        report = classification_report(Y_test, Y_pred)
        self.logger.info("\nEvaluación del modelo:")
        self.logger.info(report)

        # Calcular y loggear métricas a MLflow
        self.__metrics = self._calculate_metrics(Y_test, Y_pred)
        self._log_mlflow_metrics(self.__metrics)

        # Loggear classification report como artifact
        self._log_mlflow_text(report, artifact_file="classification_report.txt")

        # Cerrar el run después de la evaluación
        self._end_mlflow_run()

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calcula métricas de evaluación del modelo."""
        return {
            "test_accuracy": accuracy_score(y_true, y_pred),
            "test_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "test_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "test_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def predict(self, X):
        self._ensure_model_trained()
        return self.best_model.predict(X)

    def get_model_attributes(self):
        self._ensure_model_trained()
        clf = self.best_model.named_steps["classifier"]
        return {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "feature_importances": clf.feature_importances_.tolist(),
            "best_params": self.best_params_,
        }

    def _optimize_model(self, param_distributions):
        """Ejecuta RandomizedSearchCV para optimizar hiperparámetros."""
        X_train = self.datasets["trainX"]
        Y_train = self.datasets["trainY"]
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=self.model_trainer_config.iterations,
            cv=self.model_trainer_config.cv_folds,
            scoring=self.model_trainer_config.scoring,
            n_jobs=-1,
            random_state=self.general_params.random_state,
            verbose=self.model_trainer_config.verbose,
        )
        search.fit(X_train, Y_train)
        return search

    def get_performance_metrics(self):
        if self.__metrics:
            return self.__metrics
        else:
            raise Exception("la función evaluate() debe ser llamada primero")


class ModelTrainerGB(ModelTrainerBase):
    def __init__(self, pipeline, datasets):
        mlconfigloader = MLConfigLoader()
        self.model_trainer_config = mlconfigloader.getParameter("model_trainer", ModelTrainerConfig())
        self.general_params = mlconfigloader.general_parameters
        super().__init__(pipeline, datasets)
        self._initialize_mlflow_tracker(model_name="GradientBoosting")

    def _createModel(self):
        classifier = GradientBoostingClassifier(random_state=self.general_params.random_state)
        return Pipeline([("preprocessing", self.pipeline), ("classifier", classifier)])

    def train(self):
        self._start_mlflow_run()

        try:
            self.pipeline = self._createModel()
            search = self._optimize_model(PARAM_DIST_GB)
            self.best_model = search.best_estimator_
            self.best_params_ = search.best_params_

            self.logger.info("\nMejores parámetros Gradient Boosting:")
            self.logger.info(self.best_params_)

            # Loggear parámetros y modelo a MLflow
            params = self._build_training_params()
            self._log_mlflow_params(params)
            self._log_mlflow_model(
                self.best_model, artifact_path="gradient_boosting_model", input_example=self.datasets["trainX"]
            )
            # Registrar modelo en Model Registry
            if self.mlflow_tracker and self.mlflow_tracker._active_run:
                self.mlflow_tracker.register_model("turkish_music_emotion_gb", "gradient_boosting_model")
        except Exception as e:
            self._end_mlflow_run()
            raise e

    def _build_training_params(self) -> Dict[str, Any]:
        """Construye diccionario de parámetros para logging a MLflow."""
        params = {f"best_{k}": v for k, v in self.best_params_.items()}
        params.update(
            {
                "random_state": self.general_params.random_state,
                "cv_folds": self.model_trainer_config.cv_folds,
                "scoring": self.model_trainer_config.scoring,
                "iterations": self.model_trainer_config.iterations,
                "model_type": "GradientBoosting",
            }
        )
        return params

    def evaluate(self):
        self._ensure_model_trained()

        X_test = self.datasets["testX"]
        Y_test = self.datasets["testY"]
        Y_pred = self.best_model.predict(X_test)

        # Generar classification report
        report = classification_report(Y_test, Y_pred)
        self.logger.info("\nEvaluación Gradient Boosting:")
        self.logger.info(report)

        # Calcular y loggear métricas a MLflow
        metrics = self._calculate_metrics(Y_test, Y_pred)
        self._log_mlflow_metrics(metrics)

        # Loggear classification report como artifact
        self._log_mlflow_text(report, artifact_file="classification_report.txt")

        # Cerrar el run después de la evaluación
        self._end_mlflow_run()

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calcula métricas de evaluación del modelo."""
        return {
            "test_accuracy": accuracy_score(y_true, y_pred),
            "test_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "test_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "test_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def predict(self, X):
        self._ensure_model_trained()
        return self.best_model.predict(X)

    def get_model_attributes(self):
        self._ensure_model_trained()
        clf = self.best_model.named_steps["classifier"]
        return {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "learning_rate": clf.learning_rate,
            "subsample": clf.subsample,
            "best_params": self.best_params_,
        }

    def _optimize_model(self, param_distributions):
        """Ejecuta RandomizedSearchCV para optimizar hiperparámetros."""
        X_train = self.datasets["trainX"]
        Y_train = self.datasets["trainY"]
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=self.model_trainer_config.iterations,
            cv=self.model_trainer_config.cv_folds,
            scoring=self.model_trainer_config.scoring,
            n_jobs=-1,
            random_state=self.general_params.random_state,
            verbose=self.model_trainer_config.verbose,
        )
        search.fit(X_train, Y_train)
        return search
