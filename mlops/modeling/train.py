from dataclasses import dataclass

from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


# Agregar clase para preprocessing y build model_revisar si ya se tiene en otro archivo#


def build_preprocessing_pipeline(k_features=15, n_components=8):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
            ("pca", PCA(n_components=n_components)),
        ]
    )


@dataclass
class ModelTrainerConfig(BaseDataClassModel):
    iterations: int
    cv_folds: int
    scoring: str
    random_state: int
    verbose: int

    def __init__(self, **kwargs):
        """
        Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)


class ModelTrainer(ModelTrainerBase):

    def __init__(self, pipeline, datasets):
        mlconfigloader = MLConfigLoader()
        self.model_trainer_config = mlconfigloader.getParameter("model_trainer", ModelTrainerConfig())
        self.general_params = mlconfigloader.general_parameters
        super().__init__(pipeline, datasets)
        self.model = None

    def _createModel(self):
        classifier = RandomForestClassifier(random_state=self.general_params.random_state)
        return Pipeline([("preprocessing", self.pipeline), ("classifier", classifier)])

    def train(self):
        self._createModel()
        search = self.optimize_model(PARAM_DIST_RF)
        self.best_model = search.best_estimator_
        self.logger.info("\nMejores parámetros encontrados:")
        self.logger.info(search.best_params_)
        self.best_params_ = search.best_params_

    def evaluate(self):
        X_test = self.datasets["testX"]
        Y_test = self.datasets["testY"]
        Y_pred = self.model.predict(X_test)
        self.logger.info("\nEvaluación del modelo:")
        self.logger.info(classification_report(Y_test, Y_pred))

    def predict(self, X):
        return self.best_model.predict(X)

    def get_model_attributes(self):
        clf = self.best_model.named_steps["classifier"]
        return {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "feature_importances": clf.feature_importances_.tolist(),
            "best_params": self.best_params_,
        }

    def optimize_model(self, param_distributions):
        """
        Ejecuta RandomizedSearchCV para optimizar hiperparámetros.
        """
        X_train = self.datasets["trainX"]
        Y_train = self.datasets["trainY"]
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=self.model_trainer_config.iterations,
            cv=self.model_trainer_config.cv_folds,
            scoring=self.model_trainer_config.scoring,
            n_jobs=-1,
            random_state=self.model_trainer_config.random_state,
            verbose=self.model_trainer_config.verbose,
        )
        search.fit(X_train, Y_train)
        return search


class ModelTrainerGB(ModelTrainerBase):
    def __init__(self, pipeline, datasets):
        # Cargar configuración específica para Gradient Boosting
        mlconfigloader = MLConfigLoader()
        self.model_trainer_config = mlconfigloader.getParameter("model_trainer", ModelTrainerConfig())
        self.general_params = mlconfigloader.general_parameters
        super().__init__(pipeline, datasets)
        self.model = None

    def _createModel(self):
        classifier = GradientBoostingClassifier(random_state=self.general_params.random_state)
        return Pipeline([("preprocessing", self.pipeline), ("classifier", classifier)])

    def train(self):
        self._createModel()
        search = self.optimize_model(PARAM_DIST_GB)
        self.model = search.best_estimator_
        self.logger.info("\nMejores parámetros Gradient Boosting:")
        self.logger.info(search.best_params_)
        self.best_params_ = search.best_params_

    def evaluate(self):
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        X_test = self.datasets["testX"]
        Y_test = self.datasets["testY"]
        Y_pred = self.model.predict(X_test)
        self.logger.info("\nEvaluación Gradient Boosting:")
        self.logger.info(classification_report(Y_test, Y_pred))

    def predict(self, X):
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        return self.model.predict(X)

    def get_model_attributes(self):
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        clf = self.model.named_steps["classifier"]
        return {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "learning_rate": clf.learning_rate,
            "subsample": clf.subsample,
            "best_params": self.best_params_,
        }

    def optimize_model(self, param_distributions):
        X_train = self.datasets["trainX"]
        Y_train = self.datasets["trainY"]
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=self.model_trainer_config.iterations,
            cv=self.model_trainer_config.cv_folds,
            scoring=self.model_trainer_config.scoring,
            n_jobs=-1,
            random_state=self.model_trainer_config.random_state,
            verbose=self.model_trainer_config.verbose,
        )
        search.fit(X_train, Y_train)
        return search
