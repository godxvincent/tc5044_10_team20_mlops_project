from dataclasses import dataclass
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from mlops.base.steps import ModelTrainerBase
from mlops.config import BaseDataClassModel, MLConfigLoader

PARAM_DIST_RF = {
    'classifier__n_estimators': randint(100, 1000),
    'classifier__max_depth': [None] + list(range(5, 20)),
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__bootstrap': [True, False],
    'classifier__class_weight': [None, 'balanced']
}

@dataclass
class ModelTrainerConfig(BaseDataClassModel):
    iterations:int

    def __init__(self, **kwargs):
        """
            Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)

class ModelTrainer(ModelTrainerBase):

    def __init__(self, pipeline, datasets):
        self.model_trainer_config = MLConfigLoader().getParameter("model_trainer", ModelTrainerConfig())
        super().__init__(pipeline, datasets)

    def _createModel(self):
        pass

    def train(self):
        self.optimize_model(PARAM_DIST_RF)

    def evaluate(self):
        pass

    def predict(self):
        pass

    def get_model_attributes(self):
        pass

    def optimize_model(self, param_distributions):
        """
        Ejecuta RandomizedSearchCV para optimizar hiperparámetros.
        """
        X_train=self.datasets["trainX"]
        Y_train=self.datasets["trainY"]
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=self.model_trainer_config.iterations,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
            )
        search.fit(X_train, Y_train)
        return search