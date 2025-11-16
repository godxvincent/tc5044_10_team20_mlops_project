import pytest
from sklearn.datasets import make_classification

from mlops.modeling.pipeline import build_preprocessing_pipeline
from mlops.modeling.train import ModelTrainer, ModelTrainerGB


@pytest.fixture
def dummy_dataset():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return {"trainX": X, "trainY": y, "testX": X, "testY": y}


@pytest.fixture
def preprocessing_pipeline():
    return build_preprocessing_pipeline(k_features=10, n_components=5)


def test_model_trainer_training(dummy_dataset, preprocessing_pipeline):
    trainer = ModelTrainer(preprocessing_pipeline, dummy_dataset)
    trainer.train()
    assert trainer.best_model is not None
    assert hasattr(trainer.best_model, "predict")
    assert isinstance(trainer.get_model_attributes(), dict)


def test_model_trainer_gb_training(dummy_dataset, preprocessing_pipeline):
    trainer = ModelTrainerGB(preprocessing_pipeline, dummy_dataset)
    trainer.train()
    assert trainer.model is not None
    assert hasattr(trainer.model, "predict")
    attrs = trainer.get_model_attributes()
    assert "learning_rate" in attrs
    assert "subsample" in attrs
