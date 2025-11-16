import os

import pytest
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from mlops.modeling.train import ModelTrainer, ModelTrainerGB


@pytest.fixture(autouse=True)
def setup_data():
    # Code to run before each test
    print("\nSetting up before test")
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    yield
    # Code to run after each test (teardown)
    os.environ.pop("ENV_FOR_DYNACONF", None)
    print("\nTearing down after test")


@pytest.fixture
def dummy_dataset():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return {"trainX": X, "trainY": y, "testX": X, "testY": y}


@pytest.fixture
def preprocessing_pipeline():
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


def test_model_trainer_predict(dummy_dataset, preprocessing_pipeline):
    trainer = ModelTrainer(preprocessing_pipeline, dummy_dataset)
    trainer.train()
    preds = trainer.predict(dummy_dataset["testX"])
    assert len(preds) == len(dummy_dataset["testY"])
    assert all(p in [0, 1] for p in preds)


def test_model_trainer_gb_predict(dummy_dataset, preprocessing_pipeline):
    trainer = ModelTrainerGB(preprocessing_pipeline, dummy_dataset)
    trainer.train()
    preds = trainer.predict(dummy_dataset["testX"])
    assert len(preds) == len(dummy_dataset["testY"])
    assert all(p in [0, 1] for p in preds)
