import os

import pytest
from sklearn.impute import SimpleImputer

from mlops.features import FeatureEngineProcessor


@pytest.fixture(autouse=True)
def setup_data():
    # Code to run before each test
    print("\nSetting up before test")
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    yield
    # Code to run after each test (teardown)
    os.environ.pop("ENV_FOR_DYNACONF", None)
    print("\nTearing down after test")


def test_feature_engine_processor_initialization():
    processor = FeatureEngineProcessor()
    assert processor.config is not None
    assert isinstance(processor.numerical_features, list)
    assert len(processor.numerical_features) > 0


def test_create_imputer_returns_column_transformer():
    processor = FeatureEngineProcessor()
    imputer_tuple = processor._createImputer()
    simple_imputer = imputer_tuple[1]
    assert isinstance(imputer_tuple, tuple)
    assert isinstance(simple_imputer, SimpleImputer)
    assert hasattr(simple_imputer, "transform")


def test_create_standardizer_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createStandardizer() is None


def test_create_scaler_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createScaler() is not None


def test_create_outlier_processor_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createOutlierProcessor() is None


def test_create_feature_reducer_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createFeatureReducer() is not None
