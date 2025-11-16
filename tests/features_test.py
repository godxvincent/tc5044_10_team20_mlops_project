from sklearn.compose import ColumnTransformer

from mlops.modeling.features import FeatureEngineProcessor


def test_feature_engine_processor_initialization():
    processor = FeatureEngineProcessor()
    assert processor.config is not None
    assert isinstance(processor.numerical_features, list)
    assert len(processor.numerical_features) > 0


def test_create_imputer_returns_column_transformer():
    processor = FeatureEngineProcessor()
    imputer = processor._createImputer()
    assert isinstance(imputer, ColumnTransformer)
    assert hasattr(imputer, "transform")


def test_create_standardizer_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createStandardizer() is None


def test_create_scaler_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createScaler() is None


def test_create_outlier_processor_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createOutlierProcessor() is None


def test_create_feature_reducer_not_implemented():
    processor = FeatureEngineProcessor()
    assert processor._createFeatureReducer() is None
