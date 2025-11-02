from mlops.base.steps import DataLoaderBase, FeatureEngineProcessorBase, MLPipelineBase, ModelTrainerBase


class MLPipelinePrueba(MLPipelineBase):

    def __init__(self):
        super().__init__()

    def load_data_step(self, file_name: str):
        pass

    def clean_up_data_step(self):
        pass

    def train_step(self):
        pass

    def evaluate_step(self):
        pass

    def feature_engineering_step(self):
        pass


class DataLoaderPrueba(DataLoaderBase):

    def __init__(self):
        super().__init__()

    def load_file(self, file_name: str):
        pass

    def get_shape(self):
        pass

    def get_statistics(self):
        pass

    def get_train_test_dataset(self):
        pass


class FeatureEngineProcessorPrueba(FeatureEngineProcessorBase):

    def __init__(self):
        super().__init__()

    def _createImputer(self):
        pass

    def _createStandardizer(self):
        pass

    def _createScaler(self):
        pass

    def _createOutlierProcessor(self):
        pass

    def _createFeatureReducer(self):
        pass


class ModelTrainerPrueba(ModelTrainerBase):
    def __init__(self, pipeline, datasets):
        super().__init__(pipeline, datasets)

    def _createModel(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def get_model_attributes(self, **kwargs):
        pass


def test_model_trainer_class():
    from pandas import DataFrame
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("test", "test")])
    datasets = {
        "trainX": DataFrame(),
        "trainY": DataFrame(),
        "testX": DataFrame(),
        "testY": DataFrame(),
    }
    test_class = ModelTrainerPrueba(pipeline, datasets)
    assert test_class is not None


def test_data_loader_class():
    test_class = DataLoaderPrueba()
    assert test_class is not None


def test_mlpipeline_class():
    test_class = MLPipelinePrueba()
    assert test_class is not None


def test_feature_engine_class():
    test_class = FeatureEngineProcessorPrueba()
    assert test_class is not None
    assert test_class.createPipeline() is not None
