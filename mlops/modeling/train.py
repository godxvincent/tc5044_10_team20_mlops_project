from mlops.base.steps import ModelTrainerBase


class ModelTrainer(ModelTrainerBase):

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

    def get_model_attributes(self):
        pass