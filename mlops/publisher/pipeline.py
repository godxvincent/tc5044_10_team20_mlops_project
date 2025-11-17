import os
import pathlib
import sys

from mlops.pipeline import MLPipeline

if __name__ == "__main__":

    sys.path.append(str(pathlib.Path(__name__).parent.parent.absolute().parent))
    dynaconf_env = os.getenv("ENV_FOR_DYNACONF", None)
    if not dynaconf_env:
        os.environ["ENV_FOR_DYNACONF"] = "local"

    ml_pipeline = MLPipeline()
    ml_pipeline.load_data_step("turkish_music_emotion_modified.csv")
    ml_pipeline.clean_up_data_step()
    ml_pipeline.feature_engineering_step()
    ml_pipeline.train_step()
    ml_pipeline.evaluate_step()
