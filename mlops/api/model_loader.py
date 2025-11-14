import os
from functools import lru_cache

import mlflow
import mlflow.pyfunc
import pandas as pd

# Usa el MLflow tracking server que ya tienes en Docker
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Ruta del modelo en el registro de MLflow (ajústala a tu modelo real)
# Ejemplo: models:/turkish_music_emotion_rf/1
MODEL_URI = os.getenv("MODEL_URI", "models:/turkish_music_emotion_rf/1")


@lru_cache(maxsize=1)
def get_model() -> mlflow.pyfunc.PyFuncModel:  # type: ignore
    """
    Carga el modelo registrado en MLflow usando el MODEL_URI.
    Se cachea con lru_cache para no cargarlo en cada petición.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return model


def predict_from_dict(features: dict) -> str:
    """
    Convierte el diccionario de features a DataFrame y obtiene la predicción.
    """
    df = pd.DataFrame([features])
    model = get_model()
    preds = model.predict(df)
    # Asumimos que regresa algo tipo array con 1 valor
    return str(preds[0])
