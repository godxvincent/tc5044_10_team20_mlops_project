import os
import pickle

import mlflow
import mlflow.sklearn

# ----------------------------
# Configuración MLflow
# ----------------------------
mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "turkish_music_emotion_rf"
MODEL_PATH = "models/model.pkl"  # cámbialo si tu modelo se llama diferente

# ----------------------------
# Cargar modelo local
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Registrar modelo en MLflow
# ----------------------------
with mlflow.start_run(run_name="register_model_run"):
    print("📌 Logueando modelo en MLflow...")

    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name=MODEL_NAME)

    print(f"✅ Modelo registrado como: {MODEL_NAME}")
    print("📌 Revisa en: http://localhost:5000/#/models")
