import logging

from fastapi import FastAPI, HTTPException

from mlops.api.model_loader import MODEL_URI, get_model, predict_from_dict
from mlops.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Turkish Music Emotion Serving API - Team 20",
    version="0.1.0",
    description=("API para servir el modelo de clasificación de emociones de música turca " "registrado en MLflow."),
)


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """
    Intenta cargar el modelo al iniciar la app.
    Si falla, la API seguirá levantando, pero /health lo reportará.
    """
    try:
        get_model()
        logger.info("Modelo cargado correctamente desde %s", MODEL_URI)
    except Exception as exc:  # noqa: BLE001
        logger.error("No se pudo cargar el modelo en startup: %s", exc)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def healthcheck() -> HealthResponse:
    """
    Verifica el estado de la API y del modelo.
    """
    try:
        get_model()
        return HealthResponse(status="ok", model_loaded=True, model_uri=MODEL_URI)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False, model_uri=MODEL_URI)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Endpoint principal de inferencia.

    Recibe un diccionario de features y devuelve la predicción del modelo.
    """
    try:
        pred = predict_from_dict(payload.features)
        return PredictionResponse(
            prediction=pred,
            model_uri=MODEL_URI,
            detail="Predicción generada correctamente.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error durante la predicción: %s", exc)
        raise HTTPException(status_code=500, detail="Error al generar la predicción") from exc
