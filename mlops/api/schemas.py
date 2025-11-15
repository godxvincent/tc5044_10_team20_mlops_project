from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Payload de entrada para /predict.

    `features` es un diccionario donde la llave es el nombre de la columna
    y el valor es numérico (float/int). Debe coincidir con las columnas
    que espera el pipeline entrenado.
    """

    features: Dict[str, float] = Field(
        ...,
        description="Diccionario de features de entrada, e.g. {'feature1': 0.1, 'feature2': 3.4}",
    )


class PredictionResponse(BaseModel):
    """
    Respuesta del endpoint /predict.
    """

    prediction: str = Field(..., description="Clase/emoción predicha por el modelo")
    model_uri: str = Field(..., description="Ruta del modelo en MLflow (models:/name/version)")
    detail: Optional[str] = Field(
        default=None,
        description="Información adicional (opcional)",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado de la API")
    model_loaded: bool = Field(..., description="Indica si el modelo se cargó correctamente")
    model_uri: str = Field(..., description="Ruta del modelo configurado")
