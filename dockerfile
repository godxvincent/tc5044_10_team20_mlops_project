# -----------------------------
# 1. Imagen base
# -----------------------------
FROM python:3.11-slim

# Evitar buffering
ENV PYTHONUNBUFFERED=1

# -----------------------------
# 2. Crear carpeta del proyecto
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Copiar dependencias
# -----------------------------
COPY requirements.txt /app/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 4. Copiar el código del proyecto
# -----------------------------
COPY mlops /app/mlops

# Configurar variable para MLflow (puedes sobrescribir en docker run)
ENV MODEL_URI="models:/turkish_music_emotion_rf/1"

# -----------------------------
# 5. Exponer puerto para FastAPI
# -----------------------------
EXPOSE 8000

# -----------------------------
# 6. Comando para ejecutar FastAPI
# -----------------------------
CMD ["uvicorn", "mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
