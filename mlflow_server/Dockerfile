FROM python:3.9-slim

# Dependencias necesarias para MLflow + PostgreSQL + MinIO
RUN apt update && \
    apt install -y gcc libpq-dev && \
    pip install \
    mlflow \
    boto3 \
    minio \
    pymysql \
    cryptography \
    psycopg2-binary

EXPOSE 5000

RUN mkdir -p /mlflow
WORKDIR /mlflow

# CMD debe ir en UNA SOLA LÍNEA
CMD ["mlflow", "server", "--backend-store-uri", "postgresql://mlflow:mlflowpwd@postgres-server:5432/mlflowdb", "--default-artifact-root", "s3://mlflow/", "--host", "0.0.0.0", "--port", "5000"]
