FROM python:3.9-slim

RUN apt update && \ 
    apt install -y gcc libpq-dev && \
    pip install mlflow minio \
    boto3 \
    pymysql \
    cryptography \
    psycopg2==2.8.5

EXPOSE 5000

RUN mkdir -p /mlflow
WORKDIR /mlflow