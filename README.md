# Team20 MLOps Project
Private project for MLOps course 

## Inicializacion de este repositorio

Antes de iniciar a trabajar con este repositorio por favor garantizar que tiene python 3.13.1 instalado (usando pyenv es una buena alternativa).

`pyenv install 3.13.1`

Una vez el python este garantizado, deben crear un entorno virtual para ello ejecutar la siguiente instrucción.

`python -m venv venv`

Luego de crear el ambiente virtual debes activar el ambiente virtual con el siguiente comando.

`source venv/bin/activate`


Una vez este el ambiente virtual activado puede instalar los paquetes del archivo requirements.txt usando el comando siguiente

`pip install -r requirements.txt`

Esto instalara toda la paqueteria necesaria.


## Configuración de DVC (primera vez)

Este proyecto ya cuenta con una configuración inical de DVC, dado que estamos trabajando con una primera version que usa configuración local, deberá solicitar las credenciales de la API de google para poder autenticar las peticiones de DVC. 
Con dichas credenciales debera crear un archivo dentro del folder `.dvc` con nombre `config.local` el cual tendra una estructura como la siguiente


```txt
['remote "datasets"']
    gdrive_acknowledge_abuse = true
    gdrive_client_id = xxxxxxxxxxx-yyyyyyyyyyyyyyyyyyyyyy.apps.googleusercontent.com
    gdrive_client_secret = XXXXXXX-yyyyyyyyyyyyyyyyyy
```

Pueden añadir las credenciales mediante los siguientes comandos

`dvc remote modify datasets gdrive_client_id 'client_id'`

`dvc remote modify datasets gdrive_client_secret 'secret'`

Para descargar los datasets de Drive pueden correr el siguiente comando

`dvc pull`

Probablemente tienes que dar permiso al uso Google Drive API para tu proyecto Google Cloud

## Uso de DVC

### Incluir nuevos archivos

En el caso de querer incluir nuevos de entrada estos deben ser alojados en la ruta `datasets/input`.
Para los archivos de salida se debe utilizar los la ruta `dataset/output`

Tanto los archivos nuevos como los archivos generados por el proceso deben ser versionados.

Para versionar un archivo nuevo o uno modificado se debe usar el siguiente comando.

`dvc add <paht>/<file_name>` pej. `dvc add datasets/input/test.txt`

Una vez se han agregado al versionamiento, se deben subir  al repositorio remoto. Para ello se debe ejecutar el siguiente comando.

`dvc push` 

Esto subira al google drive del equipo todos los archivos que se hayan versionado.

### Descargar archivos cargados en el repositorio remoto.

Como verá en la configuración inicial ya se han subido los archivos iniciales del ejercicio por lo tanto para descargar la versión cargada del dataset, ejecute el siguiente comando.

`dvc pull` pej. `dvc add datasets/input/test.txt`

### Links de utilidad de DVC

* [DVC Start](https://dvc.org/doc/start)
* [Config Files](https://dvc.org/doc/user-guide/project-structure/configuration)
* [Configuracion de Google Drive](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)

## Configuración de Docker y MLflow

Este proyecto utiliza Docker Compose para gestionar los servicios de MLflow, PostgreSQL y MinIO.

### Configuración inicial

Antes de levantar los servicios, es necesario configurar las variables de entorno:

1. Copia el archivo `env.example` a `.env`:
   ```bash
   cp env.example .env
   ```

2. (Opcional) Edita el archivo `.env` si necesitas modificar las credenciales o puertos por defecto:
   - `PG_USER`, `PG_PASSWORD`, `PG_DATABASE`: Credenciales de PostgreSQL
   - `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`: Credenciales de MinIO
   - `MINIO_PORT`, `MINIO_CONSOLE_PORT`: Puertos para MinIO y su consola web
   - `MLFLOW_PORT`: Puerto para el servidor de MLflow

### Levantar los servicios

Para levantar todos los servicios (MLflow, PostgreSQL, MinIO) ejecuta:

```bash
docker compose up -d
```

Este comando iniciará:
- **MLflow Server**: Disponible en `http://localhost:5000`
- **PostgreSQL**: Base de datos para el backend store de MLflow (puerto 5432)
- **MinIO**: Servidor S3-compatible para almacenar artifacts (puerto 9000, consola en 9001)

### Verificar el estado de los servicios

Para ver el estado de los contenedores:

```bash
docker compose ps
```

### Ver los logs

Para ver los logs de todos los servicios:

```bash
docker compose logs -f
```

Para ver los logs de un servicio específico:

```bash
docker compose logs -f mlflow
docker compose logs -f postgres
docker compose logs -f s3
```

### Detener los servicios

Para detener todos los servicios:

```bash
docker compose down
```

Para detener y eliminar los volúmenes (esto eliminará los datos almacenados):

```bash
docker compose down -v
```

### Acceso a los servicios

Una vez levantados los servicios, puedes acceder a:

- **MLflow UI**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (usuario y contraseña según `.env`)

## MLFlow

MLFlow es una plataforma para gestionar el ciclo de vida de ML:

- Tracking: registra parámetros, métricas, artefactos y código de cada experimento.
- Models: estandariza y versiona modelos (formato empaquetado y "flavors").
- Model Registry: catálogo/registro con versiones y stages (None, Staging, Production, Archived).

El servidor de MLflow está configurado para ejecutarse en Docker. Si prefieres ejecutarlo localmente sin Docker, puedes usar:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

El cliente puede conectarse al servidor de MLflow mediante la siguiente línea de código:

```python
mlflow.set_tracking_uri("http://localhost:5000")  # Con Docker
# o
mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Sin Docker
```

La configuración del proyecto en `config.yaml` está configurada para usar el servidor de Docker por defecto (`localhost:5000`).

### Links de utilidad de MLFlow

* [MLFlow Docs](https://mlflow.org/docs/2.5.0/quickstart.html#)

## Linting y Formateo Automático de Código

Como parte de la reestructuración del proyecto, se integraron herramientas para mantener un código limpio, consistente y conforme a las buenas prácticas de Python (PEP8).

### Herramientas instaladas
- **Black** → formateador automático de código.  
- **Flake8** → analizador de estilo y detección de errores.  
- **isort** → ordena automáticamente las importaciones.  
- **pre-commit** → ejecuta verificaciones antes de cada commit.  

### Configuración automática
Estas herramientas se ejecutan de forma local y también en el pipeline de **GitHub Actions** (ver `.github/workflows/lint.yml`).

#### Configuración local
1. Instalar dependencias:
    pip install black flake8 isort pre-commit
2. Instalar los hooks de pre-commit:
    pre-commit install
3. Ejecutar el formato y análisis manualmente (opcional):
    black .
    isort .
    flake8 .
#### Ejecución con Makefile
También se pueden ejecutar con:
    make lint         # Corre Flake8
    make format       # Aplica Black + isort
    make lint-report  # Genera reportes de linters

## Visualización de Gráficos
La clase llamada Plotter en mlops/plots.py permite generar gráficos de forma flexible y segura tanto en entornos de notebooks como de terminal

### Características principales
1. Detección automática del entorno: notebook o script.
2. Soporte para backends no interactivos (Agg) para ejecución en CI/CD.
3. Configurable mediante la clase auxiliar PlotConfig.
4. Métodos disponibles:
- **plot()** → gráfico de líneas.
- **bar()** → gráfico de barras.
- **scatter()** → gráfico de dispersión.

## Serving del modelo con FastAPI

Para exponer el modelo vía API se creó el módulo `mlops.api` con:

- `mlops/api/model_loader.py`: carga el modelo registrado en MLflow usando `MODEL_URI`
  (por ejemplo: `models:/turkish_music_emotion_rf/1`).
- `mlops/api/schemas.py`: define los esquemas Pydantic para la entrada y salida:
  - `PredictionRequest`: recibe un diccionario de features (`features: Dict[str, float>`).
  - `PredictionResponse`: devuelve la predicción y la ruta del modelo (`model_uri`).
- `mlops/api/main.py`: define la aplicación FastAPI con:
  - `GET /health`: verifica el estado de la API y del modelo.
  - `POST /predict`: recibe un JSON con `features` y devuelve la predicción.

### Ejecución local

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_URI=models:/turkish_music_emotion_rf/1
uvicorn mlops.api.main:app --reload
