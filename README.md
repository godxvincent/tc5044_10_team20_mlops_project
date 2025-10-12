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

## MLFlow

MLFlow es una plataforma para gestionar el ciclo de vida de ML:

- Tracking: registra parámetros, métricas, artefactos y código de cada experimento.
- Models: estandariza y versiona modelos (formato empaquetado y “flavors”).
- Model Registry: catálogo/registro con versiones y stages (None, Staging, Production, Archived).

MLFlow necesita un server donde almacenara los resultados, para correr este server en local pueden correr el siguiente comando

`mlflow server --host 127.0.0.1 --port 8080` 

El cliente puede conectarse al servidor de mlflow mediante la siguiente linea de codigo

`mlflow.set_tracking_uri("http://127.0.0.1:8080")`

MLFlow creara las carpetas /mlruns y /mlartifacts para almacenar los resultados de los modelos, posteriormente trabajaremos para tener un servidor externo donde guardaremos estos resultados.

### Links de utilidad de MLFlow

* [MLFlow Docs](https://mlflow.org/docs/2.5.0/quickstart.html#)

Nota de prueba de Miriam
Nota de cambio1 de Esmeralda