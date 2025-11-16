import os

import pytest

from mlops.dataset import DataLoader
from mlops.modeling.constants import EXPECTED_SCHEMA


@pytest.fixture(autouse=True)
def setup_data():
    # Code to run before each test
    print("\nSetting up before test")
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    yield
    # Code to run after each test (teardown)
    os.environ.pop("ENV_FOR_DYNACONF", None)
    print("\nTearing down after test")


def test_dataloader_class_get_shape():
    dl_class = DataLoader()
    dl_class.load_file("turkish_music_emotion_modified.csv")
    print(EXPECTED_SCHEMA)
    (filas, columnas) = dl_class.get_shape()
    assert filas == 408
    assert columnas == 52


# Verifica que el método load_file carga correctamente un archivo CSV y devuelve un DataFrame.


def test_load_file_returns_dictionary():
    dl_class = DataLoader()
    result = dl_class.load_file("turkish_music_emotion_modified.csv")
    assert result is not None
    assert result.get("shape") is not None
    assert result.get("shape")[0] > 0


# Verifica que get_shape() coincide con la forma del DataFrame cargado


def test_get_shape_consistency():
    dl_class = DataLoader()
    result = dl_class.load_file("turkish_music_emotion_modified.csv")
    shape_from_load_file = result["shape"]
    shape_from_method = dl_class.get_shape()
    assert shape_from_load_file == shape_from_method


# Verifica que se manda mensaje de error si el archivo no existe


def test_load_file_invalid_path():
    dl_class = DataLoader()
    with pytest.raises(FileNotFoundError):
        dl_class.load_file("archivo_inexistente.csv")


# Si DataLoader tiene lógica para validar el esquema, deberías probar que detecta errores.


def test_schema_validation_logic():
    dl_class = DataLoader()
    result = dl_class.load_file("turkish_music_emotion_modified.csv")
    mismatches = [col for col in EXPECTED_SCHEMA if col not in result["statistics"].columns]
    assert len(mismatches) == 0
