import os

import pytest

from mlops.dataset import DataLoader


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
    (filas, columnas) = dl_class.get_shape()
    assert filas == 408
    assert columnas == 52
