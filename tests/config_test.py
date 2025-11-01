import os
from dataclasses import dataclass

import pytest

from mlops.config import BaseDataClassModel, MLConfigLoader


@dataclass
class LoggerConfigDePrueba(BaseDataClassModel):
    """
    Clase modelo del config logger para propositos de testeo.
    """

    log_level: str
    log_format: str
    log_date_format: str
    log_stream: str = None
    log_file: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture(autouse=True)
def setup_data():
    # Code to run before each test
    print("\nSetting up before test")
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    yield
    # Code to run after each test (teardown)
    os.environ.pop("ENV_FOR_DYNACONF", None)
    print("\nTearing down after test")


def test_loadParseConfig():
    loggerConfig = LoggerConfigDePrueba()
    configLoader = MLConfigLoader()
    logger_config = configLoader.getParameter("logger", loggerConfig)
    assert logger_config.log_stream is not None, "Error: No se pudo cargar la configuración"
