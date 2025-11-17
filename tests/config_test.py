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


# No necesitas definir __init__ manualmente si usas @dataclass, a menos que estés haciendo algo especial. En este caso, puedes eliminarlo:


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
    assert isinstance(logger_config.log_level, str), "Error: log_level debe ser string"
    assert logger_config.log_format is not None, "Error: log_format no cargado"
    assert logger_config.log_date_format is not None, "Error: log_date_format no cargado"
