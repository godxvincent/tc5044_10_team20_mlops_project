import os
from mlops.base.logger import BaseLogger
from unittest.mock import Mock, patch
import pytest
from logging import Logger


class OtherClass(BaseLogger):

    def __init__(self, name="OtherClassLogger"):
        # Pasamos el nombre del logger al padre
        super().__init__(name=name)

    def sum(self, a, b):
        return a + b
    
    def sayHello(self, name):
        self.logger.info(f"Hello {name}")

def test_sum_logs_correctly(caplog):
    """
    Test que verifica que el decorador de logging funciona para el método sum.
    """
    testClass = OtherClass()
    assert testClass.sum(1,2) == 3
    # Verificar que el decorador automático registró la llamada a la función
    assert "function sum was called" in caplog.text
    assert "function sum returned: 3" in caplog.text

def test_sayHello_logs_message(caplog):
    """Test que verifica que el método sayHello registra el mensaje esperado."""
    testClass = OtherClass()
    testClass.sayHello("Test")
    # Verificar que el mensaje DEBUG fue registrado
    assert "Hello Test" in caplog.text
    assert caplog.records[0].levelname == "DEBUG"

class LoggerMock(Logger):
    def __init__(self, name, level = 0):
        super().__init__(name, level)
        self.handlers = []


@patch("logging.getLogger")
def test_mock_logger(mock_requests_getLogger, caplog):

    os.environ["ENV_FOR_DYNACONF"] = "testing"
    lm = LoggerMock("testLogger")

    mock_requests_getLogger.return_value = lm
    testClass = OtherClass()
    testClass.sayHello("Test")
    mock_requests_getLogger.assert_called()

    os.environ.pop("ENV_FOR_DYNACONF", None)