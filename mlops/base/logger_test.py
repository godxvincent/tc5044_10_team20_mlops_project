from mlops.base.logger import BaseLogger

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
