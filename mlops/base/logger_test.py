import pytest

from mlops.base.logger import BaseLogger

class OtherClass(BaseLogger):

    def __init__(self):
        super().__init__()

    def sum(self, a, b):
        return a + b
    
    def sayHello(self, name):
        self.logger.info(f"Hello {name}")
    

def test_sum():
    testClass = OtherClass()
    assert testClass.sum(1,2) == 3

def test_sayHello():
    testClass = OtherClass()
    testClass.sayHello("Test")

