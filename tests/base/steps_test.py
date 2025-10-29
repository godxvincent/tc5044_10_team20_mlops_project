import pytest
from mlops.base.steps import DataLoaderBase, defaultFunctionTest

def test_defaultFunctionTest():
    result = defaultFunctionTest(5,5)
    assert result == 10