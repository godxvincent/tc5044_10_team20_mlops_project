import logging
import functools
import sys
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    log_file = None
    log_level = logging.DEBUG
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_stream = sys.stdout

class BaseLogger:
    def __init__(self, name: str, config: LoggerConfig = LoggerConfig()):
        # Clave: Solo configurar si no hay handlers configurados.
        # Pytest configura sus propios handlers, por lo que esta llamada se omitirá
        # cuando se ejecute con pytest, evitando el conflicto.
        if not logging.getLogger().handlers:
            logging.basicConfig(
                filename=config.log_file,
                # No se puede usar stream y filename al mismo tiempo.
                stream=config.log_stream if not config.log_file else None,
                level=config.log_level,
                format=config.log_format)
        self.logger = logging.getLogger(name)
        self.__decorate_functions()

    def __decorate_functions(self):        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.logger.debug(f"function {func.__name__} was called with args: {args} and kwargs: {kwargs} ")
                result = func(*args, **kwargs)
                if result:
                    self.logger.debug(f"function {func.__name__} returned: {result}")
                else:
                    self.logger.debug(f"function {func.__name__} finished correctly")
                return result
            return wrapper

        for func in self.__dir__():
            if callable(self.__getattribute__(func)) and not func.startswith("__"):
                function_itself = self.__getattribute__(func)
                self.__setattr__(func, decorator(function_itself))
                
