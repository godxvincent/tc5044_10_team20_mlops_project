import logging
import functools
import sys
from dataclasses import dataclass
from mlops.config import MLConfigLoader, BaseDataClassModel



@dataclass
class LoggerConfig(BaseDataClassModel):
    log_level:str
    log_format:str
    log_date_format:str
    log_stream:str = None
    log_file:str = None

    def __init__(self, **kwargs):
        """
            Clase modelo de datos para la configuración del logger.
        """
        super().__init__(**kwargs)

class BaseLogger:
    def __init__(self, name: str):
        # Clave: Solo configurar si no hay handlers configurados.
        # Pytest configura sus propios handlers, por lo que esta llamada se omitirá
        # cuando se ejecute con pytest, evitando el conflicto.
        self.__logger_config = LoggerConfig()
        self.__load_logger_config()
        if not logging.getLogger().handlers:
            logging.basicConfig(
                filename=self.__logger_config.log_file,
                # No se puede usar stream y filename al mismo tiempo.
                stream=self.__logger_config.log_stream if not self.__logger_config.log_file else None,
                level=self.__logger_config.log_level,
                format=self.__logger_config.log_format)
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

    def __load_logger_config(self):
        configLoader = MLConfigLoader()
        self.__logger_config = configLoader.getParameter("logger", self.__logger_config)
