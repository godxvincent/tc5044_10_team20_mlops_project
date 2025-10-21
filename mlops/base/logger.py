import logging
import functools

class BaseLogger:
    def __init__(self, log_file="app.log"):
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.__decorate_functions()

    def __decorate_functions(self):
        def decorator(func):
            # @functools.wraps(func)
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
                


