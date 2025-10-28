from typing import Any
from dynaconf import Dynaconf
from dataclasses import dataclass

DEFAULT_ROOT_PARAMS = "MLCONFIG"

@dataclass
class BaseDataClassModel:
    """
        Esta clase será usada como base para la extracción de datos de configuración. 
        Cualquier nuevo dataclass object debera heredar de esta clase para que funcione correctamente el parseo de la configuración.
    """

    def __init__(self, **kwargs):
        self.update_from_dict(kwargs)

    def update_from_dict(self, data: dict):
        for field in self.__dataclass_fields__.values():
            if field.name in data:
                setattr(self, field.name, data[field.name])


class MLConfigLoader():

    def __init__(self):
        self.settings = dict()
        self.__load()

    def __load(self):
        try:
            self.settings = Dynaconf(
                envvar_prefix = "MLOPS",
                settings_files = ["config.yaml"],
                environments = True,
                load_dotenv = True,
                env_switcher = "MLOPS_ENV",
                merge_enabled = True,
            )
            rootParameters = self.settings.get(DEFAULT_ROOT_PARAMS,None)
            if rootParameters == None:
                raise FileNotFoundError("Config file not found") 
        except AttributeError as ae:
            print("Root parameters not found")
            print("Detailed error: ", ae)


    def getParameter(self, parameterName:str, object_instance:BaseDataClassModel) -> Any:
        rootParameters = self.settings.get(DEFAULT_ROOT_PARAMS,None)
        lowerCaseParameterName = parameterName.lower()
        parameter = rootParameters.get(lowerCaseParameterName, None)
        if parameter != None:
            try:
                object_instance.update_from_dict(parameter.to_dict())
                return object_instance
            except Exception as e:
                print("Error: ", e)      
        
        raise ValueError(f"Parameter {parameterName} not found")

