
from dataclasses import dataclass
from omegaconf import OmegaConf
import pathlib
import os

from dynaconf import Dynaconf

# Pendiente revisar si hace sentido tener estos dataclass
@dataclass
class MLFlowConfig:
    server: str = ""
    port: str = ""

@dataclass
class MLConfig:
    mlFlowConfig: MLFlowConfig
    
class MLConfigLoader():

    def __init__(self):
        self.config = MLConfig(MLFlowConfig())
        self.__load()

    def __load(self):
        try:
            self.settings = Dynaconf(
                envvar_prefix = "MLOPS",
                settings_files = ["config.yaml"],
                environments = True,
                load_dotenv = True,
                env_switcher = "MLOPS_ENV",
                type = "yaml",
                merge_enabled = True,
            )
            rootParameters = self.settings.get("MLCONFIG",None)
            if rootParameters == None:
                raise FileNotFoundError("Config file not found") 
        except AttributeError as ae:
            print("Root parameters not found")
            print("Detailed error: ", ae)


    def getParameter(self, parameterName:str) -> any:
        
        rootParameters = self.settings.get("MLCONFIG",None)
        lowerCaseParameterName = parameterName.lower()
        parameter = rootParameters.get(lowerCaseParameterName, None)
        if parameter != None:
            return parameter
        
        raise ValueError(f"Parameter {parameterName} not found")


a = MLConfigLoader()
a.getParameter("mlflowconfig")