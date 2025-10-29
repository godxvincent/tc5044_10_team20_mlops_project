from mlops.base.steps import FeatureEngineProcessorBase
from sklearn.compose import ColumnTransformer
from typing import Optional

class FeatureEngineProcessor(FeatureEngineProcessorBase):

    def __init__(self):
        super().__init__()

    def _createImputer(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de imputar los valores faltantes.
        """
        pass

    def _createStandardizer(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de estandarizar los valores como la clase.
        """
        pass

    def _createScaler(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de escalar los valores de un conjunto de features.
        """
        pass

    def _createOutlierProcessor(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de lidiar con los valores atipicos.
        """
        pass

    def _createFeatureReducer(self) -> Optional[ColumnTransformer]:
        """
            La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer 
            que se encargue de definir la estrategía para reducir el número de features necesarios para el modelo.
            Por ejemplo hacer uso de PCA.
            No olvidar que para aplicar PCA se requiere hacer scaling the las columnas que se van a reducir. Para garantizar 
            esto se ha añadido el atributo __PCAfeaturesScaled que debe ser cambiado a True si este metodo si hizo el scaler de 
            los feature para PCA.
        """
        pass