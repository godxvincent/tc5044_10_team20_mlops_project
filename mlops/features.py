from typing import Optional

from sklearn.compose import ColumnTransformer

from mlops.base.steps import FeatureEngineProcessorBase


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
        que defina la estrategia para reducir el número de features necesarios para el modelo
        (por ejemplo, usando PCA).
        Para aplicar PCA se requiere hacer *scaling* a las columnas que se reducirán. Para garantizarlo,
        se añadió el atributo `__PCAfeaturesScaled`, que debe cambiarse a `True` cuando este método
        aplique el *scaler* a los features destinados a PCA.
        """
        pass
