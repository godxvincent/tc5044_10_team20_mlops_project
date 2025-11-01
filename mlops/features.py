from dataclasses import dataclass
from typing import Optional

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import PCA

from mlops.base.steps import FeatureEngineProcessorBase
from mlops.config import BaseDataClassModel, MLConfigLoader


@dataclass
class FeatureEngineProcessorConfig(BaseDataClassModel):
    k_features: int
    n_components: int

    def __init__(self, **kwargs):
        """
        Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)


class FeatureEngineProcessor(FeatureEngineProcessorBase):

    def __init__(self):
        self.config = MLConfigLoader().getParameter("feature_engine", FeatureEngineProcessorConfig())
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

        return make_column_transformer((PCA(n_components=self.config.n_components), self.numerical_features))
