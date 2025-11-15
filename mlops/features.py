from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer

from mlops.base.steps import FeatureEngineProcessorBase
from mlops.config import BaseDataClassModel, MLConfigLoader
from mlops.modeling.constants import EXPECTED_SCHEMA


@dataclass
class FeatureEngineProcessorConfig(BaseDataClassModel):
    k_features: int
    n_components: int
    n_components_f: float

    def __init__(self, **kwargs):
        """
        Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)


class FeatureEngineProcessor(FeatureEngineProcessorBase):

    def __init__(self):
        self.config = MLConfigLoader().getParameter("feature_engine", FeatureEngineProcessorConfig())
        self.numerical_features = [x for x in EXPECTED_SCHEMA if EXPECTED_SCHEMA[x] in ["float64", "int64"]]
        super().__init__()

    def _createImputer(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de imputar los valores faltantes.
        """
        return ("imputer", SimpleImputer(strategy="median"), self.numerical_features)

    def _createStandardizer(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de estandarizar los valores como la clase.
        """
        pass

    def _createScaler(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de escalar los valores de un conjunto de features.
        """
        self._PCAfeaturesScaled = True
        return ("scaler", StandardScaler(), self.numerical_features)

    def _createOutlierProcessor(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que se encargue de lidiar con los valores atipicos.
        """
        pass

    def _createFeatureReducer(self) -> Optional[Tuple]:
        """
        La implementación concreta de esta función debe contemplar crear un objeto del tipo ColumnTransformer
        que defina la estrategia para reducir el número de features necesarios para el modelo
        (por ejemplo, usando PCA).
        Para aplicar PCA se requiere hacer *scaling* a las columnas que se reducirán. Para garantizarlo,
        se añadió el atributo `__PCAfeaturesScaled`, que debe cambiarse a `True` cuando este método
        aplique el *scaler* a los features destinados a PCA.
        """
        if self.config.n_components_f and self.config.n_components:
            self.logger.warning(
                "Esta configurado el numero de componentes en entero y decimal se usará por default el decimal"
            )
        if self.config.n_components_f:
            return ("PCA", PCA(n_components=self.config.n_components_f), self.numerical_features)
        elif self.config.n_components:
            return ("PCA", PCA(n_components=self.config.n_components), self.numerical_features)
        else:
            raise ValueError(
                "Al menos debe haber un numero de componentes definido para aplicar PCA (Verificar la configuracion)"
            )
