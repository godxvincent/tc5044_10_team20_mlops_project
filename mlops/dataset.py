"""
Propósito:
- Reunir las funciones y pasos del EDA del notebook en una API clara y reutilizable.
- Mantener firmas y lógica equivalentes: `resumen_nulos`, `valores_unicos`, `eliminar_outliers`,
  info de dataset y exportación del CSV limpio.
- Permitir uso por script/CLI y desde otros módulos.

"""


from dataclasses import dataclass
from mlops.base.steps import DataLoaderBase
from typing import Dict, List, Tuple, Optional
from pandas import DataFrame, read_csv, Series
import pandas as pd
from mlops.config import BaseDataClassModel, MLConfigLoader


@dataclass
class DataLoaderConfig(BaseDataClassModel):
    test_data_percentage: str

    def __init__(self, **kwargs):
        """
        Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)


class DataLoader(DataLoaderBase):

    def __init__(self):
        self.data_loader_config = MLConfigLoader().getParameter("data_loader", DataLoaderConfig())
        super().__init__()

    def loadFile(self, file_name: str) -> Dict:
        try:
            self.df = read_csv(file_name)
        except FileNotFoundError as e:
            self.logger.debug(f"File {file_name} not found")
            raise e

    def getShape(self) -> Tuple[int, int]:
        return self.df.shape

    def getStatistics(self) -> DataFrame:
        return self.df.describe()

    # def getTrainTestDataSet(self) -> Dict[str, DataFrame]:
    # test_size = self.data_loader_config.test_data_percentage
    # Deje esta variable solo comentada, si no va a ser usada, por favor, borrarla.

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    # )

    # --- Métodos adicionales refactorizados del notebook ---

    def get_info(self) -> None:
        """
        Imprime la información (info) del DataFrame.
        """
        if self._check_df_loaded():
            self.logger.info("Mostrando df.info():")
            self.df.info()

    def get_null_summary(self) -> Optional[Series]:
        """
        Obtiene un resumen de los valores nulos.
        """
        if self._check_df_loaded():
            nulos = self.df.isnull().sum()
            return nulos[nulos > 0].sort_values(ascending=False)
        return None

    def run_cleaning_pipeline(
        self, 
        target_col: str = 'Class', 
        cols_to_drop: List[str] = ['mixed_type_col'],
        outlier_k: float = 3.0
    ) -> None:
        """
        Ejecuta el pipeline de limpieza completo
        """
        if not self._check_df_loaded():
            return

        self.logger.info("Iniciando pipeline de limpieza de datos...")
        
        # 1. Convertir tipos y limpiar 
        self._convert_to_numeric(target_col=target_col, cols_to_drop=cols_to_drop)
        
        # 2. Imputar valores nulos 
        self._impute_missing_values(target_col=target_col, strategy='median')
        
        # 3. Eliminar outliers
        self._remove_outliers_iqr(target_col=target_col, k=outlier_k)
        
        # 4. Normalizar columna 'Class' 
        self._clean_target_column(target_col=target_col)
        
        self.logger.info(f"Pipeline de limpieza completo. Shape final: {self.getShape()}")

    def get_class_distribution(self, target_col: str = 'Class') -> Optional[Series]:
        """
        Retorna el conteo de valores de la columna objetivo.
        """
        if self._check_df_loaded(target_col):
            return self.df[target_col].value_counts()
        return None

    def get_correlation_matrix(self) -> Optional[DataFrame]:
        """
        Retorna la matriz de correlación de las variables numéricas.
        
        """
        if self._check_df_loaded():
            numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
            return numeric_df.corr()
        return None

    def save_cleaned_file(self, output_file_name: str) -> None:
        """
        Guarda el DataFrame limpio a un nuevo archivo CSV.
        
        """
        if self._check_df_loaded():
            try:
                self.df.to_csv(output_file_name, index=False)
                self.logger.info(f"Dataset limpio exportado como '{output_file_name}'")
            except Exception as e:
                self.logger.error(f"No se pudo guardar el archivo: {e}")
                raise e

    # --- Métodos privados (lógica interna del pipeline) ---

    def _check_df_loaded(self, check_col: Optional[str] = None) -> bool:
        """Helper para verificar si el DataFrame está cargado."""
        if self.df is None:
            self.logger.warning("DataFrame no cargado. Llama a loadFile() primero.")
            return False
        if check_col and check_col not in self.df.columns:
            self.logger.warning(f"La columna '{check_col}' no se encuentra en el DataFrame.")
            return False
        return True

    def _convert_to_numeric(self, target_col: str, cols_to_drop: List[str]):
        """
        Limpia y convierte columnas a numérico.
        
        """
        # 1. Dropear columnas innecesarias
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        self.logger.info(f"Columnas eliminadas: {cols_to_drop}")

        # 2. Identificar columnas que deberían ser numéricas pero son 'object'
        numeric_cols = self.df.columns.difference([target_col])
        
        for col in numeric_cols:
            # Solo aplicar limpieza de string si es 'object'
            if self.df[col].dtype == 'object':
                self.logger.debug(f"Limpiando y convirtiendo columna object: {col}")
                # Limpiar espacios
                self.df[col] = self.df[col].astype(str).str.strip().str.replace(',', '', regex=False)
            
            # 3. Convertir a numérico 
            # errors='coerce' convierte errores de conversión en NaN
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.logger.info("Conversión a numérico completada.")

    def _impute_missing_values(self, target_col: str, strategy: str = 'median'):
        """
        Imputa valores nulos para características numéricas y limpia nulos del target.
        
        """
        # 1. Imputar numéricos con la mediana
        if strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
            self.logger.info(f"Valores numéricos nulos imputados con la '{strategy}'.")

        # 2. Eliminar filas donde el target ('Class') es nulo
        initial_rows = self.df.shape[0]
        self.df.dropna(subset=[target_col], inplace=True)
        rows_dropped = initial_rows - self.df.shape[0]
        self.logger.info(f"Eliminadas {rows_dropped} filas con target ('{target_col}') nulo.")

    def _remove_outliers_iqr(self, target_col: str, k: float):
        """
        Elimina outliers usando el método IQR.
        
        """
        initial_rows = self.df.shape[0]
        df_clean = self.df.copy()
        
        # Aplicar solo a columnas numéricas (todas excepto el target)
        numeric_cols = df_clean.columns.difference([target_col])
        
        for c in numeric_cols:
            if df_clean[c].dtype in ['float64', 'int64']:
                Q1 = df_clean[c].quantile(0.25)
                Q3 = df_clean[c].quantile(0.75)
                IQR = Q3 - Q1
                low = Q1 - k * IQR
                high = Q3 + k * IQR
                # Mantener solo las filas dentro del rango
                df_clean = df_clean[(df_clean[c] >= low) & (df_clean[c] <= high)]
        
        self.df = df_clean
        rows_dropped = initial_rows - self.df.shape[0]
        self.logger.info(f"Eliminados {rows_dropped} outliers usando IQR (k={k}).")

    def _clean_target_column(self, target_col: str):
        """
        Estandariza la columna 'Class' a minúsculas y sin espacios.
        """
        if self._check_df_loaded(target_col):
            self.df[target_col] = self.df[target_col].astype(str).str.strip().str.lower()
            self.logger.info(f"Columna target '{target_col}' estandarizada (strip, lower).")


if __name__ == "__main__":
    dataLoader = DataLoader()
    dataLoader.loadFile("data/external/turkish_music_emotion_original.csv")
    print(dataLoader.getShape())
    print(dataLoader.getStatistics())
    print(dataLoader.get_info())
    print(dataLoader.get_null_summary())
    print(dataLoader.run_cleaning_pipeline())
    #dataLoader.save_cleaned_file('cleaned_data.csv') #checar ruta correcta para la exportacion
