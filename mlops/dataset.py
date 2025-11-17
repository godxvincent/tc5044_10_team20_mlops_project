"""
Propósito:
- Reunir las funciones y pasos del EDA del notebook en una API clara y reutilizable.
- Mantener firmas y lógica equivalentes: `resumen_nulos`, `valores_unicos`, `eliminar_outliers`,
  info de dataset y exportación del CSV limpio.
- Permitir uso por script/CLI y desde otros módulos.

"""

import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Self, Tuple

import pandas as pd
from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import train_test_split

from mlops.base.steps import DataLoaderBase
from mlops.config import BaseDataClassModel, MLConfigLoader
from mlops.modeling.constants import EXPECTED_SCHEMA

DEFAULT_INTERIM_DIR = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "data", "interim")
DEFAULT_EXTERNAL_DIR = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "data", "external")

TARGET_COLUMN = ["Class"]


@dataclass
class DataLoaderConfig(BaseDataClassModel):
    test_size: str
    outlier_k: float

    def __init__(self, **kwargs):
        """
        Clase modelo de datos para la configuración del data loader.
        """
        super().__init__(**kwargs)


class DataCleanUpException(Exception):
    """
    Excepción personalizada para errores donde no se ha llamado la limpieza de datos.
    """

    def __init__(self, mensaje="La temperatura está fuera del rango permitido."):
        # Llamar al constructor de la clase base Exception
        super().__init__(mensaje)
        self.mensaje = mensaje

    def __str__(self):
        """Define cómo se representará la excepción cuando se imprima."""
        return "Se debe ejecutar la limpieza de datos para poder trabajar con ellos."


class DataLoader(DataLoaderBase):

    def __init__(self):
        """
        La función se encarga de cargar los parametros generales del procesor.
        """
        mlconfig_loader = MLConfigLoader()
        self.config = mlconfig_loader.getParameter("data_loader", DataLoaderConfig())
        self.general_config = mlconfig_loader.general_parameters
        super().__init__()
        self.data_cleaned = False

    def load_file(self, file_name: str) -> Dict:
        try:
            input_file = os.path.join(DEFAULT_EXTERNAL_DIR, file_name)
            self.df = read_csv(input_file)
            self.__df_original = self.df.copy()
            self.logger.warning(
                "El dataset solo fue cargado en memoria debe ejecutarse la función run_cleaning_up() explicítamente"
            )
            return {"shape": self.df.shape, "statistics": self.df.describe(include="all")}

        except FileNotFoundError as e:
            self.logger.error(f"File {file_name} not found")
            self.logger.error(f"Make sure input file is located in this directory {DEFAULT_EXTERNAL_DIR}")
            raise e

    def get_shape(self) -> Tuple[int, int]:
        if self.__check_df_loaded():
            return self.df.shape
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    def get_statistics(self) -> DataFrame:
        if self.__check_df_loaded:
            return self.df.describe(include="all")
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    def get_train_test_dataset(self) -> Dict[str, DataFrame]:

        if self.data_cleaned:
            random_state = self.general_config.random_state
            test_size = self.config.test_size
            X = self.df.drop(TARGET_COLUMN, axis=1)
            y = self.df[TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            return {
                "trainX": X_train,
                "trainY": y_train,
                "testX": X_test,
                "testY": y_test,
            }
        raise DataCleanUpException()

    # --- Métodos adicionales refactorizados del notebook ---

    def get_info(self) -> Optional[DataFrame]:
        """
        Imprime la información (info) del DataFrame.
        """
        if self.__check_df_loaded():
            self.logger.info("Mostrando df.info():")
            return self.df.info()
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    def get_null_summary(self) -> Optional[Series]:
        """
        Obtiene un resumen de los valores nulos.
        """
        if self.__check_df_loaded():
            nulos = self.df.isnull().sum()
            return nulos[nulos > 0].sort_values(ascending=False)

        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    def get_column_distribution(self, target_col: str = "Class") -> Optional[Series]:
        """
        Retorna el conteo de valores de la columna objetivo.
        """
        if self.__check_df_loaded(target_col):
            self.logger.info(f"Distribución de la columna {target_col}:")
            return self.df[target_col].value_counts()
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    def get_correlation_matrix(self) -> Optional[DataFrame]:
        """
        Retorna la matriz de correlación de las variables numéricas.
        """
        if self.__check_df_loaded():
            numeric_df = self.df.select_dtypes(include="number")
            return numeric_df.corr()
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    #
    def run_cleaning_up(self, save_cleaned_file: bool = False, output_file_name: str = "cleaned_data.csv") -> Self:
        """
        Esta función se encarga de garantizar que el dataset que se procesara tiene la estructura minima
        requerida para los siguientes pasos.
        En esencia elimina cualquier columna que no siga el esquema de datos esperado.
        Por otra parte, garantiza que cualquier dato NaN en las columnas numericas o flotantes
        tengan valores numéricos.
        """
        if not self.__check_df_loaded():
            raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

        self.logger.info("Iniciando pipeline de limpieza de datos...")

        # 1. Drop columns not follow expected schema
        set_valid_columns = set(EXPECTED_SCHEMA.keys())
        set_df_columns = set(self.df.columns)
        invalid_columns = set_df_columns - set_valid_columns
        if invalid_columns:
            self.df = self.df.drop(columns=invalid_columns, errors="ignore")
            self.logger.warning(f"Columnas inválidas eliminadas: {invalid_columns}")

        # 2. Convertir tipos y limpiar NaN values
        self.__convert_to_numeric()

        # 3. Eliminar outliers
        self.__remove_outliers_iqr()

        # 4. Normalizar columna 'Class'
        self.__clean_target_column(TARGET_COLUMN[0])

        self.logger.info(f"Limpieza inicial del dataset completada. Shape final: {self.get_shape()}")

        self.data_cleaned = True

        if save_cleaned_file:
            self.__save_cleaned_file(output_file_name)

    def __save_cleaned_file(self, file_name: str) -> None:
        """
        Guarda el DataFrame limpio a un nuevo archivo CSV.

        """
        if self.__check_df_loaded():
            try:
                os.makedirs(DEFAULT_INTERIM_DIR, exist_ok=True)
                output_file_name = os.path.join(DEFAULT_INTERIM_DIR, file_name)
                self.df.to_csv(output_file_name, index=False)
                self.logger.info(f"Dataset limpio exportado como {output_file_name}")
            except Exception as e:
                self.logger.error(f"No se pudo guardar el archivo: {e}")
                raise e

    # --- Métodos privados (lógica interna del pipeline) ---

    def __check_df_loaded(self, check_col: Optional[str] = None) -> bool:
        """Helper para verificar si el DataFrame está cargado."""
        if self.df is None:
            self.logger.warning("DataFrame no cargado. Llama a loadFile() primero.")
            return False
        if check_col and check_col not in self.df.columns:
            self.logger.warning(f"La columna {check_col} no se encuentra en el DataFrame.")
            return False
        return True

    def __convert_to_numeric(self):
        """
        Limpia y convierte columnas a numérico.
        """
        numeric_cols = set(EXPECTED_SCHEMA.keys()).difference(TARGET_COLUMN)
        for col in numeric_cols:
            # Solo aplicar limpieza de string si es 'object'
            if self.df[col].dtype == "object":
                self.logger.debug(f"Limpiando y convirtiendo columna object: {col}")
                # Limpiar espacios
                self.df[col] = self.df[col].astype(str).str.strip().str.replace(",", "", regex=False)

            # Convertir a numérico y rellenar con ceros
            # errors='coerce' convierte errores de conversión en NaN
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.logger.info("Conversión a numérico completada.")

    def __remove_outliers_iqr(self):
        """
        Elimina outliers usando el método IQR.
        """
        outlier_k = self.config.outlier_k
        initial_rows = self.df.shape[0]
        df_clean = self.df.copy()

        # Aplicar solo a columnas numéricas (todas excepto el target)
        numeric_cols = df_clean.columns.difference(TARGET_COLUMN)

        for column in numeric_cols:
            if df_clean[column].dtype in ["float64", "int64"]:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                low = Q1 - outlier_k * IQR
                high = Q3 + outlier_k * IQR
                # Mantener solo las filas dentro del rango
                df_clean = df_clean[(df_clean[column] >= low) & (df_clean[column] <= high)]

        self.df = df_clean
        rows_dropped = initial_rows - self.df.shape[0]
        self.logger.info(f"Eliminados {rows_dropped} outliers usando IQR (outlier_k={outlier_k}).")

    def __clean_target_column(self, target_col: str):
        """
        Estandariza la columna 'Class' a minúsculas y sin espacios.
        """
        if self.__check_df_loaded(target_col):
            # 1. Eliminar filas donde el target ('Class') es nulo
            initial_rows = self.df.shape[0]
            self.df.dropna(subset=[target_col], inplace=True)
            rows_dropped = initial_rows - self.df.shape[0]
            self.logger.info(f"Eliminadas {rows_dropped} filas con target ({target_col}) nulo.")

            # 2. Estandarizar el valor de la columna target
            self.df[target_col] = self.df[target_col].astype(str).str.strip().str.lower()
            self.logger.info(f"Columna target {target_col} estandarizada (strip, lower).")

    def get_original_shape(self) -> Tuple[int, int]:
        if self.__check_df_loaded():
            return self.__df_original.shape
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")

    def get_original_statistics(self) -> DataFrame:
        if self.__check_df_loaded():
            return self.__df_original.describe(include="all")
        raise Exception("DataFrame no cargado. Llama a loadFile() primero.")


# if __name__ == "__main__":
#     dataLoader = DataLoader()
#     dataLoader.load_file("data/external/turkish_music_emotion_modified.csv")
#     print(dataLoader.get_shape())
#     print(dataLoader.get_statistics())
#     print(dataLoader.get_info())
#     print(dataLoader.get_null_summary())
#     print(dataLoader.run_cleaning_up(save_cleaned_file=True, output_file_name='cleaned_data_refactored.csv'))
# dataLoader.save_cleaned_file('cleaned_data.csv') #checar ruta correcta para la exportacion
