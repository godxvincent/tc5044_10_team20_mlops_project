import os

import pytest

from mlops.pipeline import MLPipeline


# --- Fixtures (Accesorios) para configurar los Mocks ---
@pytest.fixture(autouse=True)
def setup_data():
    # Code to run before each test
    print("\nSetting up before test")
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    yield
    # Code to run after each test (teardown)
    os.environ.pop("ENV_FOR_DYNACONF", None)
    print("\nTearing down after test")


# ------#
def test_evaluate_dataload_step():
    pipeline = MLPipeline()
    pipeline.load_data_step("turkish_music_emotion_modified.csv")
    filas, columnas = pipeline.get_dataframe_shape()
    assert filas == 408
    assert columnas == 52


def test_evaluate_dataload_statistics_step():
    pipeline = MLPipeline()
    pipeline.load_data_step("turkish_music_emotion_modified.csv")
    df = pipeline.get_dataframe_statistics()
    # print(df)
    assert type(df) is not type(None)
    transpose = df.transpose()
    assert "max" in transpose.columns
    assert "std" in transpose.columns
    assert "mean" in transpose.columns


def test_evaluate_cleanup_step():
    pipeline = MLPipeline()
    pipeline.load_data_step("turkish_music_emotion_modified.csv")
    pipeline.clean_up_data_step()
    filas_original, columnas_original = pipeline.get_original_dataframe_shape()
    filas_limpia, columnas_limpia = pipeline.get_dataframe_shape()
    assert filas_original > filas_limpia
    assert columnas_original > columnas_limpia


def test_evaluate_cleanup_statistics_step():
    pipeline = MLPipeline()
    pipeline.load_data_step("turkish_music_emotion_modified.csv")
    pipeline.clean_up_data_step()
    df_original = pipeline.get_original_dataframe_statistics()
    df_limpio = pipeline.get_dataframe_statistics()
    assert type(df_original) is not type(None)
    assert type(df_limpio) is not type(None)
    transpose_original = df_original.transpose()
    transpose_limpio = df_limpio.transpose()
    assert "max" in transpose_original.columns
    assert "std" in transpose_original.columns
    assert "mean" in transpose_original.columns
    assert "max" in transpose_limpio.columns
    assert "std" in transpose_limpio.columns
    assert "mean" in transpose_limpio.columns
    df_combinado = df_original.merge(df_limpio, how="left", left_index=True, right_index=True, suffixes=["_x", "_y"])
    # print(df_combinado.columns)
    assert len(df_combinado[df_combinado["Class_x"] != df_combinado["Class_y"]]) > 0


def test_evaluate_training_params_step():
    # 1. Configuración: Inicializar y asignar el ModelTrainer simulado
    pipeline = MLPipeline()
    # Inyectamos el ModelTrainer "listo" para usar
    # Esto esta mal, porque debemos llamar al modelo real no a un mock externo.
    pipeline.load_data_step("turkish_music_emotion_modified.csv")
    pipeline.clean_up_data_step()
    pipeline.feature_engineering_step()
    pipeline.train_step()

    # 2. Ejecutar la función
    params = pipeline.get_model_params()

    # 3. Assertions
    assert "n_estimators" in params, "Falta 'n_estimators' en los parámetros."
    assert (
        params["n_estimators"] >= 100 and params["n_estimators"] <= 1000
    ), "El valor de n_estimators no coincide con el esperado"
    assert isinstance(params["best_params"], dict), "best_params debe ser un diccionario."
    assert "classifier__max_depth" in params["best_params"]


def test_evaluate_training_metrics_step():
    # 1. Configuración: Inicializar y asignar el ModelTrainer simulado
    pipeline = MLPipeline()
    pipeline.load_data_step("turkish_music_emotion_modified.csv")
    pipeline.clean_up_data_step()
    pipeline.feature_engineering_step()
    pipeline.train_step()
    pipeline.evaluate_step()

    # 2. Ejecutar la función (asumiendo que llama a self.__model_trainer.get_metrics())
    metrics = pipeline.get_model_metrics()

    # 3. Assertions
    # Calculando el valor esperado para el set de datos [0, 1, 0, 1, 0] y [0, 1, 0, 0, 0]
    # Accuracy (3/5) = 0.6

    # Usamos el umbral que proporcionaste (0.61)
    assert metrics["test_accuracy"] >= 0.6, "La precisión es menor a 0.60 (esperado)."

    # Afirmamos la estructura
    metricas_clave = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    for metric in metricas_clave:
        assert metric in metrics, f"Falta la clave de métrica: {metric}"
        assert 0.0 <= metrics[metric] <= 1.0, f"La métrica {metric} está fuera del rango [0.0, 1.0]."
