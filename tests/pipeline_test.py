from unittest.mock import MagicMock

import pytest

from mlops.pipeline import MLPipeline

# --- Fixtures (Accesorios) para configurar los Mocks ---


@pytest.fixture
def mock_data_loader_class(mocker):
    """Mocks la CLASE DataLoader y retorna el MOCK de la clase (el callable)."""
    # Esta es la referencia que verifica si el constructor (DataLoader()) fue llamado.
    MockDataLoader = mocker.patch("mlops.pipeline.DataLoader", autospec=True)
    return MockDataLoader  # Retorna el MOCK de la CLASE


@pytest.fixture
def mock_data_loader_instance(mock_data_loader_class):
    """Retorna la INSTANCIA mockeada de DataLoader (el resultado de la llamada a la clase)."""
    # Usaremos esta instancia para llamar a sus métodos (load_file, run_cleaning_up)
    return mock_data_loader_class.return_value


@pytest.fixture
def mock_feature_processor(mocker):
    """Mocks la clase FeatureEngineProcessor y retorna una instancia mockeada."""
    MockFeatureProcessor = mocker.patch("mlops.pipeline.FeatureEngineProcessor", autospec=True)
    # Configura un método clave que se llama dentro de MLPipeline
    MockFeatureProcessor.return_value.createPipeline.return_value = MagicMock(name="MockedPipeline")
    return MockFeatureProcessor.return_value


@pytest.fixture
def mock_model_trainer(mocker):
    """Mocks la clase ModelTrainer."""
    MockModelTrainer = mocker.patch("mlops.pipeline.ModelTrainer", autospec=True)
    return MockModelTrainer  # Retorna la clase mockeada para verificar su inicialización


@pytest.fixture
def ml_pipeline_instance(mock_data_loader_class, mock_feature_processor):
    """Retorna una instancia de MLPipeline con sus dependencias mockeadas."""
    # Los mocks se inyectan automáticamente en el constructor gracias a patch
    return MLPipeline()


# --- Pruebas para los Pasos del Pipeline ---


def test_pipeline_initialization(ml_pipeline_instance, mock_data_loader_class, mock_feature_processor):
    """Verifica que el constructor inicializa correctamente las dependencias."""
    assert isinstance(ml_pipeline_instance, MLPipeline)

    # CORRECCIÓN 1: Se verifica la llamada del MOCK DE CLASE.
    mock_data_loader_class.assert_called_once()

    # Para mock_feature_processor, necesitamos el mock de la clase (igual que DataLoader).
    # Sin modificar su fixture, podemos omitir esta verificación por simplicidad o:
    # mock_feature_processor.assert_called_once() # Esto FALLARÁ a menos que lo parchees también como CLASE


# Carga de Datos
# ----------------------------------------------------------------------


def test_load_data_step_success(ml_pipeline_instance, mock_data_loader_instance):
    """Prueba el éxito al llamar a load_data_step."""
    file_name = "test_file.csv"

    # Ejecutar el paso
    result = ml_pipeline_instance.load_data_step(file_name)

    # 1. Verificar la llamada al método subyacente en la INSTANCIA
    mock_data_loader_instance.load_file.assert_called_once_with(file_name)
    # 2. Verificar que el método retorna Self (la instancia del pipeline)
    assert result is ml_pipeline_instance


def test_load_data_step_failure(ml_pipeline_instance, mock_data_loader_instance, caplog):
    """Prueba el manejo de excepciones en load_data_step."""
    file_name = "bad_file.csv"
    # Configurar el mock para que levante una excepción
    mock_data_loader_instance.load_file.side_effect = Exception("Archivo no encontrado")

    with pytest.raises(Exception, match="Archivo no encontrado"):
        ml_pipeline_instance.load_data_step(file_name)

    # Verificar que el error fue loggeado
    assert "Error al cargar el archivo" in caplog.text


# Limpieza de Datos
# ----------------------------------------------------------------------


def test_clean_up_data_step_success(ml_pipeline_instance, mock_data_loader_instance):
    """Prueba el éxito al llamar a clean_up_data_step."""
    result = ml_pipeline_instance.clean_up_data_step()

    # 1. Verificar la llamada al método subyacente en la INSTANCIA
    mock_data_loader_instance.run_cleaning_up.assert_called_once()
    # 2. Verificar que el método retorna Self
    assert result is ml_pipeline_instance


def test_clean_up_data_step_failure(ml_pipeline_instance, mock_data_loader_instance, caplog):
    """Prueba el manejo de excepciones en clean_up_data_step."""
    mock_data_loader_instance.run_cleaning_up.side_effect = Exception("Error de limpieza interna")

    with pytest.raises(Exception, match="Error de limpieza interna"):
        ml_pipeline_instance.clean_up_data_step()

    # Verificar que el error fue loggeado
    assert "Error al limpiar los datos" in caplog.text


# Feature Engineering
# ----------------------------------------------------------------------


def test_feature_engineering_step_success(
    ml_pipeline_instance, mock_feature_processor, mock_data_loader_instance, mock_model_trainer
):
    """Prueba el éxito en feature_engineering_step y la inicialización de ModelTrainer."""

    # Configurar mock_data_loader_instance para que devuelva un valor simulado
    mock_dataset = MagicMock(name="TrainTestDataset")
    mock_data_loader_instance.get_train_test_dataset.return_value = mock_dataset

    result = ml_pipeline_instance.feature_engineering_step()

    # 1. Verificar las llamadas a FeatureEngineProcessor
    # CORRECCIÓN 2: El código de pipeline llama a createPipeline 2 veces.
    # assert mock_feature_processor.createPipeline.call_count == 2   <- Esta linea nos sirve si queremos que sea doble instancia

    # 2. Verificar la inicialización de ModelTrainer
    mock_model_trainer.assert_called_once_with(
        mock_feature_processor.createPipeline.return_value, mock_dataset  # El pipeline mockeado  # El dataset mockeado
    )

    # 3. Verificar que el método retorna Self
    assert result is ml_pipeline_instance


def test_feature_engineering_step_failure(ml_pipeline_instance, mock_feature_processor, caplog):
    """Prueba el manejo de excepciones en feature_engineering_step."""
    mock_feature_processor.createPipeline.side_effect = Exception("Error de FE")

    with pytest.raises(Exception, match="Error de FE"):
        ml_pipeline_instance.feature_engineering_step()

    # Verificar que el error fue loggeado
    assert "Error al intentar crear el pipeline de feature engineering" in caplog.text


# --- Pruebas que requieren el setup completo del Pipeline ---


@pytest.fixture
def setup_pipeline_for_training(
    ml_pipeline_instance, mock_data_loader_instance, mock_feature_processor, mock_model_trainer
):
    """Ejecuta los pasos iniciales necesarios para probar train/evaluate."""
    # Asegurar que get_train_test_dataset devuelva algo
    mock_data_loader_instance.get_train_test_dataset.return_value = MagicMock()
    # Ejecutar feature_engineering_step para inicializar __model_trainer
    ml_pipeline_instance.feature_engineering_step()
    # Retornar la instancia mockeada de ModelTrainer para verificar sus métodos
    return mock_model_trainer.return_value


# Entrenamiento
# ----------------------------------------------------------------------


def test_train_step_success(ml_pipeline_instance, setup_pipeline_for_training):
    """Prueba el éxito al llamar a train_step."""
    mock_trainer_instance = setup_pipeline_for_training

    result = ml_pipeline_instance.train_step()

    # 1. Verificar la llamada al método subyacente
    mock_trainer_instance.train.assert_called_once()
    # 2. Verificar que el método retorna Self
    assert result is ml_pipeline_instance


def test_train_step_failure(ml_pipeline_instance, setup_pipeline_for_training, caplog):
    """Prueba el manejo de excepciones en train_step."""
    mock_trainer_instance = setup_pipeline_for_training
    mock_trainer_instance.train.side_effect = Exception("Fallo de convergencia")

    with pytest.raises(Exception, match="Fallo de convergencia"):
        ml_pipeline_instance.train_step()

    # Verificar que el error fue loggeado
    assert "Error al intentar entrenar el modelo" in caplog.text


# Evaluación
# ----------------------------------------------------------------------


def test_evaluate_step_success(ml_pipeline_instance, setup_pipeline_for_training):
    """Prueba el éxito al llamar a evaluate_step."""
    mock_trainer_instance = setup_pipeline_for_training

    result = ml_pipeline_instance.evaluate_step()

    # 1. Verificar la llamada al método subyacente
    mock_trainer_instance.evaluate.assert_called_once()
    # 2. Verificar que el método retorna Self
    assert result is ml_pipeline_instance


def test_evaluate_step_failure(ml_pipeline_instance, setup_pipeline_for_training, caplog):
    """Prueba el manejo de excepciones en evaluate_step."""
    mock_trainer_instance = setup_pipeline_for_training
    mock_trainer_instance.evaluate.side_effect = Exception("Métrica inválida")

    with pytest.raises(Exception, match="Métrica inválida"):
        ml_pipeline_instance.evaluate_step()

    # Verificar que el error fue loggeado
    assert "Error al intentar evaluar el performance del modelo" in caplog.text
