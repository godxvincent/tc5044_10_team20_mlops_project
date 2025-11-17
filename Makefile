# ============================================================
# VARIABLES
# ============================================================

PYTHON := python
VENV := .venv
DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
INTERIM_DIR := $(DATA_DIR)/interim
PROCESSED_DIR := $(DATA_DIR)/processed
MODELS_DIR := models
NOTEBOOKS_DIR := notebooks
REPORTS_DIR := reports
FIGURES_DIR := figures
SRC_DIR := mlops
PYTHONPATH=$(pwd)
ENV_FOR_DYNACONF=local

# ============================================================
# CONFIGURACIÓN DE ENTORNO
# ============================================================

## Crear entorno virtual e instalar dependencias
create_environment:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -r requirements.txt

## Instalar dependencias directamente
install:
	pip install -r requirements.txt

## Mostrar instrucciones para activar el entorno virtual
activate:
	@echo "Para activar el entorno: source $(VENV)/bin/activate (Linux/Mac)"
	@echo "o .venv\\Scripts\\activate (Windows)"

# ============================================================
# LIMPIEZA
# ============================================================

## Limpiar archivos temporales y modelos
clean:
	rm -rf $(INTERIM_DIR)/*
	rm -rf $(PROCESSED_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(REPORTS_DIR)/*
	rm -rf $(FIGURES_DIR)/*
	find . -type d -name "__pycache__" -exec rm -rf {} +

## Limpieza adicional de artefactos de Python / build
clean-pyc:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

## Limpieza de archivos de compilación o build
clean-build:
	rm -rf build/ dist/ *.egg-info

# ============================================================
# DATOS
# ============================================================

## Generar o descargar datos
data:
	$(PYTHON) $(SRC_DIR)/dataset.py

## Crear features
features:
	$(PYTHON) $(SRC_DIR)/features.py

# ============================================================
# MODELOS
# ============================================================

## Entrenar modelos
train:
	$(PYTHON) $(SRC_DIR)/modeling/train.py

## Predecir con modelos entrenados
predict:
	$(PYTHON) $(SRC_DIR)/modeling/predict.py

# ============================================================
# REPORTES Y FIGURAS
# ============================================================

## Generar figuras (plots)
plots:
	$(PYTHON) $(SRC_DIR)/plots.py

## Ejecutar notebooks automáticamente
notebooks:
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/*.ipynb

## Generar reportes finales (opcional)
report:
	@echo "Generando reportes en $(REPORTS_DIR)..."
	# aquí podrías agregar comandos para generar reportes automáticamente

# ============================================================
# QA / LINTERS
# ============================================================

## Analizar código fuente con Flake8
lint:
	flake8 $(SRC_DIR)

## Formatear código con Black + ordenar imports con isort
format:
	black --line-length=120 .
	isort --profile=black --line-length=120 .

## Verificar estilo con Black, isort y Flake8 (sin modificar)
lint-all:
	black --check --line-length=120 .
	isort --check-only --profile=black --line-length=120 .
	flake8 .

## Generar reportes TXT (Black/Flake8) en tools/lint_reports/
lint-report:
	$(PYTHON) tools/run_linters_and_save_reports.py --check --verbose

## Instalar hooks de pre-commit
precommit-install:
	pip install pre-commit
	pre-commit install

## Ejecutar manualmente los hooks de pre-commit
precommit-run:
	pre-commit run --all-files

## Limpiar metadatos de notebooks (nbstripout)
nb-clean:
	pip install nbstripout
	nbstripout --install
	nbstripout $$(git ls-files "*.ipynb")
	
## Ejecuta el mismo lint que corre en CI (GitHub Actions)
ci-lint:
	@echo "==> Black --check"
	black --check --line-length=120 .
	@echo "==> isort --check-only"
	isort --check-only --profile=black --line-length=120 .
	@echo "==> Flake8"
	flake8 
	.PHONY: ci-lint
# ============================================================
# TESTS
# ============================================================

## Ejecutar pruebas unitarias
test:
	pytest ./tests

itest:
	pytest ./integration_tests


# ============================================================
# PLOTS
# ============================================================
plots:
	$(PYTHON) $(SRC_DIR)/plots.py
	
# ============================================================
# PLOTS
# ============================================================
plots:
	$(PYTHON) $(SRC_DIR)/plots.py
	
# ============================================================
# OTROS
# ============================================================

## Cargar variables del proyecto en PYTHONPATH
load_pp:
	export PYTHONPATH=$(pwd)

# ============================================================
# HELP AUTOMÁTICO
# ============================================================

# Este bloque permite generar ayuda automáticamente a partir de las descripciones (##)
# Uso: `make help`
# Muestra todos los targets con su descripción alineada.
help: ## Mostrar esta ayuda con la lista de comandos
	@echo ""
	@echo "Comandos disponibles:"
	@echo "---------------------------------------------------------------"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z0-9_.-]+:.*?##/ {printf "  %-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo "---------------------------------------------------------------"
	@echo ""
