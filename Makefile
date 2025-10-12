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
 
# ============================================================
#CONFIGURACIÓN DE ENTORNO
# ============================================================
 
# Crear entorno virtual e instalar dependencias

create_environment:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -r requirements.txt
	
# Instalar dependencias directamente

install:
	pip install -r requirements.txt
 
# Activar entorno virtual (solo muestra mensaje)

activate:
	@echo "Para activar el entorno: source $(VENV)/bin/activate (Linux/Mac)"
	@echo "o .venv\\Scripts\\activate (Windows)"
 
# ============================================================
# LIMPIEZA
# ============================================================
 
clean:
	rm -rf $(INTERIM_DIR)/*
	rm -rf $(PROCESSED_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(REPORTS_DIR)/*
	rm -rf $(FIGURES_DIR)/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
 
# ============================================================
# DATOS
# ============================================================
# Generar o descargar datos

data:
	$(PYTHON) $(SRC_DIR)/dataset.py
 
# Crear features
features:
	$(PYTHON) $(SRC_DIR)/features.py
 
# ============================================================
# MODELOS
# ============================================================
# Entrenar modelos
train:
	$(PYTHON) $(SRC_DIR)/modeling/train.py
 
# Predecir con modelos entrenados

predict:
	$(PYTHON) $(SRC_DIR)/modeling/predict.py
 
# ============================================================
# REPORTES Y FIGURAS
# ============================================================
 
# Generar figuras (plots)
plots:
	$(PYTHON) $(SRC_DIR)/plots.py

# Generar notebooks automáticamente
notebooks:

	jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/*.ipynb
 
# Generar reportes finales (opcional)
report:
	@echo "Generando reportes en $(REPORTS_DIR)..."
	# aquí podrías agregar comandos para generar reportes automáticamente
 
# ============================================================
# QA / LINTERS
# ============================================================
 
lint:
	flake8 $(SRC_DIR)
 
# ============================================================
# AYUDA
# ============================================================
help:
	@echo "Comandos disponibles:"
	@echo "  make create_environment   -> Crear entorno virtual e instalar dependencias"
	@echo "  make install              -> Instalar dependencias"
	@echo "  make clean                -> Limpiar archivos temporales y modelos"
	@echo "  make data                 -> Procesar/descargar datos"
	@echo "  make features             -> Generar features"
	@echo "  make train                -> Entrenar modelos"
	@echo "  make predict              -> Predecir con modelos"
	@echo "  make plots                -> Generar figuras"
	@echo "  make notebooks            -> Ejecutar notebooks"
