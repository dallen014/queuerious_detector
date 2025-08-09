#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = queuerious_detector
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m unittest discover -s tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
	$(PYTHON_INTERPRETER) -m queuerious_detector.dataset

## Create TF-IDF features
.PHONY: features-lg
features-lg:
	$(PYTHON_INTERPRETER) -m queuerious_detector.features --model lg

## Create SBERT features
.PHONY: features-sbert
features-sbert:
	$(PYTHON_INTERPRETER) -m queuerious_detector.features --model sbert

## Train model (lg | rf | svm | xgb |all)
.PHONY: train-%
train-%:
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.train --model $*

## Predict & evaluate model (lg | rf | svm | xgb |all)
.PHONY: predict-%
predict-%:
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.predict --model $*

#################################################################################
# FULL PIPELINES                                                                     #
#################################################################################

## Full TF-IDF + Logistic Regression pipeline
.PHONY: pipeline-lg
pipeline-lg: data features-lg
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.train --model lg
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.predict --model lg

## Full SBERT + RF/SVM/XGB models pipeline
.PHONY: pipeline-sbert-all
pipeline-sbert-all: data features-sbert
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.train --model rf
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.predict --model rf
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.train --model svm
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.predict --model svm
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.train --model xgb
	$(PYTHON_INTERPRETER) -m queuerious_detector.modeling.predict --model xgb

## Run all models pipeline
.PHONY: pipeline-all
pipeline-all: pipeline-lg pipeline-sbert-all

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_%\-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
