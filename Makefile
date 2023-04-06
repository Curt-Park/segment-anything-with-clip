PYTHON=3.8
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	pip install -r requirements.txt

run:
	gradio app.py

setup-dev:
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black .
	isort .

lint:
	flake8 .
