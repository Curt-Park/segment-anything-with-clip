EFFICIENTVIT_SAM_URL := "https://huggingface.co/han-cai/efficientvit-sam/resolve/main"
EFFICIENTVIT_SAM_MODEL := "xl1.pt"

BASENAME=$(shell basename $(CURDIR))
PYTHON=3.10


define download
	@if [ ! -f $(2) ]; then \
		echo "Download $(2)..."; \
		wget "$(1)/$(2)"; \
	fi
endef


env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	pip install -r requirements.txt

model:
	$(call download,$(EFFICIENTVIT_SAM_URL),$(EFFICIENTVIT_SAM_MODEL))

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
