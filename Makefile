init:
	pip install -r requirements.txt

build: init
	python setup.py sdist

test:
	pytest tests

.PHONY: init test
