init:
	pip install -r requirements.txt

build: init
	python setup.py sdist

test:
	nosetests tests

.PHONY: init test
