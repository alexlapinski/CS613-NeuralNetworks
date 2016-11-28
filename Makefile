init:
	pip install -r requirements.txt

build: init
	python setup.py sdist

part1:
	python -m cs613_hw4.main --binary-ann

part2:
	python -m cs613_hw4.main --precision-recall

part3:
	python -m cs613_hw4.main --multi-ann

test:
	pytest tests

.PHONY: init test
