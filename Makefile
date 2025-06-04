clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-git set-dev
test: set-test

set-git:
	git config --local commit.template .gitmessage

set-style-dep:
	pip3 install ruff

set-precommit:
	pip3 install pre-commit
	pre-commit install

set-dev:
	pip3 install -r requirements.txt

set-test:
	python3 -m unittest tests/

set-style:
	ruff check --fix .
	ruff format .

check-quality:
	ruff check .
	ruff format --check .

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
