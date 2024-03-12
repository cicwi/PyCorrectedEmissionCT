.PHONY: clean clean-test clean-pyc clean-build docs help install_dev
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 corrct tests examples --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 corrct tests examples --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source corrct setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs/.nojekyll:
	mkdir -p docs
	touch docs/.nojekyll

docs_env:
	conda install configparser dataclasses -y
	conda install sphinx_rtd_theme commonmark coverage pycodestyle pyflakes recommonmark -y

docs: docs/.nojekyll install_dev ## generate Sphinx HTML documentation, including API docs
	rm -f doc_sources/corrct.rst
	rm -f doc_sources/modules.rst
	sphinx-apidoc -o doc_sources/ corrct
	make -C doc_sources clean
	make -C doc_sources html
	$(BROWSER) docs/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C doc_sources html' -R -D .

install: clean ## install the package to the active Python's site-packages
	python setup.py install

install_dev:
	# https://stackoverflow.com/a/28842733
	pip install -e .[dev]

prepare_packages:
	conda install conda-build setuptools wheel twine build -y

conda_package:
	conda build conda/

pip_package:
	# python setup.py clean --all
	# python setup.py sdist bdist_wheel
	python -m build
