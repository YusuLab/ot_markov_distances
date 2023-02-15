# -------------------------------------------------
#  Some boilerplate
#  ------------------------------------------------
#
PACKAGE=ot_markov_distances
RUN_IN_ENV=poetry run

ALL_SOURCE_FILES=$(shell find $(PACKAGE) -type f)
ALL_DOC_SOURCE_FILES=$(shell find staticdocs -type f)

# timestamp directory
.make:
	mkdir -p $@

all: .make/build-docs

run-tests: deps dev-deps 
	$(RUN_IN_ENV) pytest --cov=carsreco --cov-report xml

.make/deps: pyproject.toml | .make
	poetry install
	touch $@

.make/%-deps: pyproject.toml | .make
	poetry install --with $*
	touch $@

#-----------------------------------------------------------
# Documentation
# ---------------------------------------------------------

# publish-docs: docgen
# 	aws s3 cp "docs/build/html/" s3://$(S3_DOCS_BUCKET) --recursive

.make/build-docs: .make/docs-dir .make/autodoc docs/source/README.md .make/docs-deps | .make
	$(RUN_IN_ENV) $(MAKE) -C docs html
	touch $@

.make/docs-dir: $(ALL_DOC_SOURCE_FILES) | .make
	cp -RTf staticdocs docs
	touch $@

.make/autodoc: $(ALL_SOURCE_FILES) .make/docs-deps | .make
	$(RUN_IN_ENV) sphinx-apidoc -f -T -e -o  docs/source $(PACKAGE)
	touch $@

docs/source/README.md: .make/docs-dir ./README.md 
	cp ./README.md docs/source/README.md


# nbconvert: docs-dir dev-deps
# 	pipenv run jupyter-nbconvert --to rst tests/Process.ipynb  --output-dir ./docs/source
# 	sed -i "/^\s*INFO:/d" ./docs/source/Process.rst #remove the innumerable INFO: lines
#

#---------------------------------------------------------------
#Others
#---------------------------------------------------------------


# install-kernel: deps
# 	pipenv run python -m ipykernel install --user --name=ECE143-project-env

clean:
	rm -rf docs
	rm -rf .make


.PHONY: clean docs docs-dir autodoc nbconvert docgen local-deploy run-server
