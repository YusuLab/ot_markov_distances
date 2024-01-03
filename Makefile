# -------------------------------------------------
#  Some boilerplate
#  ------------------------------------------------
#
PACKAGE=ot_markov_distances
RUN_IN_ENV=poetry run

ALL_SOURCE_FILES=$(shell find $(PACKAGE) -type f)
ALL_DOC_SOURCE_FILES=$(shell find staticdocs -type f)

README=$(shell ls -1 README.* | head -n 1 | tr -d '\n')

# timestamp directory
.make:
	mkdir -p $@

all: .make/build-docs .make/run-tests

run-tests: .make/run-tests

doc: .make/build-docs

.make/run-tests coverage.xml: .make/deps .make/test-deps |.make
	$(RUN_IN_ENV) pytest --cov=$(PACKAGE) --cov-report xml

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

.make/build-docs: .make/docs-dir .make/autodoc docs/source/$(README) .make/docs-deps | .make
	$(RUN_IN_ENV) $(MAKE) -C docs html
	touch $@

.make/docs-dir: $(ALL_DOC_SOURCE_FILES) | .make
	cp -RTf staticdocs docs
	touch $@

.make/autodoc: $(ALL_SOURCE_FILES) .make/docs-deps | .make
	$(RUN_IN_ENV) sphinx-apidoc -f -T -e -o  docs/source $(PACKAGE)
	touch $@

docs/source/$(README): .make/docs-dir ./$(README) 
	cp ./$(README) docs/source/$(README)


# nbconvert: docs-dir dev-deps
# 	pipenv run jupyter-nbconvert --to rst tests/Process.ipynb  --output-dir ./docs/source
# 	sed -i "/^\s*INFO:/d" ./docs/source/Process.rst #remove the innumerable INFO: lines
#

#---------------------------------------------------------------
#Others
#---------------------------------------------------------------

jupyter-kernel: .make/jupyter-kernel

.make/jupyter-kernel: .make/dev-deps | .make
	$(RUN_IN_ENV) python -m ipykernel install --user --name=ot_markov_distances-env
	touch $@

clean:
	rm -rf docs
	rm -rf .make


.PHONY: clean run-tests doc
