# Build, package, test, and clean
PROJECT=eql_source_layouts
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
NUMBATEST_ARGS=--doctest-modules -v --pyargs -m use_numba
LINT_FILES=setup.py $(PROJECT)
BLACK_FILES=setup.py $(PROJECT) notebooks
FLAKE8_FILES=setup.py $(PROJECT) notebooks

help:
	@echo "Commands:"
	@echo ""
	@echo "  test      run the test suite and report coverage"
	@echo "  format    run black to automatically format the code"
	@echo "  check     run code style and quality checks (black and flake8)"
	@echo "  lint      run pylint for a deeper (and slower) quality check"
	@echo "  clean     clean up build and generated files"
	@echo ""

test:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); MPLBACKEND='agg' pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

format:
	black $(BLACK_FILES)

check:
	black --check $(BLACK_FILES)
	flake8 $(FLAKE8_FILES)

lint:
	pylint --jobs=0 $(LINT_FILES)

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache
	rm -rvf $(TESTDIR) dask-worker-space
