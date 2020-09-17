# Check styling and test notebooks and extra code
NOTEBOOKS=notebooks
PYTEST_ARGS=--cov --cov-report=term-missing --cov -v
LINT_FILES=$(NOTEBOOKS)
BLACK_FILES=$(NOTEBOOKS)
FLAKE8_FILES=$(NOTEBOOKS)

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
	MPLBACKEND='agg' pytest $(PYTEST_ARGS) $(NOTEBOOKS)

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
	rm -rvf build dist *.egg-info __pycache__ .coverage .cache .pytest_cache
	rm -rvf dask-worker-space
