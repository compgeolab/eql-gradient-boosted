# Check styling and test notebooks and extra code
NOTEBOOKS=notebooks
RESULTS=results
MANUSCRIPT=manuscript
FIGS=$(MANUSCRIPT)/figs
PYTEST_ARGS=--cov --cov-report=term-missing --cov -v
LINT_FILES=$(NOTEBOOKS)
BLACK_FILES=$(NOTEBOOKS)
FLAKE8_FILES=$(NOTEBOOKS)
JUPYTEXT=jupytext --execute --set-kernel - --to notebook

help:
	@echo "Commands:"
	@echo ""
	@echo "  run       run every notebook on the repository"
	@echo "  test      run the test suite and report coverage"
	@echo "  format    run black to automatically format the code"
	@echo "  check     run code style and quality checks (black and flake8)"
	@echo "  lint      run pylint for a deeper (and slower) quality check"
	@echo "  clean     clean up build and generated files"
	@echo "  sync      make jupytext to sync notebooks and scripts"
	@echo ""


sync:
	jupytext --to notebook --set-kernel - $(NOTEBOOKS)/*.py

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


SYNTH_SURVEYS_AND_TARGET_GRID = \
	$(RESULTS)/ground_survey/survey.csv \
	$(RESULTS)/airborne_survey/survey.csv \
	$(RESULTS)/target.nc \
	$(RESULTS)/synthetic-surveys.json

GRID_GROUND_RESULTS = \
	$(RESULTS)/ground_survey/parameters.json \
	$(RESULTS)/ground_survey/best_predictions-block_averaged_sources.nc \
	$(RESULTS)/ground_survey/best_predictions-grid_sources.nc \
	$(RESULTS)/ground_survey/best_predictions-source_below_data.nc

GRID_AIRBORNE_RESULTS = \
	$(RESULTS)/airborne_survey/parameters.json \
	$(RESULTS)/airborne_survey/best_predictions-block_averaged_sources.nc \
	$(RESULTS)/airborne_survey/best_predictions-grid_sources.nc \
	$(RESULTS)/airborne_survey/best_predictions-source_below_data.nc

FIGURES = $(FIGS)/airborne-survey.pdf \
	$(FIGS)/airborne_survey_differences.pdf \
	$(FIGS)/ground_survey_differences.pdf \
	$(FIGS)/ground-survey.pdf \
	$(FIGS)/target-grid.pdf

LATEX_VARIABLES = \
	$(MANUSCRIPT)/best-parameters-airborne-survey.tex \
	$(MANUSCRIPT)/best-parameters-ground-survey.tex \
	$(MANUSCRIPT)/parameters-airborne-survey.tex \
	$(MANUSCRIPT)/parameters-ground-survey.tex \
	$(MANUSCRIPT)/source-layouts-schematics.tex \
	$(MANUSCRIPT)/synthetic-surveys.tex



run: synthetic grid_ground_survey grid_airborne_survey latex_variables figures

synthetic: $(SYNTH_SURVEYS_AND_TARGET_GRID)

figures: $(FIGURES) \
	$(FIGS)/depth_types.pdf \
	$(FIGS)/block-averaged-sources-schematics.pdf \
	$(FIGS)/source-layouts-schematics.pdf

grid_ground_survey: $(GRID_GROUND_RESULTS)

grid_airborne_survey: $(GRID_AIRBORNE_RESULTS)

latex_variables: $(LATEX_VARIABLES)


$(SYNTH_SURVEYS_AND_TARGET_GRID): $(NOTEBOOKS)/??-synthetic-surveys.py
	$(JUPYTEXT) $<
	touch $(SYNTH_SURVEYS_AND_TARGET_GRID)

$(RESULTS)/source-layouts-schematics.json $(FIGS)/source-layouts-schematics.pdf: \
	$(NOTEBOOKS)/??-source-layouts-schematics.py $(RESULTS)/ground_survey/survey.csv
	$(JUPYTEXT) $<
	touch $(RESULTS)/source-layouts-schematics.json $(FIGS)/source-layouts-schematics.pdf

$(GRID_GROUND_RESULTS): \
	$(NOTEBOOKS)/??-grid-ground-data.py $(RESULTS)/ground_survey/survey.csv $(RESULTS)/target.nc
	$(JUPYTEXT) $<
	touch $(GRID_GROUND_RESULTS)

$(GRID_AIRBORNE_RESULTS): \
	$(NOTEBOOKS)/??-grid-airborne-data.py $(RESULTS)/airborne_survey/survey.csv $(RESULTS)/target.nc
	$(JUPYTEXT) $<
	touch $(GRID_AIRBORNE_RESULTS)

$(FIGURES): $(NOTEBOOKS)/??-figures.py
	$(JUPYTEXT) $<
	touch $(FIGURES)

$(FIGS)/depth_types.pdf: $(NOTEBOOKS)/??-depth-types-schematics.py data/survey_1d.csv
	$(JUPYTEXT) $<
	touch $@

$(FIGS)/block-averaged-sources-schematics.pdf: $(NOTEBOOKS)/??-block-averaged-schematics.py
	$(JUPYTEXT) $<
	touch $@

$(LATEX_VARIABLES): $(NOTEBOOKS)/??-generate_latex_variables.py \
	$(RESULTS)/synthetic-surveys.json \
	$(RESULTS)/ground_survey/parameters.json \
	$(RESULTS)/airborne_survey/parameters.json \
	$(RESULTS)/source-layouts-schematics.json
	$(JUPYTEXT) $<
	touch $(LATEX_VARIABLES)
