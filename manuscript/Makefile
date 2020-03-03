# The name of the main .tex file to build.
PROJECT = manuscript
# List other .tex needed files
TEX_FILES= synthetic_model_surveys.tex parameters_ground_survey.tex parameters_airborne_survey.tex best_parameters_ground_survey.tex best_parameters_airborne_survey.tex
# Folder where output will be placed
OUTDIR = _output
# Name of the output file
OUTPUT = $(OUTDIR)/$(PROJECT).pdf

LATEX_COMPILER = latexmk
LATEX_FLAGS = -pdf -outdir=$(OUTDIR)
PDFVIEWER = xdg-open


all: $(OUTPUT)

clean:
	rm -rf $(OUTDIR)

show: all
	@( $(PDFVIEWER) $(OUTPUT) 2> /dev/null; )

$(OUTPUT): $(PROJECT).tex $(TEX_FILES)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<
