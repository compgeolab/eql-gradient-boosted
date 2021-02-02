# The name of the main .tex file to build.
PROJECT = manuscript
GJI = manuscript-gji
# List other .tex needed files
TEX_FILES = variables.tex content.tex appendix.tex keywords.tex abstract.tex references.bib
RESULTS = results/australia.tex results/best-parameters-airborne-survey.tex results/best-parameters-ground-survey.tex results/boost-overlapping.tex results/parameters-airborne-survey.tex results/parameters-ground-survey.tex results/source-layouts-schematics.tex results/synthetic-surveys.tex
# Folder where output will be placed
OUTDIR = _output
# Name of the output file
OUTPUT = $(OUTDIR)/$(PROJECT).pdf
OUTPUT_GJI = $(OUTDIR)/$(GJI).pdf

LATEX_COMPILER = latexmk
LATEX_FLAGS = -pdf -outdir=$(OUTDIR)
PDFVIEWER = xdg-open


all: $(OUTPUT)

gji: $(OUTPUT_GJI)

clean:
	rm -rf $(OUTDIR)

show: all
	@( $(PDFVIEWER) $(OUTPUT) 2> /dev/null; )

show-gji: gji
	@( $(PDFVIEWER) $(OUTPUT_GJI) 2> /dev/null; )

$(OUTPUT): $(PROJECT).tex $(TEX_FILES) $(RESULTS)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<

$(OUTPUT_GJI): $(GJI).tex $(TEX_FILES) $(RESULTS)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<
