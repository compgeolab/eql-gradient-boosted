# The name of the main .tex file to build.
PREPRINT = preprint
GJI = gji
# List other .tex needed files
TEX_FILES = variables.tex content.tex appendix.tex keywords.tex abstract.tex references.bib
RESULTS = results/australia.tex results/best-parameters-airborne-survey.tex results/best-parameters-ground-survey.tex results/boost-overlapping.tex results/parameters-airborne-survey.tex results/parameters-ground-survey.tex results/source-layouts-schematics.tex results/synthetic-surveys.tex
# Folder where output will be placed
OUTDIR = _output
# Name of the output file
PREPRINT_PDF = $(OUTDIR)/$(PREPRINT).pdf
GJI_PDF = $(OUTDIR)/$(GJI).pdf

LATEX_COMPILER = latexmk
LATEX_FLAGS = -pdf -outdir=$(OUTDIR)
PDFVIEWER = xdg-open


all: preprint gji

preprint: $(PREPRINT_PDF)

gji: $(GJI_PDF)

clean:
	rm -rf $(OUTDIR)

show-preprint: preprint
	@( $(PDFVIEWER) $(PREPRINT_PDF) 2> /dev/null; )

show-gji: gji
	@( $(PDFVIEWER) $(GJI_PDF) 2> /dev/null; )

$(PREPRINT_PDF): $(PREPRINT).tex $(TEX_FILES) $(RESULTS)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<

$(GJI_PDF): $(GJI).tex $(TEX_FILES) $(RESULTS)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<
