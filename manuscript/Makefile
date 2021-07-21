# The name of the main .tex file to build.
PREPRINT = preprint
GJI = gji
# List other .tex needed files
TEX_FILES = variables.tex content.tex appendix.tex abstract.tex references.bib
RESULTS = results/australia.tex results/best-parameters-airborne-survey.tex results/best-parameters-ground-survey.tex results/boost-overlapping.tex results/parameters-airborne-survey.tex results/parameters-ground-survey.tex results/source-layouts-schematics.tex results/synthetic-surveys.tex
# Folder where output will be placed
OUTDIR = _output
# Name of the output file
PREPRINT_PDF = $(OUTDIR)/$(PREPRINT).pdf
GJI_PDF = $(OUTDIR)/$(GJI).pdf

# Latex compiler and pdfviewer
LATEX_COMPILER = latexmk
LATEX_FLAGS = -pdf -outdir=$(OUTDIR)
PDFVIEWER = xdg-open

# Variables for generating the diffs against older versions
SUBMITTED_V1 = submitted-v1
DIFFS_SUBMITTED_V1 = gji-diffsubmitted-v1
DIFFS_SUBMITTED_V1_PDF = $(OUTDIR)/$(DIFFS_SUBMITTED_V1).pdf
LATEXDIFF_VC = latexdiff-vc
LATEXDIFF_VC_FLAGS = --git --flatten --disable-citation-markup


all: preprint gji

preprint: $(PREPRINT_PDF)

gji: $(GJI_PDF)

clean:
	rm -rf $(OUTDIR) *diffs*.tex

word-count:
	texcount -merge $(PREPRINT).tex
	texcount -merge $(GJI).tex

show-preprint: preprint
	@( $(PDFVIEWER) $(PREPRINT_PDF) 2> /dev/null; )

show-gji: gji
	@( $(PDFVIEWER) $(GJI_PDF) 2> /dev/null; )

$(PREPRINT_PDF): $(PREPRINT).tex $(TEX_FILES) $(RESULTS)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<

$(GJI_PDF): $(GJI).tex $(TEX_FILES) $(RESULTS)
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<


# diffs
# -----

show-diffs-submitted-v1: diffs-submitted-v1
	@( $(PDFVIEWER) $(DIFFS_SUBMITTED_V1_PDF) 2> /dev/null; )

diffs-submitted-v1: $(DIFFS_SUBMITTED_V1_PDF)

$(DIFFS_SUBMITTED_V1_PDF): $(DIFFS_SUBMITTED_V1).tex
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<

$(DIFFS_SUBMITTED_V1).tex: $(GJI).tex
	$(LATEXDIFF_VC) $(LATEXDIFF_VC_FLAGS) -r $(SUBMITTED_V1) $<
