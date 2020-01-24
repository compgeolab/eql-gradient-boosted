# All tex files in the working directory
TEX_FILES = $(shell find . -maxdepth 1 -name '*.tex')

# Output directory name
OUTPUT = _output

# Output PDF files
PDFS = $(addprefix $(OUTPUT)/, $(TEX_FILES:.tex=.pdf))

LATEX_COMPILER = latexmk
LATEX_FLAGS = -pdf -outdir=$(OUTPUT)
PDFVIEWER = xdg-open


all: $(PDFS)

clean:
	rm -rf $(OUTPUT)

show: all
	@ ( $(foreach PDF,$(PDFS),$(PDFVIEWER) $(PDF) 2> /dev/null;) )

$(OUTPUT)/%.pdf: %.tex | $(OUTPUT)/
	$(LATEX_COMPILER) $(LATEX_FLAGS) $<

$(OUTPUT)/:
	mkdir $(OUTPUT)
