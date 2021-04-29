OUTDIR = _output
SLIDES_DIR = slides
SLIDES_PDF = $(OUTDIR)/slide1.pdf $(OUTDIR)/slide2.pdf $(OUTDIR)/slide3a.pdf $(OUTDIR)/slide3b.pdf $(OUTDIR)/slide4a.pdf $(OUTDIR)/slide4b.pdf $(OUTDIR)/slide4c.pdf $(OUTDIR)/slide5.pdf $(OUTDIR)/slide6.pdf $(OUTDIR)/slide7.pdf $(OUTDIR)/slide8.pdf
PRESENTATION_PDF = slides.pdf

PDF_VIEWER = xdg-open
INKSCAPE = inkscape
INKSCAPE_ARGS = --export-area-drawing --without-gui --export-type=pdf
PDFTK = pdftk
PDFTK_ARGS = cat output

all: $(PRESENTATION_PDF)

clean:
	rm -rf $(OUTDIR)
	rm $(PRESENTATION_PDF)

show: $(PRESENTATION_PDF)
	$(PDF_VIEWER) $<

$(PRESENTATION_PDF): $(SLIDES_PDF)
	$(PDFTK) $(SLIDES_PDF) $(PDFTK_ARGS) $@

$(OUTDIR):
	mkdir -p $@

$(OUTDIR)/%.pdf: $(SLIDES_DIR)/%.svg | $(OUTDIR)
	$(INKSCAPE) $< $(INKSCAPE_ARGS) --export-filename=$@
