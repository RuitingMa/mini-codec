# Build the technical report PDF.
#
# Requires a working LaTeX install (TeXLive, MiKTeX, MacTeX) with
# pdflatex + bibtex on PATH. No external .sty files needed.
#
# Targets:
#   make            -> build main.pdf (default)
#   make clean      -> remove build artefacts (keeps main.pdf)
#   make distclean  -> remove main.pdf as well

MAIN = main

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib figures/*.png
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex   $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf
