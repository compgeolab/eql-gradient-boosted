# LaTeX sources for the manuscript

This folder contains all the files needed to build two PDF versions of the
manuscript using LaTeX: one for the preprint and another one for submitting
*Geophysical Journal International*.
The two main LaTeX documents are:

- `preprint.tex`: for building the preprint
- `gji.tex`: for building the manuscript using the GJI template

These two files only contain a header and some configurations, but the content
of the article is split in separate files, which are then imported directly
into `preprint.tex` and `gji.tex`.
The extra files are:

- `abstract.tex`: contains the text for the abstract
- `content.tex`: contains the body of the paper, with sections, figures,
    equations, etc.
- `appendix.tex`: contains the appendix.
- `variables.tex`: defines some variables that contain additional information
    like the title of the manuscript, information about the authors, keywords
    and more.

The information about the references can be found inside `references.bib`,
while the bibliography styles used for the preprint and the submission
manuscript are inside `apalike-doi.bst` and `gji.bst`, respectively.

This folder also contains the files provided by GJI to build the manuscript
using their template (`gji.cls`, `gji_extra.sty` and `times.sty`).

The figures of the article can be found under the `figs` directory.
They are automatically created by the Jupyter notebooks inside the `notebooks`
folder.
These notebooks also store some numerical values that we include in the
manuscript as LaTeX variables defined in files under the `results` folder.
They are also automatically created by the notebooks.

## How to build the manuscript

In order to build the manuscript, you need to install a LaTeX distribution and
the `latexmk` tool.

The `Makefile` has rules for building the PDF of the manuscript.

To build the preprint PDF, run:

```
make preprint
```

To build the PDF of the submission manuscript, run:

```
make gji
```

You can also ask for a word and figures count using:

```
make word-count
```
