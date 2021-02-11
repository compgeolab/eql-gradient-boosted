# Gradient-boosted equivalent sources

by
Santiago Soler
and Leonardo Uieda

A preprint version is available
[here]( https://raw.githubusercontent.com/compgeolab/eql-gradient-boosted/gh-pages/manuscript.pdf?token=ACYBWRNGVBJUURFPKHBJWZLAFZWA4)
<!-- Replace this link with the doi in figshare when ready -->

## About

We present the gradient-boosted equivalent sources: a new methodology for
interpolating very large datasets of gravity and magnetic observations even on
modest personal computers, without the high computer memory needs of the
classical equivalent sources technique.
This new method is inspired by the gradient-boosting technique, mainly used in machine learning solutions.

<!-- Include an abstract figure with caption -->

## Abstract

The equivalent source technique is a powerful and widely used method for
processing gravity and magnetic data.  Nevertheless, its major
drawback is the large computational cost in terms of processing time and
computer memory.
We present two techniques for reducing the computational cost of equivalent
source processing: block-averaging source locations and the
gradient-boosted equivalent source algorithm.
Through block-averaging, we reduce the number of source coefficients that
must be estimated while retaining the minimum desired resolution in the final
processed data.
With the gradient boosting method, we estimate the sources coefficients in
small batches along overlapping windows, allowing us to reduce the computer
memory requirements arbitrarily to conform to the constraints of the
available hardware.
We show that the combination of block-averaging and gradient-boosted
equivalent sources is capable of producing accurate interpolations through
tests against synthetic data.
Moreover, we demonstrate the feasibility of our method by gridding a gravity
dataset covering Australia with over 1.7 million observations using a modest
personal computer.


## Reproducing the results

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/compgeolab/eql-gradient-boosted

or [click here to download a zip archive](https://github.com/compgeolab/eql-gradient-boosted/archive/master.zip).

All source code used to generate the results and figures in the paper are in
the `notebooks` folder. There you can find the [Jupyter](https://jupyter.org/)
notebooks that performs all the calculations to generate all figures and
results presented in the paper.
Inside the `notebooks/boost_and_layouts` folder you can find the Python files
that define functions and classes that implement the new methodologies
introduced in the paper.

The sources for the manuscript text and figures are in `manuscript`.

See the `README.md` files in each directory for a full description.


### Setting up your environment

You'll need a working Python 3 environment with all the standard
scientific packages installed (numpy, pandas, scipy, matplotlib, etc).
The easiest (and recommended) way to get this is to download and install the
[Anaconda Python distribution](https://www.anaconda.com/).

Besides the standard scientific packages that come pre-installed with Anaconda,
you'll also need to install some extra libraries like: Numba for just-in-time
compilation; Harmonica, Verde, Boule and Pooch from the
[Fatiando a Terra](https://www.fatiando.org) project; Cartopy and PyGMT for
generating maps and more.

Instead of manually install all the dependencies, they can all be automatically
installed using a conda environment.

1. Change directory to the cloned git repository:
    ```
    cd eql-gradient-boosted
    ```
1. Create a new conda environment from the `environment.yml` file:
    ```
    conda env create -f environment.yml
    ```
1. Activate the new environment:
    ```
    conda activate eql-gradient-boosted
    ```

For more information about managing conda environments visit this
[User Guide](https://conda.io/docs/user-guide/tasks/manage-environments.html)

## License

All source code is made available under a BSD 3-clause license.  You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors.  See `LICENSE.md` for the full license text.

Data and the results of numerical tests are available under the
[Creative Commons Attribution 4.0 License (CC-BY)](https://creativecommons.org/licenses/by/4.0/).

The manuscript text and figures are not open source.
The authors reserve the rights to the article content.
<!-- , which has been accepted for publication in -->
<!-- Geophysical Journal International. -->

