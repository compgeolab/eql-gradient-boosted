# A better strategy for interpolating gravity and magnetic data

[Santiago Soler](https://santisoler.github.io/)<sup>1,2</sup>
and
[Leonardo Uieda](https://www.leouieda.com/)<sup>3</sup>

> <sup>1</sup>CONICET, Argentina<br>
> <sup>2</sup>Instituto Geofísico Sismológico Volponi, Universidad Nacional de San Juan, Argentina<br>
> <sup>3</sup>Department of Earth, Ocean and Ecological Sciences, School of Environmental Sciences, University of Liverpool, UK<br>

Presented at the EGU2020 General Assembly.

|        |Info|
|-------:|:---|
|Session |[G4.3: Acquisition and processing of gravity and magnetic field data and their integrative interpretation](https://meetingorganizer.copernicus.org/EGU2020/session/35332)|
|Abstract|doi:[10.5194/egusphere-egu2020-549](https://doi.org/10.5194/egusphere-egu2020-549)|
|Slides  |doi:[10.6084/m9.figshare.12217973](https://doi.org/10.6084/m9.figshare.12217973)|

The figures have been created by running `egu2020_figures.py`,
except for `draws.svg` which was created using Inkscape.

## Abstract

We present a new strategy for gravity and magnetic data interpolation and
processing. Our method is based on the equivalent layer technique (EQL) and
produces more accurate interpolations when compared with similar EQL methods.
It also reduces the computation time and memory requirements, both of which
have been severe limiting factors.

The equivalent layer technique (also known as equivalent source, radial basis
functions, or Green’s functions interpolation) is used to predict the value of
gravity and magnetic fields (or transformations thereof) at any point based on
the data gathered on some observation points. It consists in estimating a
source distribution that produces the same field as the one measured and using
this estimate to predict new values. It generally outperforms other
general-purpose 2D interpolators, like the minimum curvature or bi-harmonic
splines, because it takes into account the height of measurements and the fact
that these fields are harmonic functions. Nevertheless, defining a layout for
the source distribution used by the EQL is not trivial and plays an important
role in the quality of the predictions.

The most widely used source distributions are: (a) a regular grid of point
sources and (b) one point source beneath each observation point. We propose a
new source distribution: (c) divide the area into blocks, calculate the average
location of observation points inside each block, and place one point source
beneath each average location. This produces a smaller number of point sources
in comparison with the other source distributions, effectively reducing the
computational load. Traditionally, the source points are located: (i) all at
the same depth or (ii) each source point at a constant relative depth beneath
its corresponding observation point. Besides these two, we also considered
(iii) a variable relative depth for each source point proportional to the
median distance to its nearest neighbours. The combination of source
distributions and depth configurations leads to seven different source layouts
(the regular grid is only compatible with the constant depth configuration).

We have scored the performance of each configuration by interpolating synthetic
ground and airborne gravity data, and comparing the interpolation against the
true values of the model. The block-averaged source layout (c) with variable
relative depth (iii) produces more accurate interpolation results (R² of 0.97
versus R² of 0.63 for the traditional grid layout) in less time than the
alternatives (from 2 to 10 times faster on our test cases). These results are
consistent between ground and airborne survey layouts. Our conclusions can be
extrapolated to other applications of equivalent layers, such as upward
continuation, reduction-to-the-pole, and derivative calculation. What is more,
we expect that these optimizations can benefit similar spatial prediction
problems beyond gravity and magnetic data.

The source code developed for this study is based on the EQL implementation
available in Harmonica (fatiando.org/harmonica), an open-source Python library
for modelling and processing gravity and magnetic data.

## Citation

Please cite this abstract as:

> Soler, S. R. and Uieda, L.: A better strategy for interpolating gravity and
> magnetic data, EGU General Assembly 2020, Online, 4–8 May 2020, EGU2020-549,
> https://doi.org/10.5194/egusphere-egu2020-549, 2019

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img
alt="Creative Commons License" style="border-width:0"
src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br>
The slides and abstract are licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
