## hw1
In this hw, we do some exploratory data analysis.

### Part 1 – State College weather station data
We show the importance of removing missing and trace data from calculations of summary statistics and schematic plots 
(fancy box plots). 

[Interact with the notebook via binder](https://mybinder.org/v2/gh/zmoon92/PSU-METEO-515/master?filepath=hw1%2Fhw1p1.ipynb).

Or view the notebook here on GH.

### Part 2 – NAO and AMO
We investigate the NAO (North Atlantic Oscillation) 
and AMO (Atlantic Multidecadal Oscillation) through:

* summary statistics
* histograms and kernel density estimates (KDEs) at various bin- and band-widths. 

[Interact with the notebook via binder](https://mybinder.org/v2/gh/zmoon92/PSU-METEO-515/master?filepath=hw1%2Fhw1p2.ipynb).

Or view the notebook here on GH. 

**Note** that in the assignment description it says to use the time series of annual means for this part, but I did not do this. This allows for a larger N for the histograms. 

### KDE/histogram interactive plot using Bokeh
There are two working versions of this inside subdirectory ['kde-slider'](./kde-slider): 

* [Bokeh app version](./kde-slider/bokeh_py-cb): requires a running Bokeh server (`bokeh serve --show kde-slider_bokeh.py` to start the app
* [JS callback version](./kde-slider/bokeh_js-cb): uses a LUT and JS callbacks (as opposed to Python), so can be run in the web browser. 
