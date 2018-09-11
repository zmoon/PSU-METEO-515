#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:24:08 2018

@author: zmoon
"""

from __future__ import division
#from collections import OrderedDict
import datetime as dt

from bokeh.embed import file_html
from bokeh.io import curdoc
from bokeh.layouts import layout, row, column, widgetbox
from bokeh.models import CustomJS, Slider#, ColumnDataSource, Select
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.resources import CDN
#import matplotlib.dates as mdates
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import scipy.stats as ss
#from sklearn.neighbors import KernelDensity
#import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, _plot_corr
from statsmodels.nonparametric.kde import KDEUnivariate
#from statsmodels.robust.scale import mad

#plt.close('all')


#%% load LUTs


x_kde = np.linspace(-3.9, 3.9, 400)  # should load from file...
dbw = 0.002
bws = np.arange(0.01, 2.0+dbw, dbw)
kdes = np.loadtxt('lut_kde.csv', delimiter=',')




#%% bokeh

hist_color = '#557ef6'  # '#6e90f7', '#557ef6', '#3d6bf5', 'blue'
kde_color =  '#ea4800'  # '#ea4800', '#ea3c00', 'orange'

#> create bokeh figure
f = figure(plot_width=750, plot_height=450, 
           title='KDE for NAO',
           x_axis_label='twice-normalized NAO index', 
           y_axis_label='density',
           x_range=(-4, 4), y_range=(0, 0.5),
           )
f.grid.minor_grid_line_color = '#eeeeee'


#> initial data
bw0 = 0.07

#counts_hist, x_hist = calcHist(bw0)
#source_hist = ColumnDataSource({'x': x_hist, 'counts': counts_hist, 'bw': bw0*np.ones(x_hist.shape)})

source_kde = ColumnDataSource({'all_kdes': kdes, 
                               'x': x_kde, 'bws': bws, 
                               'kde_curr': kdes[2,:],
                               })


#f.line(x='x', y='counts', alpha=0.3, color='blue', 
#       legend='', source=source_hist)
#f.vbar(x='x', top='counts', width='bw',
#       alpha=0.9, color=hist_color,   
#       legend='normalized counts', source=source_hist)

f.line(x='x', y='kde_curr', alpha=0.85, color=kde_color, line_width=3,  
       legend='KDE', source=source_kde)


#> create slider
bw_slider_callback = CustomJS(args=dict(source=source_kde), code="""
  let data = source.data;
  let bw = cb_obj.value;
  console.log('JS activated')
  
  let all_kdes = data['all_kdes'];
  let bws = data['bws'];
  let kde_curr = data['kde_curr'];
  
  for (let i = 0; i < bws.length; i++) {
    console.log(i)
    if (bws[i] === bw) {
      break;
    }
  kde_curr = all_kdes[i,:];
  }
  source.change.emit();
""")

bw_slider = Slider(start=0.01, end=2.0, step=dbw, value=bw0, format='0[.]000', 
                   title='bin/band-width', width=500)
bw_slider.js_on_change('value', bw_slider_callback)


#l = layout([[f], [bw_slider]])
l = column(f, bw_slider)

curdoc().add_root(l)

show(l)
#html = file_html(layout, CDN, 'kde-slider_bokeh')
#output_file('kde-slider_bokeh.html')

