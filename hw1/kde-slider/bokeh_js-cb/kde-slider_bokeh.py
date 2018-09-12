#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:24:08 2018

@author: zmoon
"""

from __future__ import division
#from collections import OrderedDict
#import datetime as dt

from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.io import reset_output
from bokeh.io.doc import curdoc, set_curdoc
from bokeh.layouts import layout, row, column, widgetbox
from bokeh.models import CustomJS, Slider#, ColumnDataSource, Select
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.resources import CDN
#import matplotlib.dates as mdates
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import scipy.stats as ss
#from sklearn.neighbors import KernelDensity
#import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, _plot_corr
#from statsmodels.nonparametric.kde import KDEUnivariate
#from statsmodels.robust.scale import mad

#plt.close('all')


#%% load LUTs


x_kde = np.linspace(-3.9, 3.9, 400)  # should load from file...
dbw = 0.002                          # this too
bws = np.arange(0.01, 2.0+dbw, dbw)  # and this stuff. too make sure the LUT and slider match up
kdes = np.loadtxt('lut_kde.csv', delimiter=',')


x_hist_all = np.loadtxt('hist_bincenters.txt', delimiter=',')

hist_counts_all = np.loadtxt('hist_counts.txt', delimiter=',')



#%% bokeh prepare data

doc = Document()  # create fresh Document
set_curdoc(doc)

#> initial data
bw0 = 0.07
ibw0 = 30   # 0.01 + 30*0.002

#counts_hist, x_hist = calcHist(bw0)
#source_hist = ColumnDataSource({'x': x_hist, 'counts': counts_hist, 'bw': bw0*np.ones(x_hist.shape)})

x_hist_lut_dict = {}  # could be combined with the next one
hist_counts_lut_dict = {}
kde_dict = {'x': x_kde}
for i, bw in enumerate(bws):
    k = '{:.3f}'.format(bw)
    
    kde_dict[k] = kdes[i,:]
    
    x_hist_lut_dict[k] = x_hist_all[i,:]
    hist_counts_lut_dict[k] = hist_counts_all[i,:]
    

kde_dict['kde_curr'] = kdes[ibw0,:]  # this indexing here should be set up to correspond to initial slider value and the bw values used to construct the LUT

source_kde = ColumnDataSource(kde_dict)
    
source_hist_counts_lut = ColumnDataSource(hist_counts_lut_dict)
source_x_hist_lut = ColumnDataSource(x_hist_lut_dict)

#source_hist_curr = ColumnDataSource({'x': x_hist_all[ibw0,x_hist_all[ibw0,:] != -99],
#                                     'counts': hist_counts_all[ibw0,x_hist_all[ibw0,:] != -99],
#                                     })
source_hist_curr = ColumnDataSource({'x': x_hist_all[ibw0,:], 
                                    'counts': hist_counts_all[ibw0,:],
                                    'bw': np.full(x_hist_all.shape[1], bw0),
                                    })

#%% bokeh plot

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


#f.line(x='x', y='counts', alpha=0.3, color='blue', 
#       legend='', source=source_hist)
f.vbar(x='x', top='counts', width='bw',
       alpha=0.9, color=hist_color,   
       legend='normalized counts', source=source_hist_curr)

f.line(x='x', y='kde_curr', alpha=0.85, color=kde_color, line_width=3,  
       legend='KDE', source=source_kde)


#> create slider
sources = dict(source_kde=source_kde, 
               source_hist_counts_lut=source_hist_counts_lut,
               source_x_hist_lut=source_x_hist_lut,
               source_hist_curr=source_hist_curr,
               )
bw_slider_callback = CustomJS(args=sources, code="""
  console.log('JS activated')
  let bw = cb_obj.value.toFixed(3);
  console.log('bw: ' + bw)

  // -----------------------------------------------
  // first KDE

  var data = source_kde.data;
  
  var kde_curr = data['kde_curr'];
  //console.log('new data: ' + data[bw]);
  
  //kde_curr = data[+bw];
  for (let i = 0; i < kde_curr.length; i++) {
    kde_curr[i] = data[bw][i];
  }
  
  // -----------------------------------------------
  // now hist
  
  var bincenters_lut = source_x_hist_lut.data;
  var counts_lut = source_hist_counts_lut.data;

  var hist_curr = source_hist_curr.data;
  //var x_hist_curr = hist_curr['x'];
  //var counts_hist_curr = hist_curr['counts'];
  
  for (let i = 0; i < hist_curr['x'].length; i++) {
    hist_curr['x'][i]      = bincenters_lut[bw][i];
    hist_curr['counts'][i] = counts_lut[bw][i];
    hist_curr['bw'][i]     = bw;  // parseFloat()
  }

  
  // -----------------------------------------------
  // send the changes
  source_kde.change.emit();
  source_hist_curr.change.emit();
""")

bw_slider = Slider(start=0.01, end=2.0, step=dbw, value=bw0, format='0[.]000', 
                   title='bin/band-width', width=500)
bw_slider.js_on_change('value', bw_slider_callback)


#l = layout([[f], [bw_slider]])
l = column(f, bw_slider)

#show(l)

doc.add_root(l)  # for using `bokeh serve` or writing html, but not for show() ( I think )

html = file_html(l, CDN, 'kde-slider_bokeh')
with open('kde-slider_bokeh.html', 'w') as f:
    f.write(html)

#output_file('kde-slider_bokeh.html')  # is this the same as using file_html and writing it out??

