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

#%% fns

def read(fpath):  # could make this
    """ """
    pass



#%% load data

nao_fpath = '../data/nao.long.data.txt'
amo_fpath = '../data/amon.us.long.data.txt'  # us for unsmoothed

#> NAO
nao_raw = np.genfromtxt(nao_fpath, skip_header=1, skip_footer=6)[:,1:]

with open(nao_fpath, 'r') as f:
    yr_range = f.readline().split()
t_nao = pd.date_range(start='{}/01/01'.format(yr_range[0]), freq='MS', periods=nao_raw.size)
nao = nao_raw.reshape((nao_raw.size,))

nao = pd.Series(nao, index=t_nao)
nao[nao == -99.99] = np.nan
nao.dropna(inplace=True)

#> AMO
amo_raw = np.genfromtxt(amo_fpath, skip_header=1, skip_footer=4)[:,1:]

with open(amo_fpath, 'r') as f:
    yr_range = f.readline().split()
t_amo = pd.date_range(start='{}/01/01'.format(yr_range[0]), freq='MS', periods=amo_raw.size)
amo_us = amo_raw.reshape((amo_raw.size,))

amo_us = pd.Series(amo_us, index=t_amo)

amo_us[amo_us == -99.99] = np.nan
amo_us.dropna(inplace=True)

#> select time period
#  is assigning a .loc different from dropping in place? (i.e. is the Series copied, using memory?)
ya = dt.datetime(1900, 1, 1)
yb = dt.datetime(2015, 12, 31)
nao = nao.loc[(nao.index >= ya) & (nao.index <= yb)]
amo_us = amo_us.loc[(amo_us.index >= ya) & (amo_us.index <= yb)]


#%% normalizing

nao_bar = np.nanmean(nao)
s_nao = np.nanstd(nao)
nao_rn = (nao - nao_bar) / s_nao  # rn for re-normalized

amo_us_bar = np.nanmean(amo_us)
s_amo_us = np.nanstd(amo_us)
amo_us_n = (amo_us - amo_us_bar) / s_amo_us  # n for normalized




#%% hists + kernel density est
#      for different smoothing parameters

#> pre-calculations for hist binning
nao_rn_size = nao_rn.size
nao_rn_min = nao_rn.min()
nao_rn_max = nao_rn.max()
amo_us_n_size = amo_us_n.size
amo_us_n_min = amo_us_n.min()
amo_us_n_max = amo_us_n.max()

x_kde = np.linspace(-3.9, 3.9, 400)

def calcHist(h_bw=0.1):
    """ """
    #> create bins, spaced evenly
    #  currently starts at min value, but could instead start at floored value
    #  so as to get a bin centered on 0
    h_bin_edges_nao = np.arange(nao_rn_min, nao_rn_max+h_bw, h_bw)
    
    #> relative frequency binning
    h_bin_counts = np.histogram(nao_rn, bins=h_bin_edges_nao, density=True)[0]  # also returns bin edges but we have them already

    #> bin locs
    h_bin_centers_nao = h_bin_edges_nao[:-1] + 0.5*h_bw
    
    return h_bin_counts, h_bin_centers_nao


def calcKDE(kd_bw=0.1):
    """ """
    
    #> KDE using StatsModels
    kde = KDEUnivariate(nao_rn)
    kde.fit(bw=kd_bw)
    
    return kde.evaluate(x_kde)
    

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
bw0 = 0.06

counts_hist, x_hist = calcHist(bw0)
source_hist = ColumnDataSource({'x': x_hist, 'counts': counts_hist, 'bw': bw0*np.ones(x_hist.shape)})
#source_hist2 = ColumnDataSource({'x': x_hist, 'counts': counts_hist})

kde = calcKDE(bw0)
source_kde = ColumnDataSource({'x': x_kde, 'density': kde})

#f.line(x='x', y='counts', alpha=0.3, color='blue', 
#       legend='', source=source_hist)
f.vbar(x='x', top='counts', width='bw',
       alpha=0.9, color=hist_color,   
       legend='normalized counts', source=source_hist)

f.line(x='x', y='density', alpha=0.85, color=kde_color, line_width=3,  
       legend='KDE', source=source_kde)


#> create slider
bw_slider = Slider(start=0.01, end=2.0, step=0.005, value=bw0, format='0[.]000', 
                   title='bin/band-width', width=500)

def bw_slider_callback(attr, old, new):
    """ """
    bw = new
    
    counts_hist, x_hist = calcHist(bw)
    new_hist = {'x': x_hist, 'counts': counts_hist, 'bw': bw*np.ones(x_hist.shape)}
    
    kde = calcKDE(bw)
    new_kde = {'x': x_kde, 'density': kde}
    
    source_hist.data = new_hist
    source_kde.data = new_kde

bw_slider.on_change('value', bw_slider_callback)

#l = layout([[f], [bw_slider]])
l = column(f, bw_slider)

curdoc().add_root(l)

show(l)
#html = file_html(layout, CDN, 'kde-slider_bokeh')
#output_file('kde-slider_bokeh.html')

