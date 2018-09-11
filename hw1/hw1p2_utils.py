#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:48:26 2018

@author: zmoon
"""

from __future__ import division
from collections import OrderedDict
#import datetime as dt

#import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
#from sklearn.neighbors import KernelDensity
#import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, _plot_corr
#from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.robust.scale import mad

plt.close('all')

#%% fns

def load_data():  # or use a read fn instead of writing things 2x...
    """ """
    nao_fpath = './data/nao.long.data.txt'
    amo_fpath = './data/amon.us.long.data.txt'  # us for unsmoothed
    
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
    
    
    return nao, amo_us
    

def calc_YK_pd(x):
    """Calculate Yule-Kendall skewness index using Pandas quantile method"""
    try:
        q1 = x.quantile(0.25)
        q2 = x.quantile(0.5)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        return ((q3-q2)-(q2-q1)) / iqr
        
    except AttributeError:
        print('requires pd.Series input')


def summary_stats(x, save_table=False, print_table=True):
    """Descriptive stats for input np array or pd series x"""

    mean = x.mean()  # using array method (Numpy) / Series method (Pandas)

    median = x.median()  # using array method (Numpy) / Series method (Pandas)

    std = x.std()  # using array method (Numpy) / Series method (Pandas)

    iqr = ss.iqr(x)  # using Scipy Stats

    # c is a normalization constant
    # that we only need/want if we are relating MAD to the standard deviation
    #   https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
    mad_sm = mad(x, c=1)  # using StatsModels (Pandas only has *mean* absolute dev)
    
    skew = ss.skew(x)  # using Scipy Stats; Series also have skew() method
    
    skew_yk = calc_YK_pd(x)
    
    names = ['mean', 'median', 'std', 'IQR', 'MAD', 'skewness', 'Y-K']
    varz  = [mean,   median,   std,   iqr,   mad_sm, skew, skew_yk]  # `vars` is a built-in
    d = OrderedDict([(n, v) for n, v in zip(names, varz)])
    
    return d


def make_table(dataframe, names):
    """ """
    results = {}
    for i, v in enumerate(names):
            
        stats = summary_stats(dataframe[i])
        
        if i == 0:  # print header
            fmt = '{:^8s}'*(len(stats)+1)
            print(fmt.format('', *stats.keys()))

        fmt = '{:^8s}' + '{:>8.3f}'*len(stats)
        print(fmt.format(v, *stats.values()))
        results[v] = stats
        
    return results

