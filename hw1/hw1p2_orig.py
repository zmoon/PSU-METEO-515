#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:48:26 2018

@author: zmoon
"""

from __future__ import division
from collections import OrderedDict
import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
#from sklearn.neighbors import KernelDensity
#import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, _plot_corr
from statsmodels.nonparametric.kde import KDEUnivariate
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
    
    
    return nao_rn, amo_us_n
    

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


#%% load data

nao, amo_us = load_data()

#> select time period
#  is assigning a .loc different from dropping in place? (i.e. is the Series copied, using memory?)
ya = dt.datetime(1900, 1, 1)
yb = dt.datetime(2015, 12, 31)
nao = nao.loc[(nao.index >= ya) & (nao.index <= yb)]
amo_us = amo_us.loc[(amo_us.index >= ya) & (amo_us.index <= yb)]


#> normalizing

nao_bar = np.nanmean(nao)
s_nao = np.nanstd(nao)
nao_rn = (nao - nao_bar) / s_nao  # rn for re-normlized

amo_us_bar = np.nanmean(amo_us)
s_amo_us = np.nanstd(amo_us)
amo_us_n = (amo_us - amo_us_bar) / s_amo_us  # n for normalized
    



#%% a) summary stats

#df = pd.DataFrame(data=[])
stats = make_table([nao_rn, amo_us_n], ['NAO', 'AMO'])


#%% b) plot time series

f1, [a1, a2] = plt.subplots(2, 1, figsize=(9, 3), sharex=True, sharey=True, num='ts')

#> some settings for both plots
a1.set_xlim( (mdates.date2num(nao.index[0]), mdates.date2num(nao.index[-1])) ) 
lw = 0.5
txtbboxprops = dict(boxstyle='square',
                    facecolor='0.6', edgecolor='0.4', alpha=0.65, 
                    )
txtcolor = 'g'
txtsize = 12

a1.plot(nao_rn, lw=lw)
#a1.set_ylabel('normalized NAO index ()')

#> find good spot for txt
#  note: could really use `transform=ax.transAxes` for this instead
xlim, ylim = a1.get_xlim(), a1.get_ylim()
xtxt = xlim[0] + (xlim[-1]-xlim[0])*0.011
ytxt = ylim[0] + (ylim[-1]-ylim[0])*0.93

#s = 'NAO \small{2x-normalized}'  # pdf saved with tex text has spacing issues, for the fig.text and too much right padding in the text bbox
s = 'NAO 2x-normalized'
a1.text(xtxt, ytxt, s, #usetex=True,
        color=txtcolor, size=txtsize, va='top', ha='left',
        bbox=txtbboxprops)
#a1.set_ylabel('unitless', style='italic')

a2.plot(amo_us_n, lw=lw)
#a2.set_ylabel('normalized, unsmoothed AMO index ()')
#a2.set_ylabel('unitless', style='italic')
a2.set_xlabel('year')

#s = 'AMO \small{normalized}'
s = 'AMO normalized'
a2.text(xtxt, ytxt, s, #usetex=True,
        color=txtcolor, size=txtsize, va='top', ha='left', 
        bbox=txtbboxprops)

#s = r'\textit{unitless}'+'\n'+r'\small{(standardized/normalized anomaly)}'
#s = '*unitless*\n(standardized/normalized anomaly)'
#f1.text(0.02, 0.55, s, #usetex=True, 
#        ha='center', va='center', rotation='vertical')
f1.text(0.012, 0.55, 'unitless', 
        ha='center', va='center', rotation='vertical', style='italic')
f1.text(0.03, 0.55, '(standardized/normalized anomaly)', 
        ha='center', va='center', rotation='vertical', size=8)

f1.tight_layout(h_pad=0.10, rect=[0.025, 0, 1.0, 1.0])


#%% c) hists + kernel density est
#      for different smoothing parameters

nao_rn_size = nao_rn.size
nao_rn_min = nao_rn.min()
nao_rn_max = nao_rn.max()
amo_us_n_size = amo_us_n.size
amo_us_n_min = amo_us_n.min()
amo_us_n_max = amo_us_n.max()

kd_bandwidths  = [0.05, 0.1, 0.25, 0.5, 1.0]
x_kde = np.linspace(-3.5, 3.5, 1000)
kde_color = '#ea4800'  # '#ea3c00', 'orange'
kde_lw = 1.5

hist_binwidths = kd_bandwidths#[0.05, 0.1, 0.5, 1, 2]

f2, aa = plt.subplots(len(kd_bandwidths), 2, 
                      figsize=(7.0, 9.0), sharex=True, sharey=True, num='hists-and-kd',
                      )

txtbboxprops2 = txtbboxprops
txtbboxprops2.update({'alpha': 1.0})  # for overlaying in between subplots

for i, (a1, a2) in enumerate(aa):
    
    kd_bw = kd_bandwidths[i]
    h_bw = hist_binwidths[i]
    
    h_bins_nao = np.arange(nao_rn_min, nao_rn_max+h_bw, h_bw)
    a1.hist(nao_rn, bins=h_bins_nao, density=True,
            alpha=0.7, ec='0.35', lw=0.25)
    
    #> using sklearn
#    kde = KernelDensity(kernel='gaussian', bandwidth=kd_bw).fit(nao_rn)
#    log_dens = kde.score_samples(x_kde)
#    a1.plot(x_kde, np.exp(log_dens), 'r-')
    
    #> using StatsModels
    kde = KDEUnivariate(nao_rn)
    kde.fit(bw=kd_bw)
    a1.plot(x_kde, kde.evaluate(x_kde), '-', color=kde_color, lw=kde_lw)
    
#    s = 'hist binwidth = {:g}\nKDE bandwidth = {:g}'.format(h_bw, kd_bw)
#    a1.text(xtxt, ytxt, s,
#        color=txtcolor, size=11, va='center', ha='center', zorder=3, 
#        bbox=txtbboxprops2)
    
    h_bins_amo = np.arange(amo_us_n_min, amo_us_n_max+h_bw, h_bw)
    a2.hist(amo_us_n, bins=h_bins_amo, density=True,
            alpha=0.7, ec='0.35', lw=0.25)

    #> using StatsModels
    kde = KDEUnivariate(amo_us_n)
    kde.fit(bw=kd_bw)
    a2.plot(x_kde, kde.evaluate(x_kde), '-', color=kde_color, lw=kde_lw)


#> label things
aa[0,0].set_title('NAO')
aa[0,1].set_title('AMO')

ytxts = np.linspace(0.15, 0.9, len(kd_bandwidths))[::-1]
for i, (kd_bw, h_bw) in enumerate(zip(kd_bandwidths, hist_binwidths)):
    
    s = 'hist binwidth = {:g}\nKDE bandwidth = {:g}'.format(h_bw, kd_bw)
    f2.text(0.53, ytxts[i], s,
        color=txtcolor, size=10, va='center', ha='center',
        bbox=txtbboxprops2)


#f2.text(0.015, 0.55, 'relative frequency', #usetex=True, 
#        ha='center', va='center', rotation='vertical')

aa[2,0].set_ylabel('relative frequency (hists) / density (KDE)')

f2.text(0.53, 0.0, 'standardized anomaly', #usetex=True, 
        ha='center', va='bottom')
        
f2.tight_layout(h_pad=0.05, w_pad=0.05, rect=[0.0, 0.006, 1.0, 1.0])
#f2.tight_layout(h_pad=0.05, w_pad=0.05)


#%% d) autocorrelation

ilags = np.arange(0, 36+1, 1)  # symmetrical problem

def calc_acorr(x, ilags=ilags, corr_method='Pearson'):
    """Calculate autocorrelation for certain corr method
    Subsets the input x vector to do so
    Though since we have extra data outside the bounds we could use it instead...
    """
    x = np.array(x)
    
#    if corr_method == 'Pearson':
#        f = ss.pearsonr
#    elif corr_method == 'KT':
#        f = ss.kendalltau
#    elif corr_method == 'Spearman'
#        f = ss.spearmanr
#    else:
#        print('not supported')
#        return False
    
    fns = {'Pearson': ss.pearsonr, 
           'KT': ss.kendalltau,
           'Spearman': ss.spearmanr}
    
    try: 
        f = fns[corr_method]
    
    except KeyError:
        print('not supported')
        return
    
    acorr = np.zeros(ilags.shape)
    for j, ilag in enumerate(ilags):
        
        if ilag > 0:
            x0 = x[:-ilag]
            y = x[ilag:]
        else:
            x0 = x
            y = x
        
#        print(f(x0, y))
        acorr[j] = f(x0, y)[0]
    
    return acorr
    

indices = {'NAO': nao_rn, 'AMO': amo_us_n}
corr_methods = {'Pearson': "Pearsons's $r$", 
                'KT': r"Kendall's $\tau$", 
                'Spearman': r"Spearman's $\rho$"}

f3, aa = plt.subplots(len(corr_methods), len(indices),
                      figsize=(6.5, 5.5), sharex=True, sharey=True, num='auto-corr')

xtxt = ilags.mean()
ytxt = 0.97

for i, corr_method in enumerate(corr_methods):
    rowaa = aa[i]
    
    for j, indexname in enumerate(indices):
        ax = rowaa[j]
        
        if j == 0:
            assert(indexname == 'NAO')  # for title..
        
        data = indices[indexname]
        
        acorr = calc_acorr(data, corr_method=corr_method)
        
        markers, stemlines, baseline = ax.stem(acorr)
        plt.setp(markers,   color='#006bb3', ms=4, zorder=2)
        plt.setp(stemlines, color='0.2', linewidth=1.0, zorder=1)
        plt.setp(baseline,  color='#006bb3', linewidth=1.5, zorder=1)
                 
#        ax.fill_between(  color='#99d6ff')  # for confidence interval...
        
#        plt.figure()
#        plot_acf(acorr, ax=ax)
#        _plot_corr(ax, 'title', acorr, None, ilags, [], True, {}) 
        
        s = corr_methods[corr_method]
#        s = indexname + '\n' + r'\small{{{}}}'.format(corr_methods[corr_method])
        ax.text(xtxt, ytxt, s, #usetex=True,
            color=txtcolor, size=12, va='top', ha='left',
            bbox=txtbboxprops)


aa[0,0].set_title('NAO')
aa[0,1].set_title('AMO')

f3.text(0.015, 0.55, 'correlation coefficient', #usetex=True, 
        ha='center', va='center', rotation='vertical')

f3.text(0.55, 0.0, 'lag', #usetex=True, 
        ha='center', va='bottom')
        
f3.tight_layout(h_pad=0.05, w_pad=0.05, rect=[0.025, 0.005, 1.0, 1.0])

    
#%% save figs

#for fignum in plt.get_fignums():#[f1, f2, f3]:
#    f = plt.figure(fignum)
#    f.savefig('hw1p2_'+f.canvas.get_window_title()+'.pdf',
#              transparent=True,
#              bbox_inches='tight', pad_inches=0.05,
#              )
#    f.savefig('hw1p2_'+f.canvas.get_window_title()+'.png',
#              transparent=True, dpi=300,
#              bbox_inches='tight', pad_inches=0.05,
#              )
