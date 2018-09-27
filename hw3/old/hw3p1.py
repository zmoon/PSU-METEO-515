# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:20:02 2018

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
import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf, _plot_corr
#from statsmodels.nonparametric.kde import KDEUnivariate
#from statsmodels.robust.scale import mad

plt.close('all')

#%% load data
#   note that this time we are using the unsmoothed, non-detrended AMO

amo_fpath = './data/amon.us.long.mean.data'

amo_raw = np.genfromtxt(amo_fpath, skip_header=1, skip_footer=7)[:,1:]

with open(amo_fpath, 'r') as f: yr_range = f.readline().split()
t_amo = pd.date_range(start='{}/01/01'.format(yr_range[0]), freq='MS', periods=amo_raw.size)
amo_ndt_us = amo_raw.reshape((amo_raw.size,))

amo = pd.DataFrame({'amo': amo_ndt_us}, index=t_amo)
amo['julian_date'] = amo.index.to_julian_date()
amo['t_elapsed'] = amo.julian_date - amo.julian_date[0]
amo['year'] = amo.index.year
amo['decyear'] = amo.index.year + (amo.index.month-1)/12

amo[amo == -99.99] = np.nan
amo.dropna(inplace=True)

amo_annual_mean = amo.loc[amo.year<=2017, :].groupby(pd.Grouper(freq='A')).mean()
# note that this gives last day of year in the index!!
# and x.46 for decimal year, since initially the datetimes are first day of the year


#%% simple linear regression

f1, a = plt.subplots(num='ts_ols')

df_reg = amo_annual_mean  # gives slightly different answers, at least for error!. maybe becaues including the partial year?
t_reg = df_reg.year
t_plot = df_reg.index

res = sm.OLS(df_reg['amo'], sm.add_constant(t_reg), ).fit()
#print(res.summary())  # note: can plot res using `sm.graphics.abline_plot(model_results=res)`
const, slope = res.params
slope_ci = res.conf_int().iloc[1].values
slope_bse = res.bse[1]
const_bse = res.bse[0]
#slope_results.append({l: {'slope': slope, 'slope_ci': slope_ci, 'slope_bse': bse}})

#a.plot(amo['amo'], 'b-.', alpha=0.3, ms=1, lw=0.5)
a.plot(amo_annual_mean['amo'], 'b.-', alpha=0.6, ms=6, lw=1)
a.plot(t_plot, t_reg*slope+const, '-', c='r', lw=2)

s = r'$y=\beta_1 x + \beta_0${nl:s}$\beta_1 = {:.3g} \pm {:.3g}${nl:s}$\beta_0 = {:.3g} \pm {:.3g}$'.format(slope, slope_bse, const, const_bse, nl='\n')
a.text(0.02, 0.98, s,
       va='top', ha='left', transform=a.transAxes)

a.set_ylabel('AMO index (deg. C)')


#%% multiple linear regression, adding one oscillation as a predictor
#   with prescribed phase and period

df_reg = amo_annual_mean

phase = -20  # years
period = 65  # in years

y = df_reg['amo'].values  # AMO index: annual mean of monthly mean AMO

x0 = np.ones(df_reg['amo'].shape)  # intercept
x1 = df_reg['year']  # year
x2 = np.sin(2*np.pi/period*(x1-phase))  # oscillation with prescribed period and phase

X = np.vstack((x2, x1, x0)).T

res = sm.OLS(y, X).fit()

#%%
f2, a = plt.subplots(num='ts_ols2')

a.plot(amo_annual_mean['amo'], 'b.-', alpha=0.6, ms=6, lw=1)
a.plot(t_plot, np.dot(X, res.params), '-', c='r', lw=2)

stuff = []
for i in range(len(res.params)):
    stuff.append(res.params[i])
    stuff.append(res.bse[i])

param_lines = '\n'.join([r'$\beta_{:d} = {{:.3g}} \pm {{:.3g}}$'.format(i) for i in range(len(res.params)-1, 0-1, -1)])

s = r'''
$y = \beta_2 \cdot \sin\left( \frac{{2 \pi}}{{\mathrm{{period}}}} (t - \mathrm{{phase}}) \right) + \beta_1 \cdot t + \beta_0$
period = {:d}
phase = {:d}
''' + param_lines#).format(period, phase *stuff)
s = s.format(period, phase, *stuff)

#s = r'$y=\beta_1 x + \beta_0${nl:s}$\beta_1 = {:.3g} \pm {:.3g}${nl:s}$\beta_0 = {:.3g} \pm {:.3g}$'.format(slope, slope_bse, const, const_bse, nl='\n')
a.text(0.02, 0.98, s.strip(),
       va='top', ha='left', transform=a.transAxes)

a.set_ylabel('AMO index (deg. C)')



