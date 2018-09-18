#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: zmoon
"""

from __future__ import division, print_function
#import datetime as dt

from colorama import init, Fore
#init(autoreset=True)  # using init somehow turns off color output in the Spyder IPython console...
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
#import statsmodels.api as sm
#from statsmodels.robust.scale import mad

plt.close('all')
plt.style.use('seaborn-darkgrid')


#%% load data

dfname = './data/SC_data.xlsx'

convert_dateint = lambda d: pd.to_datetime(str(d))

df = pd.read_excel(dfname,
    header=None, 
    names=['date', 'Tmax', 'Tmin', 'PCP', '5', '6'],
    converters={0: convert_dateint},
    )
df.set_index('date', inplace=True)


#%% process data

#> correct for missing values

varnames = ['Tmax', 'Tmin', 'PCP']
for varname in varnames:  # probably can do without loop
    df[varname].loc[df[varname] == -99.0] = np.nan  # the missing value tag is -99

df.PCP.loc[df.PCP < 0] = np.nan  # trace precip is given the value -1

df_dropna = df.dropna()  # this drops entire rows that have any nans in them...

data = {}  # here drop nans from columns individually
for varname in varnames:
#    data[varname] = df[varname].dropna()  # gives SettingWithCopyWarning
    data[varname] = df.loc[:,varname].dropna()  # apparently this is the preferred method?: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
                                                # maybe should also do .dropna(inplace=True) in a separate line
#> analyze missing/trace data fraction
print('NaN counts:')
print(df.isna().apply(np.count_nonzero))
fmt = '  {c1:s}{{:>8s}}: {c2:s}{{:.2g}}'.format(c1=Fore.GREEN, c2=Fore.BLUE)
print('\nMissing data fraction:')
for k, v in data.items():
    print(fmt.format(k, 1-v.size/df.shape[0]))
print(fmt.format('any var', 1-df_dropna.size/df.size))

#> omit 2017?? to be fair in the annual means and such
#  but we do have all the way to Nov 20th, so think will leave it in...


#%% plot data time series
#   for sanity check

f0, aa = plt.subplots(3, 1, sharex=True)

for i, k in enumerate(varnames):
    df.plot(y=k, ax=aa.flat[i])  # the raw data, but with missing values marked with nan
 

#%% a) hist + Gaussian fit
#      for annual *average* Tmax and PCP

#> compute annual averages
#df_annual_means = df.groupby(df.index.dt.year).transform('mean')  # does not reduce row number
#df_annual_means = df.groupby(pd.TimeGrouper('A')).mean()  # TimeGrouper is deprecated
df_annual_means = df.groupby(pd.Grouper(freq='A')).mean()  # DataFrame.mean skips nans by default
df_annual_means2 = df.groupby(pd.Grouper(freq='A')).apply(np.nanmean)

deg_symbol = u'\u00B0'
pdf_color = '#ea4800'

f1, aa = plt.subplots(1, 2, figsize=(6, 3), num='annual_mean')

names = ['Tmax', 'PCP']
for i, name in enumerate(names):
    
    ax = aa[i]
    ds = df_annual_means[name]
    
    xplot = np.linspace(ds.min()*0.98, ds.max()*1.02, 400)
    
    ax.hist(ds, bins=13, density=True, alpha=0.7, ec='0.35', lw=0.25, label=None)
    
    xbar = ds.mean()
    s = ds.std()
    ax.plot(xplot, ss.norm.pdf(xplot, xbar, s), '-', c=pdf_color, alpha=0.85, lw=2, 
            label='$\mathcal{{N}}({:.2g}, {:.2g})$'.format(xbar, s**2))
    
    ax.set_title('Annual *mean* daily '+name)
    ax.text(0.98, 0.98, '$N={:d}$'.format(ds.size), va='top', ha='right', transform=ax.transAxes)
    
    ax.legend(loc='upper left')

aa[0].set_xlabel(deg_symbol+'F')
aa[0].set_ylabel('density')
aa[1].set_xlabel('inches')

f1.tight_layout()


#%% Gumbel def

def myGumbel_pdf(x, loc=0, scale=1):
    """Calculate pdf for Gumbel dist at x
    loc:   location/shift param zeta
    scale: scale param beta
    """
    return 1/scale * np.exp( -np.exp(-(x-loc)/scale) - (x-loc)/scale)


#%% b) hist + Gumbel fit
#      for annual *maximum* Tmax and PCP

df_annual_maxs = df.groupby(pd.Grouper(freq='A')).max()

f2, aa = plt.subplots(1, 2, figsize=(6, 3), num='annual_max')

names = ['Tmax', 'PCP']
for i, name in enumerate(names):
    
    ax = aa[i]
    ds = df_annual_maxs[name]
    
    xplot = np.linspace(ds.min(), ds.max(), 400)
    
    ax.hist(ds, bins=None, density=True, alpha=0.7, ec='0.35', lw=0.25, label=None)
    
    xbar = ds.mean()
    s = ds.std()
    beta_hat = s*np.sqrt(6)/np.pi
    zeta_hat = xbar - np.euler_gamma*beta_hat
    ssfit = ss.gumbel_r.pdf(xplot, zeta_hat, beta_hat)
    myfit = myGumbel_pdf(xplot, zeta_hat, beta_hat)
    assert( np.allclose(ssfit, myfit) )  # our Gumbel formula gives same results as gumbel_r, not gumbel_l...
    
    s = 'Gumbel\n$\zeta = {:.2g}$,\n' + r'$\beta = {:.2g}$'
    ax.plot(xplot, myfit, '-', c=pdf_color, alpha=0.85, lw=2, 
            label=s.format(zeta_hat, beta_hat))
#    ax.plot(xplot, myfit, '-', c='g', alpha=0.85, lw=2, label='my Gumbel')
    
    ax.set_title('Annual *maximum* daily '+name)
    ax.text(0.98, 0.98, '$N={:d}$'.format(ds.size), va='top', ha='right', transform=ax.transAxes)    

    ax.legend(loc='center right')

aa[0].set_xlabel(deg_symbol+'F')
aa[0].set_ylabel('density')
aa[1].set_xlabel('inches')

f2.tight_layout()


#%% c) q-q plot of data vs Gumbel fits

f3, aa = plt.subplots(1, 2, figsize=(6, 3), num='annual_max_q-q_gumbel')

names = ['Tmax', 'PCP']
for i, name in enumerate(names):
    
    ax = aa[i]
    ds = df_annual_maxs[name]

    ss.probplot(ds, dist=ss.gumbel_r, plot=ax)
    
    ax.set_title('Annual *maximum* daily '+name)
    
f3.tight_layout()

