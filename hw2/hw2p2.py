#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:14:03 2018

@author: zmoon
"""

from __future__ import division
#import datetime as dt

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
#import pandas as pd
import scipy.optimize as so
#import scipy.stats as ss
#import statsmodels.api as sm
#from statsmodels.robust.scale import mad

plt.close('all')
plt.style.use('seaborn-darkgrid')

#%% load data

fname = './data/twpvdisC3.b1.20150102.000000.cdf'

d = Dataset(fname)

base_time = d['base_time']  # seconds since 1970-1-1 0:00:00 0:00
time_offset = d['time_offset']  # same as time...
time = d['time']  # seconds since base time 
assert( np.all(time[:] == time_offset[:]) )
t_dt = num2date(time[:], time.units)

drop_diameter = d['drop_diameter']
num_density = d['num_density']

intercept_param = d['intercept_parameter']
slope_param = d['slope_parameter']


#%% a) DSD at chosen minute

i_t_choice = np.argmax(num_density[:,10:].sum(axis=1))
t_choice = t_dt[i_t_choice]
#print(t_choice)

Dplot = np.linspace(0, 10, 400)

f1, a = plt.subplots(figsize=(6, 3.0), num='dsd')

a.semilogy(drop_diameter[:], num_density[i_t_choice], 'o-', ms=4, label='data')  # the data
a.semilogy(Dplot, intercept_param[i_t_choice]*np.exp(-slope_param[i_t_choice]*Dplot), 
           label='fit: N$_0$=%.3g, $\lambda$=%.3g' % (intercept_param[i_t_choice], slope_param[i_t_choice]))  # the exp fit

a.set_xlabel('drop diameter $D$ (bin center) [{:s}]'.format(drop_diameter.units))
a.set_xlim(xmax=4)
a.set_ylabel('number density $N(D)$\n[{:s}]'.format('(# drops) m$^{-3}$ mm$^{-1}$'))   #num_density.units))
a.set_ylim(ymin=num_density[i_t_choice][num_density[i_t_choice] > 0].min())  # must be a cleaner way to do this... 
a.set_title(str(t_choice)[:19])  # don't care about sub-second


#%% b) more complex fit
#

def ND3(D, N_0, mu, lamb):
    """More complex Marshall Palmer number density -- drop size relation"""
    return N_0 * D**mu * np.exp(-lamb*D)


popt, pcov = so.curve_fit(ND3, drop_diameter[:], num_density[i_t_choice])

a.plot(Dplot, ND3(Dplot, *popt), 'r-',
       label='fancier fit: N$_0$=%.3g,\n$\mu$=%.3g, $\lambda$=%.3g' % tuple(popt))


a.legend(loc='center left', bbox_to_anchor=(0.98, 0.5))

f1.tight_layout()
#f1.tight_layout(rect=(0, 0, 0.5, 1.0))

