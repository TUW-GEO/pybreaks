import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

"""
Set up tutorial cases.

These functions are used in the tutorials to allow the user to use either 
artificially generated test data with an introduced break or read real 
ESA CCI SM and MERRA2 model SM data and then introduce a break.
All Tutorials will use these functions to get some data.
"""

def load_test_data(trend=0.01, bias=0.1, breaktime=datetime(2012,7,1), breaksize=(0,1), resample=False, seed=1234):
    '''
    This function creates actificial test data. An additive/multiplicative break can be introduced by the user.
    '''
    x = np.linspace(0, 18 * np.pi, 3287) 
    s_seasonality = (np.sin(x - (np.pi/2)) + 5.) * 0.1
    s_trend =  x * trend

    dt_index=pd.date_range(start='2010-01-01', end='2018-12-31', freq='D')
    np.random.seed(seed)
    rand_can = pd.Series(index=dt_index, data=np.random.normal(
        loc=0., scale=0.05, size=3287), name='CAN')
    np.random.seed(seed+1)
    rand_ref = pd.Series(index=dt_index, data=np.random.normal(
        loc=0., scale=0.05, size=3287), name='REF')

    can = s_seasonality + s_trend + rand_can.values + bias
    ref = s_seasonality + s_trend + rand_ref.values

    # additive and multiplicative relative bias
    break_index=can.loc[:breaktime].index
    can.loc[break_index] = can.loc[break_index] * breaksize[1] + breaksize[0]

    if resample:
        can, ref = can.resample('M').mean(), ref.resample('M').mean()
    
    return can, ref

def load_real_data(gpi=707393, breaktime=datetime(2012,7,1), breaksize=(0,1), resample=False):
    '''
    This function loads real observations from the ESA CCI SM v04.4 COMBINED dataset and the MERRA2 model. 
    An additive/multiplicative break can be introduced by the user.
    GPI is a location identifier for a single cell point in the ESA CCI SM data grid.
    '''
    # add a few points that can be loaded and used
    df = pd.read_csv(os.path.join(os.path.abspath(''), '..', '..', 'tests', 'test-data', 'csv_ts', 
                                  'data_{}.csv'.format(gpi)), index_col=0, parse_dates=True)
    df = df.dropna() # drop missing days here so that the tables look nice
    if breaktime == (0,1):
        can_name = 'ESA CCI SM (COMBINED)'
    else:
        can_name = 'ESA CCI SM (COMBINED) - with break'
    can = df.loc['2010-01-01':'2017-12-31', can_name]
    ref = df.loc['2010-01-01':'2017-12-31', 'MERRA2 SFMC']

    # bias the candidate before the break
    break_index = can.loc[:breaktime].index
    can.loc[break_index] = can.loc[break_index] * breaksize[1] + breaksize[0]

    if resample:
        can, ref = can.resample('M').mean(), ref.resample('M').mean()
        
    return can, ref
