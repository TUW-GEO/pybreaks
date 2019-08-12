# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import os
import numpy as np

def dict_depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0

def read_test_data(gpi, scale_factor=1.):
    """
    Read test data time series from csv files
    Parameters
    ----------
    gpi : int
        Grid Point Index of a point for which test data is stored
    scale_factor: float, optional (default: 1.0)
        Factor that the read time series is multiplied with
    """
    file = 'data_{}.csv'.format(gpi)
    testdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'test-data', 'csv_ts', file)
    ts = pd.read_csv(testdata_path, index_col=0, parse_dates=True) * scale_factor


    return ts

def create_artificial_test_data(type):
    ''' Create obvious test data of the selected type'''

    breaktime = datetime(2000,6,30) # break time belongs to second group!!
    timeframe = np.array([datetime(2000,1,1), datetime(2000,12,31)])
    if type == 'var':
        # data with a var break
        df = pd.DataFrame(index= pd.date_range(start='2000-01-01',
                                               end='2000-12-31', freq='D'),
                          data = {'candidate': ([10 , 50] * 91 + [30,31] * 92),
                                  'reference': ([30,31] * 183)})
    elif type == 'mean':
        # data with a mean break
        df = pd.DataFrame(index= pd.date_range(start='2000-01-01',
                                               end='2000-12-31', freq='D'),
                          data = {'candidate': ([10,11]*(91) + [50,51]*(92)),
                                  'reference': [30, 31] * 183})
    elif type == 'norm_mean':
        # data with a mean break
        np.random.seed(12345)
        can_p1 = np.random.normal(25, 1, 91*2)
        can_p2 = np.random.normal(35, 1, 92*2)
        ref = np.random.normal(30, 1,183*2)
        df = pd.DataFrame(index= pd.date_range(start='2000-01-01',
                                               end='2000-12-31', freq='D'),
                          data = {'candidate': np.concatenate((can_p1, can_p2), axis=0),
                                  'reference': ref})
    elif type == 'const': # constant can and ref
        # data with constant frame values
        df = pd.DataFrame(index=pd.date_range(start='2000-01-01',
                                              end='2000-12-31', freq='D'),
                          data={'candidate': [0.1]*182 + [0.9]*184,
                                'reference': [0.5]*(182+184)})
    elif type == 'empty': # empty data
        df = pd.DataFrame(index=pd.date_range(start='2000-01-01',
                                              end='2000-12-31', freq='D'),
                          data={'candidate': np.nan,
                                'reference': np.nan})
    elif type == 'asc': # ascending candiate
        df = pd.DataFrame(index=pd.date_range(start='2000-01-01',
                                              end='2000-12-31', freq='D'),
                          data={'candidate': range(366),
                                'reference': [366] * 366})
    elif type == 'asc2': # both ascending
        df = pd.DataFrame(index=pd.date_range(start='2000-01-01',
                                              end='2000-12-31', freq='D'),
                          data={'candidate': range(366),
                                'reference': range(10, 366+10)})

    else:
        df = None

    return df,  breaktime, timeframe