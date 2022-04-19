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
                          data = {'candidate': ([10. , 50.] * 91 + [30.,31.] * 92),
                                  'reference': ([30.,31.] * 183)})
    elif type == 'mean':
        # data with a mean break
        df = pd.DataFrame(index= pd.date_range(start='2000-01-01',
                                               end='2000-12-31', freq='D'),
                          data = {'candidate': ([10.,11.]*(91) + [50.,51.]*(92)),
                                  'reference': [30., 31.] * 183})
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
                          data={'candidate': np.arange(366.),
                                'reference': [366.] * 366})
    elif type == 'asc2': # both ascending
        df = pd.DataFrame(index=pd.date_range(start='2000-01-01',
                                              end='2000-12-31', freq='D'),
                          data={'candidate': np.arange(366.),
                                'reference': np.arange(10., 366.+10.)})

    else:
        df = None

    return df,  breaktime, timeframe

def compare_metrics(group_stats, group_metrics, metrics_change):
    # some group metrics
    should = group_stats['mean_CAN_group0'] - group_stats['mean_REF_group0']
    np.testing.assert_almost_equal(group_metrics['CAN_REF_mean_Diff_group0'], should)
    should = group_stats['median_CAN_group0'] - group_stats['median_REF_group0']
    np.testing.assert_almost_equal(group_metrics['CAN_REF_median_Diff_group0'], should)
    should = group_stats['median_CAN_group0'] - group_stats['median_REF_group0']
    np.testing.assert_almost_equal(group_metrics['CAN_REF_median_Diff_group0'], should)
    should = group_stats['median_CAN_FRAME'] - group_stats['median_REF_FRAME']
    np.testing.assert_almost_equal(group_metrics['CAN_REF_median_Diff_FRAME'], should)
    should = group_stats['iqr_CAN_group0'] / group_stats['iqr_REF_group0']
    np.testing.assert_almost_equal(group_metrics['CAN_REF_iqr_Ratio_group0'], should)
    should = group_stats['var_CAN_FRAME'] / group_stats['var_REF_FRAME']
    np.testing.assert_almost_equal(group_metrics['CAN_REF_var_Ratio_FRAME'], should)

    # some changes in metrics
    assert list(metrics_change.keys()) == ['group0_group1']
    should = np.abs(group_metrics['CAN_REF_PearsonR_group0'] - group_metrics['CAN_REF_PearsonR_group1'])
    np.testing.assert_almost_equal(metrics_change['group0_group1']['CAN_REF_AbsDiff_PearsonR'], should)
    should = np.abs(group_metrics['CAN_REF_SpearmanR_group0'] - group_metrics['CAN_REF_SpearmanR_group1'])
    np.testing.assert_almost_equal(metrics_change['group0_group1']['CAN_REF_AbsDiff_SpearmanR'], should)
    should = np.abs(group_metrics['CAN_REF_var_Ratio_group0'] - group_metrics['CAN_REF_var_Ratio_group1'])
    np.testing.assert_almost_equal(metrics_change['group0_group1']['CAN_REF_AbsDiff_var_Ratio'], should)
    should = np.abs(group_metrics['CAN_REF_median_Diff_group0'] - group_metrics['CAN_REF_median_Diff_group1'])
    np.testing.assert_almost_equal(metrics_change['group0_group1']['CAN_REF_AbsDiff_median_Diff'], should)
    return True

def gen_diff_cover_data(monthly=False, rand1=0.5, rand2=0.5):

    # Create a time series where the fist set covers mainly months in winter
    # and the second time series mainly months in summer.

    ds1, ds2, s1, e1, s2, e2 = gen_full_data(monthly, rand1, rand2)

    # drop all DEC JAN FEB in 1
    # drop all JUN JUL AUG in 2
    for month in [12, 1, 2]:
        ds1.loc[ds1.index.month == month ] = np.nan

    for month in [6, 7, 8]:
        ds2.loc[ds2.index.month == month ] = np.nan

    return ds1, ds2, s1, e1, s2, e2


def gen_full_data(monthly=False, rand1=0.5, rand2=0.5):
    # 2 time series with  nans randomly placed
    # for daily:
    # first day that should have a value is 2000-4-3, last 2005-10-2 for ds1
    # first day that should have a value is 2005-10-3, last 2008-8-3 for ds2
    # for monthly:
    # first month that should have a value is 2000-4-30, last 2005-09-31
    # first day that should have a value is 2005-10-31, last 2008-8-31 for ds2

    np.random.seed(282629734)

    if monthly:
        start1 = datetime(2000,4,30)
        end1 = datetime(2005,9,30)
        start2 = datetime(2005,10,31)
        end2 = datetime(2008,8, 31)

        index1 = pd.date_range(start=start1, end=end1, freq='M')
        index2 = pd.date_range(start=start2, end=end2, freq='M')

    else:
        start1 = datetime(2000, 4, 3)
        end1 = datetime(2005, 10, 2)
        start2 = datetime(2005, 10, 3)
        end2 = datetime(2008, 8, 3)

        index1 = pd.date_range(start=start1, end=end1, freq='D')
        index2 = pd.date_range(start=start2, end=end2, freq='D')

    data1 = np.array([np.random.rand(index1.size)])
    data2 = np.array([np.random.rand(index2.size)])

    rand1 = int(data1.size * rand1)
    rand2 = int(data2.size * rand2)

    data1.ravel()[np.random.choice(data1.size, rand1, replace=False)] = np.nan
    data2.ravel()[np.random.choice(data2.size, rand2, replace=False)] = np.nan

    ds1 = pd.Series(index=index1, data=data1[0]).dropna()
    ds2 = pd.Series(index=index2, data=data2[0]).dropna()

    return ds1, ds2, start1, end1, start2, end2

