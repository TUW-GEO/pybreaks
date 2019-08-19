# -*- coding: utf-8 -*-

from pybreaks.utils import merge_dicts, dt_freq, conditional_temp_resample, \
    df_conditional_temp_resample, autocorr, crosscorr, flatten_dict, days_in_month, \
    mid_month_target_values, filter_by_quantiles
import numpy as np
import pandas as pd
from tests.helper_functions import read_test_data

should_days_in_month = np.array([31., 29., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])


def test_merge_dicts():
    d1 = {0:0, 1:1}
    d2 = {2:2, 3:3}
    d = merge_dicts(d1, d2)
    for i, (k, v) in enumerate(d.items()):
        assert k == v == i

def test_conditional_temp_resample():
    ds = pd.Series(index=pd.date_range(start='2000-01-01', end='2000-12-31', freq='D'),
                   data=[1.]*366)

    assert dt_freq(ds.index) == (1., 'D')

    resampled, count = conditional_temp_resample(ds, 'M', threshold=0.1)
    assert dt_freq(resampled.index) == (1., 'M')
    np.testing.assert_almost_equal(count['count_is'], should_days_in_month)
    np.testing.assert_almost_equal(count['count_should'], should_days_in_month / 10.)
    assert all(resampled.values == 1.)

def test_df_conditional_temp_resample():
    df = pd.DataFrame(index=pd.date_range(start='2000-01-01', end='2000-12-31', freq='D'),
                      data={'data1': [1.] * 366, 'data2': [10.] * 366})
    assert dt_freq(df.index) == (1., 'D')

    resampled = df_conditional_temp_resample(df, 'M', resample_threshold=0.1)
    assert dt_freq(df.index) == (1., 'D')
    assert dt_freq(resampled.index) == (1., 'M')
    assert all(resampled['data1'].values == 1.)
    assert all(resampled['data2'].values == 10.)

def test_filter_by_qunatiles():
    df = pd.DataFrame(index=range(10), data={'data':[1,5,5,5,5,5,5,5,5,10]})
    filter_mask = filter_by_quantiles(df, 'data', 0.1, 0.9)
    assert filter_mask.iloc[0] == 1
    assert filter_mask.iloc[-1] == 1
    np.testing.assert_equal(filter_mask.iloc[1:-1].values, np.array([0,0,0,0,0,0,0,0,]))

def test_autocorr():
    df = pd.Series(index=range(10), data=[0,1] * 5)
    ac = autocorr(df, lag=[0,1,2])
    np.testing.assert_almost_equal([1, -1 , 1], ac, decimal=5)

def test_crosscorr():
    df = pd.Series(index=range(10), data=[0,1] * 5)
    cc = crosscorr(df, df, lag=[0,1,2])
    np.testing.assert_almost_equal([1, -1 , 1], cc, decimal=5)

def test_fatten_dict():
    test_dict = {'dict1': {'d1k1' : 'v1', 'd1k2' : 'v2'},
                 'dict2': {'d2k1' : 'v3', 'd2k2' : 'v4'} }

    flat_dict = flatten_dict(test_dict)
    assert(flat_dict['dict1_d1k1'] == 'v1')
    assert(flat_dict['dict1_d1k2'] == 'v2')
    assert(flat_dict['dict2_d2k1'] == 'v3')
    assert(flat_dict['dict2_d2k2'] == 'v4')

def test_days_in_month():
    dim = should_days_in_month.copy()
    for year in [2016, 2017]: # leap, no leap
        days = days_in_month(month=range(1,13), year=year, astype=float)
        if year == 2017:
            dim[1] = 28.
        else:
            dim[1] = 29.
        np.testing.assert_equal(days, dim)

def test_midmonth_target_values():
    M = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
    t = mid_month_target_values(M)
    np.testing.assert_almost_equal(t, M)

def test_dt_freq():
    ts = read_test_data(654079)
    assert dt_freq(ts.index) == (1., 'D')
    assert dt_freq(ts['CCI'].dropna().resample('M').mean().index) == (1., 'M')


if __name__ == '__main__':
    test_merge_dicts()
    test_conditional_temp_resample()
    test_df_conditional_temp_resample()
    test_filter_by_qunatiles()
    test_autocorr()
    test_crosscorr()
    test_fatten_dict()
    test_days_in_month()
    test_midmonth_target_values()
    test_dt_freq()











