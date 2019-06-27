# -*- coding: utf-8 -*-
"""
Created on Jun 03 20:57 2018

@author: wolfgang
"""

from breakadjustment.standalone_functions import *
import numpy.testing as test



def test_fatten_dict():
    test_dict = {'dict1': {'d1k1' : 'v1', 'd1k2' : 'v2'},
                 'dict2': {'d2k1' : 'v3', 'd2k2' : 'v4'} }

    flat_dict = flatten_dict(test_dict)
    assert(flat_dict['dict1_d1k1'] == 'v1')
    assert(flat_dict['dict1_d1k2'] == 'v2')
    assert(flat_dict['dict2_d2k1'] == 'v3')
    assert(flat_dict['dict2_d2k2'] == 'v4')


def test_autocorr():
    df = pd.Series(index=range(10), data=[0,1] * 5)
    ac = autocorr(df, lag=[0,1,2])
    test.assert_almost_equal([1, -1 , 1], ac, decimal=5)

def test_crosscorr():
    df = pd.Series(index=range(10), data=[0,1] * 5)
    cc = crosscorr(df, df, lag=[0,1,2])
    test.assert_almost_equal([1, -1 , 1], cc, decimal=5)


def test_conditional_temp_resample():
    ds = pd.Series(index=pd.DatetimeIndex(start='2000-01-01', end='2000-02-29', freq='D'),
                   data=15*[np.nan, 1] + [np.nan] + 14*[np.nan, 2] + [np.nan])

    resampled, count = conditional_temp_resample(ds, 'M', threshold=0.1)
    test.assert_almost_equal(resampled.values, [1,2], decimal=5)
    test.assert_almost_equal(count['count_should'], [31*0.1, 29*0.1], decimal=5)
    test.assert_almost_equal(count['count_is'], [15,14], decimal=5) # todo: check nan handling

    resampled, count = conditional_temp_resample(ds, 'M', threshold=0.6) # should fail!!
    test.assert_almost_equal(resampled.values, [1,2], decimal=5)
    test.assert_almost_equal(count['count_should'], [31*0.1, 29*0.1], decimal=5)
    test.assert_almost_equal(count['count_is'], [15,14], decimal=5) # todo: check nan handling


    pass


def test_df_conditional_temp_resample():
    pass


