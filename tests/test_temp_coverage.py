# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd
import numpy as np
from breakadjustment.standalone_functions import conditional_temp_resample
from breakadjustment.temp_coverage import compare_temp_cov, drop_months_data
import numpy.testing as nptest



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

        index1 = pd.DatetimeIndex(start=start1, end=end1, freq='M')
        index2 = pd.DatetimeIndex(start=start2, end=end2, freq='M')

    else:
        start1 = datetime(2000, 4, 3)
        end1 = datetime(2005, 10, 2)
        start2 = datetime(2005, 10, 3)
        end2 = datetime(2008, 8, 3)

        index1 = pd.DatetimeIndex(start=start1, end=end1, freq='D')
        index2 = pd.DatetimeIndex(start=start2, end=end2, freq='D')

    data1 = np.array([np.random.rand(index1.size)])
    data2 = np.array([np.random.rand(index2.size)])

    rand1 = int(data1.size * rand1)
    rand2 = int(data2.size * rand2)

    data1.ravel()[np.random.choice(data1.size, rand1, replace=False)] = np.nan
    data2.ravel()[np.random.choice(data2.size, rand2, replace=False)] = np.nan


    ds1 = pd.Series(index=index1, data=data1[0]).dropna()
    ds2 = pd.Series(index=index2, data=data2[0]).dropna()


    return ds1, ds2, start1, end1, start2, end2




def test_cover_full_D():

    ds1, ds2, s1, e1, s2, e2 = gen_full_data(False, 0, 0)

    success, cover0, cover1, dcover, mdrop = \
        compare_temp_cov(ds1, ds2, s1, e1, s2, e2, 'D', 0.01)

    assert success == True
    nptest.assert_almost_equal(dcover['diff'].values, np.array([0.] * 12))
    nptest.assert_almost_equal(cover0['coverage'].values, np.array([1.0] * 12))
    nptest.assert_almost_equal(cover1['coverage'].values, np.array([1.0] * 12))
    nptest.assert_almost_equal(mdrop, np.array([]))



def test_cover_full_M():
    ds1, ds2, s1, e1, s2, e2 = gen_full_data(True, 0, 0)

    success, cover0, cover1, dcover, mdrop = \
        compare_temp_cov(ds1, ds2, s1, e1, s2, e2, 'M', 0.01)

    assert success == True
    nptest.assert_almost_equal(dcover['diff'].values, np.array([0.] * 12))
    nptest.assert_almost_equal(cover0['coverage'].values, np.array([1.0] * 12))
    nptest.assert_almost_equal(cover1['coverage'].values, np.array([1.0] * 12))
    nptest.assert_almost_equal(mdrop, np.array([]))


def test_cover_part_D():
    ds1, ds2, s1, e1, s2, e2 = gen_diff_cover_data(False, 0, 0)

    success, cover0, cover1, dcover, mdrop = \
        compare_temp_cov(ds1, ds2, s1, e1, s2, e2, 'D', 0.01)

    diff_shd = np.array([np.nan]*2 + [0]*3 + [np.nan]*3 + [0]*3 + [np.nan])
    coverage0_shd = np.array([np.nan]*2 + [1.0]*9 + [np.nan])
    coverage1_shd = np.array([1.]*5  + [np.nan]*3 + [1.0]*4)

    assert success == False
    nptest.assert_almost_equal(dcover['diff'].values, diff_shd)
    nptest.assert_almost_equal(cover0['coverage'].values, coverage0_shd)
    nptest.assert_almost_equal(cover1['coverage'].values, coverage1_shd)
    nptest.assert_almost_equal(mdrop, np.array([1,2,6,7,8,12]))



def test_cover_part_M():
    ds1, ds2, s1, e1, s2, e2 = gen_diff_cover_data(True, 0, 0)

    success, cover0, cover1, dcover, mdrop = \
        compare_temp_cov(ds1, ds2, s1, e1, s2, e2, 'M', 0.01)

    diff_shd = np.array([np.nan]*2 + [0]*3 + [np.nan]*3 + [0]*3 + [np.nan])
    coverage0_shd = np.array([np.nan]*2 + [1.0]*9 + [np.nan])
    coverage1_shd = np.array([1.]*5  + [np.nan]*3 + [1.0]*4)

    assert success == False
    nptest.assert_almost_equal(dcover['diff'].values, diff_shd)
    nptest.assert_almost_equal(cover0['coverage'].values, coverage0_shd)
    nptest.assert_almost_equal(cover1['coverage'].values, coverage1_shd)
    nptest.assert_almost_equal(mdrop, np.array([1,2,6,7,8,12]))



def test_drop_uncoverd_months():
    ds1, ds2, s1, e1, s2, e2 = gen_diff_cover_data(False, 0, 0)

    success, cover0, cover1, dcover, mdrop = \
        compare_temp_cov(ds1, ds2, s1, e1, s2, e2, 'D', 0.01)

    assert(success==False)
    nptest.assert_almost_equal(mdrop, np.array([1,2,6,7,8,12]))

    ds1 = drop_months_data(ds1, mdrop)
    ds2 = drop_months_data(ds2, mdrop)

    success, cover0, cover1, dcover, mdrop = \
        compare_temp_cov(ds1, ds2, s1, e1, s2, e2, 'D', 0.01)

    dropped_months0 = cover0.loc[np.isnan(cover0['coverage'])].index
    dropped_months1 = cover1.loc[np.isnan(cover1['coverage'])].index

    nptest.assert_almost_equal(dropped_months0, dropped_months1)

    assert(success==True)
    assert(mdrop.size==0)








if __name__ == '__main__':
    test_drop_uncoverd_months()
    test_cover_full_D()
    test_cover_full_M()
    test_cover_part_D()
    test_cover_part_M()
