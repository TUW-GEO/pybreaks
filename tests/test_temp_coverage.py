# -*- coding: utf-8 -*-

import numpy as np
from pybreaks.temp_coverage import compare_temp_cov, drop_months_data, count_M, count_D
import numpy.testing as nptest
from tests.helper_functions import gen_full_data, gen_diff_cover_data

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

def test_count_D():
    # no missing days
    ds1, _, start1, end1, _, _ = gen_full_data(False, rand1=0.)
    df_count = count_D(ds1.index, start1, end1)
    assert all(df_count['coverage'] == 1.)
    np.testing.assert_equal(df_count['days_is'].values, df_count['days_max'].values)

    # some missing days
    ds1, _, start1, end1, _, _ = gen_full_data(False, rand1=0.5)
    df_count = count_D(ds1.index, start1, end1)
    assert not all(df_count['coverage'] == 1.)
    assert any(np.not_equal(df_count['days_is'].values, df_count['days_max'].values))

def test_count_M():
    # no missing months
    ds1, _, start1, end1, _, _ = gen_full_data(True, rand1=0.)
    df_count = count_M(ds1.index, start1, end1)
    assert all(df_count['coverage'] == 1.)
    np.testing.assert_equal(df_count['months_is'].values, df_count['months_max'].values)

    # some missing days
    ds1, _, start1, end1, _, _ = gen_full_data(True, rand1=0.1)
    df_count = count_M(ds1.index, start1, end1)
    assert not all(df_count['coverage'] == 1.)
    assert any(np.not_equal(df_count['months_is'].values, df_count['months_max'].values))

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

    nptest.assert_equal(dropped_months0.values, dropped_months1.values)

    assert(success==True)
    assert(mdrop.size==0)


if __name__ == '__main__':
    test_drop_uncoverd_months()
    test_cover_full_D()
    test_cover_full_M()
    test_cover_part_D()
    test_cover_part_M()
    test_count_D()
    test_count_M()
