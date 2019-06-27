# -*- coding: utf-8 -*-

'''
Contains functions for testing the base class for Break Detection and Adjustment
'''
from breakadjustment.base import TsRelBreakBase
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import numpy.testing as nptest
from helper_funcions import read_test_data, create_test_data


def real_base(bias_corr_method='linreg'):
    '''
    Create a base object with real data
    Parameters
    -------
    bias_corr_method : str or None
        Name of bias corr method as in pytesmo or None to ignore
    '''
    ts, breaktime = read_test_data(431790)
    base = TsRelBreakBase(ts[['CCI_41_COMBINED']], ts[['merra2']], breaktime, bias_corr_method)
    return base


def test_base(type):
    '''Create a base object with test data'''
    df, breaktime = create_test_data(type)

    if type == 'mean':
        return TsRelBreakBase(df.candidate, df.reference, breaktime, None)
    if type == 'var':
        return TsRelBreakBase(df.candidate, df.reference, breaktime, None)
    if type == 'const':
        return TsRelBreakBase(df.candidate, df.reference, breaktime, None)
    if type == 'empty':
        return TsRelBreakBase(df.candidate, df.reference, breaktime, None)


########################################


def test_empty_base():
    ''' Test data where 1 set is missing (avoid exceptions)'''
    base = test_base('empty')

    group_data = base.get_group_data(0, base.df_original, 'all')
    assert(group_data.candidate.dropna().empty)
    assert(group_data.reference.index.size == 182) # data at break time belongs to G0

    group_data = base.get_group_data(1, base.df_original, 'all')
    assert(group_data.candidate.dropna().empty)
    assert(group_data.reference.index.size == 184) # data at break time belongs to G0

    assert(base.df_original.candidate.dropna().empty == True)
    assert(base.df_original.reference.index.size == 182+184)

def test_real():
    ''' Test real CCI SM and Merra2 SM data'''
    original_data, breaktime = read_test_data(431790)
    base = real_base()
    start = datetime(1998,1,1)
    end = datetime(2007,1,1)
    ndays = (end - start).days + 1
    assert(base.df_original.dropna(how='all').index.size == ndays)

    # because bias is corrected in merra:
    assert(all(base.df_original['CCI_41_COMBINED'].dropna() == original_data['CCI_41_COMBINED'].dropna()))
    assert(all(base.df_original['merra2'].dropna() != original_data['merra2'].dropna()))

def test_real_no_bias_corr():
    '''Test real data without bias correction'''
    original_data, breaktime = read_test_data(431790)
    base = real_base(bias_corr_method=None)
    start = datetime(1998,1,1)
    end = datetime(2007,1,1)
    ndays = (end - start).days + 1
    assert(base.df_original.dropna(how='all').index.size == ndays)

    # because bias is corrected in merra:
    assert(all(base.df_original['CCI_41_COMBINED'].dropna() == original_data['CCI_41_COMBINED'].dropna()))
    assert(all(base.df_original['merra2'].dropna() == original_data['merra2'].dropna()))


def test_get_group_data():
    ''' Test getter for group data in data frame'''
    base = test_base('const')
    group_data = base.get_group_data(0, base.df_original, ['candidate', 'reference'])
    assert(len(group_data.index)==182)
    group_data = base.get_group_data(1, base.df_original, 'all')
    assert(len(group_data.index)==184)


def test_get_ts_stats():
    # todo: implement this for the new function
    return
    ''' Test groups stats and stats comparison function'''
    base = test_base('const')
    ts_group_stats, group_comparison_stats, ts_comparison_stats =\
        base.get_ts_stats(base.df_original, as_dict=True, digits=4)

    # group stats

    nptest.assert_almost_equal(ts_group_stats['min_reference_group1'], 0.5, 3)
    nptest.assert_almost_equal(ts_group_stats['min_reference_group0'], 0.5, 3)
    nptest.assert_almost_equal(ts_group_stats['min_candidate_group0'], 0.1, 3)
    nptest.assert_almost_equal(ts_group_stats['min_candidate_group1'], 0.9, 3)

    nptest.assert_almost_equal(ts_group_stats['mean_reference_group1'], 0.5, 3)
    nptest.assert_almost_equal(ts_group_stats['mean_reference_group0'], 0.5, 3)
    nptest.assert_almost_equal(ts_group_stats['mean_candidate_group0'], 0.1, 3)
    nptest.assert_almost_equal(ts_group_stats['mean_candidate_group1'], 0.9, 3)

    nptest.assert_almost_equal(ts_group_stats['var_reference_group1'], 0, 3)
    nptest.assert_almost_equal(ts_group_stats['var_reference_group0'], 0, 3)
    nptest.assert_almost_equal(ts_group_stats['var_candidate_group0'], 0, 3)
    nptest.assert_almost_equal(ts_group_stats['var_candidate_group1'], 0, 3)

    nptest.assert_almost_equal(ts_group_stats['median_reference_group1'], 0.5, 3)
    nptest.assert_almost_equal(ts_group_stats['median_reference_group0'], 0.5, 3)
    nptest.assert_almost_equal(ts_group_stats['median_candidate_group0'], 0.1, 3)
    nptest.assert_almost_equal(ts_group_stats['median_candidate_group1'], 0.9, 3)

    nptest.assert_almost_equal(ts_group_stats['iqr_reference_group1'], 0, 3)
    nptest.assert_almost_equal(ts_group_stats['iqr_reference_group0'], 0, 3)
    nptest.assert_almost_equal(ts_group_stats['iqr_candidate_group0'], 0, 3)
    nptest.assert_almost_equal(ts_group_stats['iqr_candidate_group1'], 0, 3)

    nptest.assert_almost_equal(ts_group_stats['RMSD_candidate_reference_group0'], 0.4, 3)
    nptest.assert_almost_equal(ts_group_stats['RMSD_candidate_reference_group1'], 0.4, 3)

    # group comparsion stats
    nptest.assert_almost_equal(group_comparison_stats['MinDiff_reference_group0_minus_group1'], 0, 3)
    nptest.assert_almost_equal(group_comparison_stats['MinDiff_candidate_group0_minus_group1'], -0.8, 3)
    nptest.assert_almost_equal(group_comparison_stats['MeanDiff_reference_group0_minus_group1'], 0, 3)
    nptest.assert_almost_equal(group_comparison_stats['MeanDiff_candidate_group0_minus_group1'], -0.8, 3)
    nptest.assert_almost_equal(group_comparison_stats['VarRatio_candidate_group0_over_group1'], 0.0069, 4)
    nptest.assert_almost_equal(group_comparison_stats['VarRatio_reference_group0_over_group1'], np.inf, 3)
    nptest.assert_almost_equal(group_comparison_stats['IQRRatio_reference_group0_over_group1'], np.inf, 3)
    nptest.assert_almost_equal(group_comparison_stats['IQRRatio_reference_group0_over_group1'], np.inf, 3)
    nptest.assert_almost_equal(group_comparison_stats['RMSDChange_candidate_group0_minus_group1'], 0,  3)


    # time series comparison stats
    nptest.assert_almost_equal(ts_comparison_stats['IQR_ratio_candidate_reference_group0'], np.inf, 3)
    nptest.assert_almost_equal(ts_comparison_stats['IQR_ratio_candidate_reference_group1'], np.inf, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Min_diff_candidate_reference_group1'], 0.4, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Min_diff_candidate_reference_group0'], -0.4, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Mean_diff_candidate_reference_group1'], 0.4, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Mean_diff_candidate_reference_group0'], -0.4, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Variance_ratio_candidate_reference_group0'], np.inf, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Variance_ratio_candidate_reference_group1'], np.inf, 3)
    nptest.assert_almost_equal(ts_comparison_stats['Median_diff_candidate_reference_group1'], 0.4,  3)
    nptest.assert_almost_equal(ts_comparison_stats['Median_diff_candidate_reference_group0'], -0.4,  3)

    # as dataframe
    ts_group_stats, group_comparison_stats, ts_comparison_stats = base.get_ts_stats(base.df_original, as_dict=False, digits=4)
    assert(isinstance(ts_group_stats, pd.DataFrame))
    assert(isinstance(group_comparison_stats, pd.DataFrame))

def test_calc_diff():
    ''' Test difference function between candidate and reference'''
    base = test_base('const')
    base.df_original['Q'] = base.calc_diff(base.df_original)
    assert(all(base.get_group_data(0, base.df_original, 'Q') == -0.4))
    assert(all(base.get_group_data(1, base.df_original, 'Q') == 0.4))

def test_calc_bias():
    ''' Test bias calculation function'''
    base = test_base('const')
    g0_data = base.get_group_data(0, base.df_original, 'all')
    g1_data = base.get_group_data(1, base.df_original, 'all')

    bias0 = base._calc_RMSD(g0_data, 'candidate', 'reference')
    bias1 = base._calc_RMSD(g1_data, 'candidate', 'reference')

    nptest.assert_almost_equal(bias0, 0.4)
    nptest.assert_almost_equal(bias1, 0.4)


def test_calc_resiudals_autocorr():
    '''
    Test the calculation of the autocorrelation function for the single
    regression model residuals.
    '''

    base = test_base('mean')
    autocorr = base.calc_residuals_auto_corr(lags=range(30))

    # MEAN BREAK
    assert(autocorr.iloc[0] == 1.) # first value of autocorrelation always 1
    # autocorrelation for a mean break is not affected, therefore linear
    for i in range(autocorr.index[-1]):
        assert(autocorr.iloc[i] > autocorr.iloc[i+1])

    base = test_base('var')     # VAR BREAK
    autocorr = base.calc_residuals_auto_corr(lags=range(30))
    assert(autocorr.iloc[0] == 1.)

    # autocorrelation for a var break is affected, signs have to alternate
    for i in range(autocorr.index[-2]):
        assert(np.sign(autocorr.iloc[i]) == np.sign(autocorr.iloc[i+2]))
        if i % 2 == 0:
            assert(autocorr.iloc[i] > autocorr.iloc[i+2])
        else:
            assert(autocorr.iloc[i] < autocorr.iloc[i+2])


def test_sse_around_breaktime():
    ''' Test calculation of SSE around the break time'''

    base = test_base('mean')
    sse, min_date = base.sse_around_breaktime(base.df_original,
                                              n_margin=10)

    assert(min_date== datetime(2000,6,30))

    base = test_base('var')
    sse, min_date = base.sse_around_breaktime(base.df_original,
                                              n_margin=10)

    assert(min_date== datetime(2000,6,30))


def test_validation_stats():
    raise NotImplementedError


if __name__ == '__main__':
    test_empty_base()
    test_real()
    test_real_no_bias_corr()
    test_get_group_data()
    test_get_ts_stats()
    test_calc_diff()
    test_calc_bias()
    test_calc_resiudals_autocorr()
    test_sse_around_breaktime()
