# -*- coding: utf-8 -*-

""" Contains functions for testing break test class and functions"""
import pandas as pd
from pybreaks.break_test import TsRelBreakTest
from datetime import datetime

import numpy.testing as nptest
from tests.helper_functions import read_test_data, dict_depth, create_artificial_test_data


def test_conditions():
    ''' Test conditions for the calculation of test statistics'''

    df = read_test_data(431790)
    breaktime = datetime(2000,7,1)

    # will not raise error
    test = TsRelBreakTest(candidate=df['CCI_41_COMBINED'],
                          reference=df['merra2'], alpha=0.01,
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=5, test_check_spearR_sig=(0, 0.1),
                          bias_corr_method= 'linreg')

    assert(test.error_code_test == 0)

    # will raise no data error
    ccinan = pd.DataFrame(index=df.index, columns=['CCI_41_COMBINED'])
    test = TsRelBreakTest(candidate=ccinan,
                          reference=df['merra2'], alpha=0.01,
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=5, test_check_spearR_sig=(0, 0.1),
                          bias_corr_method='linreg')
    assert (test.error_code_test == 1)

    # will raise correlation error (R)
    test = TsRelBreakTest(candidate=df['CCI_41_COMBINED'],
                          reference=df['merra2'], alpha=0.01,
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=5, test_check_spearR_sig=(1, 1),
                          bias_corr_method='linreg')
    assert (test.error_code_test == 2)

    # will raise correlation error (p)
    test = TsRelBreakTest(candidate=df['CCI_41_COMBINED'],
                          reference=df['merra2'], alpha=0.01,
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=5, test_check_spearR_sig=(0, 0),
                          bias_corr_method='linreg')
    assert (test.error_code_test == 2)


    # will raise test_min_data error
    test = TsRelBreakTest(candidate=df['CCI_41_COMBINED'],
                          reference=df['merra2'], alpha=0.01,
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=10000, test_check_spearR_sig=(0, 1),
                          bias_corr_method='linreg')
    assert (test.error_code_test == 3)


def test_mean_break():
    ''' Test mean break detection'''

    df_mean, breaktime, timeframe = create_artificial_test_data('mean')

    test = TsRelBreakTest(candidate=df_mean.candidate,
                          reference=df_mean.reference,
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=3, test_check_spearR_sig=(-1, 1),
                          bias_corr_method= None, alpha=0.01)

    test.run_tests()
    testresults, error_dict, checkstats = test.get_results()

    assert(testresults['mean']['h'] == 1)
    assert(error_dict['error_code_test'] == 0)
    assert(checkstats['n0'] == 6)
    assert(checkstats['n1'] == 6)

    assert(test.check_test_results()[1] == 'mean')
    assert(test.check_test_results()[0] == True)



def test_var_break():
    '''Test var break detection'''

    df, breaktime, timeframe = create_artificial_test_data('var')

    test = TsRelBreakTest(candidate=df.candidate,
                          reference=df.reference,
                          breaktime=breaktime, test_resample=None,
                          test_check_min_data=3, test_check_spearR_sig=(-1, 1),
                          bias_corr_method= None, alpha=0.01)

    test.run_tests()
    testresults, error_dict, checkstats = test.get_results()

    assert(testresults['var']['h'] == 1)
    assert(error_dict['error_code_test'] == 0)
    assert(checkstats['n0'] == 182)
    assert(checkstats['n1'] == 184)
    nptest.assert_almost_equal(checkstats['frame_spearmanR'], 0.8944298, 4)
    nptest.assert_almost_equal(checkstats['frame_corrPval'], 0, 4)

    assert(test.check_test_results()[1] == 'var')
    assert(test.check_test_results()[0] == True)


def test_merge_results():
    '''Test function for merged results dict'''

    df = read_test_data(431790)
    breaktime = datetime(2000, 7, 1)

    test = TsRelBreakTest(candidate=df['CCI_41_COMBINED'],
                          reference=df['merra2'],
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=3, test_check_spearR_sig=(0, 0.1),
                          bias_corr_method='linreg', alpha=0.01)

    test.run_tests()

    results_flat = test.get_flat_results()
    testresults, error_dict, checkstats = test.get_results()

    assert(dict_depth(results_flat) == 1)
    assert(dict_depth(testresults) == 3)


def test_meta_dict():
    ''' Test creation of meta info dict'''
    df = read_test_data(431790)
    breaktime = datetime(2000, 7, 1)

    test = TsRelBreakTest(candidate=df['CCI_41_COMBINED'],
                          reference=df['merra2'],
                          breaktime=breaktime, test_resample=('M', 0.3),
                          test_check_min_data=3, test_check_spearR_sig=(0, 0.1),
                          bias_corr_method='linreg', alpha=0.01)

    meta = test.get_test_meta()

    assert(meta['0'] == 'No error occurred')
    assert(meta['1'] == 'No data for selected time frame')
    assert(meta['2'] == 'Spearman correlation failed')
    assert(meta['3'] == 'Min. observations N not reached')
    assert(meta['9'] == 'Unknown Error')


def test_validation_metrics():
    # the the function from the base class
    #todo: implement
    pass



if __name__ == '__main__' :
    test_conditions()
    test_mean_break()
    test_var_break()
    test_merge_results()
    test_meta_dict()

