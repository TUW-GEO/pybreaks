# -*- coding: utf-8 -*-

from pybreaks.break_adjust import TsRelBreakAdjust
from tests.helper_functions import create_artificial_test_data, read_test_data
import numpy.testing as nptest
import numpy as np

def test_artificial_break_adjust_no_resample_no_filter():

    df, breaktime = create_artificial_test_data('mean')

    adjmodel_kwargs = dict([('regress_resample', ('M',0.3)),
                            ('model_intercept', True),
                            ('filter', None)])
    adjfct_kwargs = {}

    obj = TsRelBreakAdjust(candidate=df['candidate'],
                           reference=df['reference'],
                           breaktime=breaktime, timeframe=None,
                           adjustment_method='LMP', input_resolution='D',
                           bias_corr_method=None, adjust_group=0,
                           adjust_tf_only=False, adjust_check_pearsR_sig=(0.7, 0.01),
                           adjust_check_fix_temp_coverage=False,
                           adjust_check_min_group_range=365, adjust_check_ppcheck=(True,True),
                           create_model_plots=True,
                           test_kwargs=None, adjmodel_kwargs=adjmodel_kwargs)

    obj.test_and_adjust(min_iter=None, max_iter=3, correct_below_0=True,
                       **adjfct_kwargs)

    isbreak, breaktype, testresult, testerror, _ = obj.run_tests()

    assert(isbreak==True)
    assert(breaktype=='mean')
    assert(testerror==0)

    data_adjusted, stillbreak, adjerror = \
        obj.test_and_adjust(min_iter=0, max_iter=1, correct_below_0=True,
                            corrections_from_core=True, resample_corrections=True,
                            interpolation_method='linear')


    nptest.assert_almost_equal(data_adjusted['candidate_adjusted'].values,
                               np.array([50.,51.] * 183))

    assert(stillbreak is False)
    assert(adjerror==0)

    test_results_b4, models_iter0, test_results_aft, models_last, \
    group_stats, vertical_metrics, hor_error = obj.get_results()



def test_real_PR_adjust_no_resample_no_filter():

    df, breaktime = read_test_data(431790)

    obj = TsRelBreakAdjust(candidate=df['candidate'],
                           reference=df['reference'],
                           breaktime=breaktime, timeframe=None,
                           regress_resample=None, bias_corr_method=None,
                           n_quantiles=None, test_resample=('M', 0.3),
                           adjustment_method='PairRegress',
                           test_min_data=5, test_spearR_sig=(0, 0.01),
                           filter=None, adjust_check_pearsR_sig=(0, 0.01),
                           adjust_group=0, create_model_plots=True,
                           filter_iter_increase=None, n_plots=3)

    isbreak, breaktype, testresult, testerror, _ = obj.run_tests()

    assert(isbreak==True)
    assert(breaktype=='mean')
    assert(testerror==0)

    data_adjusted, stillbreak, adjerror = \
        obj.test_and_adjust(min_iter=0, max_iter=1, correct_below_0=True,
                            interpolation_method=None, adjust_tf_only=True)


    nptest.assert_almost_equal(data_adjusted['candidate_adjusted'].values,
                               np.array([50.,51.] * 183))

    assert(stillbreak is False)
    assert(adjerror==0)


def test_real_PR_adjust_input_check_fails():
    '''
    Select a point where the input checks fails and no adjustment is done

    Returns
    -------

    '''


def test_read_PR_adjust_output_check_fails():
    '''
    Select a point where the output checks fails and no adjustmetn is done.

    Returns
    -------

    '''
if __name__ == '__main__':
    test_artificial_break_adjust_no_resample_no_filter()
