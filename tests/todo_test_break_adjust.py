# # -*- coding: utf-8 -*-
#
# from pybreaks.break_adjust import TsRelBreakAdjust
# from tests.helper_functions import create_artificial_test_data, read_test_data
# import numpy.testing as nptest
# import numpy as np
#
# def test_artificial_break_adjust_no_resample_no_filter():
#
#     df, breaktime = create_artificial_test_data('mean')
#
#     adjmodel_kwargs = dict([('regress_resample', ('M',0.3)),
#                             ('model_intercept', True),
#                             ('filter', None)])
#     adjfct_kwargs = {}
#
#     obj = TsRelBreakAdjust(candidate=df['candidate'],
#                            reference=df['reference'],
#                            breaktime=breaktime, timeframe=None,
#                            adjustment_method='LMP', input_resolution='D',
#                            bias_corr_method=None, adjust_group=0,
#                            adjust_tf_only=False, adjust_check_pearsR_sig=(0.7, 0.01),
#                            adjust_check_fix_temp_coverage=False,
#                            adjust_check_min_group_range=365, adjust_check_ppcheck=(True,True),
#                            create_model_plots=True,
#                            test_kwargs=None, adjmodel_kwargs=adjmodel_kwargs)
#
#     obj.test_and_adjust(min_iter=None, max_iter=3, correct_below_0=True,
#                        **adjfct_kwargs)
#
#     isbreak, breaktype, testresult, testerror, _ = obj.run_tests()
#
#     assert(isbreak==True)
#     assert(breaktype=='mean')
#     assert(testerror==0)
#
#     data_adjusted, stillbreak, adjerror = \
#         obj.test_and_adjust(min_iter=0, max_iter=1, correct_below_0=True,
#                             corrections_from_core=True, resample_corrections=True,
#                             interpolation_method='linear')
#
#
#     nptest.assert_almost_equal(data_adjusted['candidate_adjusted'].values,
#                                np.array([50.,51.] * 183))
#
#     assert(stillbreak is False)
#     assert(adjerror==0)
#
#     test_results_b4, models_iter0, test_results_aft, models_last, \
#     group_stats, vertical_metrics, hor_error = obj.get_results()
#
#
#
# def test_real_PR_adjust_no_resample_no_filter():
#
#     df, breaktime = read_test_data(431790)
#
#     obj = TsRelBreakAdjust(candidate=df['candidate'],
#                            reference=df['reference'],
#                            breaktime=breaktime, timeframe=None,
#                            regress_resample=None, bias_corr_method=None,
#                            n_quantiles=None, test_resample=('M', 0.3),
#                            adjustment_method='PairRegress',
#                            test_min_data=5, test_spearR_sig=(0, 0.01),
#                            filter=None, adjust_check_pearsR_sig=(0, 0.01),
#                            adjust_group=0, create_model_plots=True,
#                            filter_iter_increase=None, n_plots=3)
#
#     isbreak, breaktype, testresult, testerror, _ = obj.run_tests()
#
#     assert(isbreak==True)
#     assert(breaktype=='mean')
#     assert(testerror==0)
#
#     data_adjusted, stillbreak, adjerror = \
#         obj.test_and_adjust(min_iter=0, max_iter=1, correct_below_0=True,
#                             interpolation_method=None, adjust_tf_only=True)
#
#
#     nptest.assert_almost_equal(data_adjusted['candidate_adjusted'].values,
#                                np.array([50.,51.] * 183))
#
#     assert(stillbreak is False)
#     assert(adjerror==0)
#
#
# def test_real_PR_adjust_input_check_fails():
#     '''
#     Select a point where the input checks fails and no adjustment is done
#
#     Returns
#     -------
#
#     '''
#
#
# def test_read_PR_adjust_output_check_fails():
#     '''
#     Select a point where the output checks fails and no adjustmetn is done.
#
#     Returns
#     -------
#
#     '''
# if __name__ == '__main__':
#     test_artificial_break_adjust_no_resample_no_filter()


from tests.helper_functions import read_test_data, create_artificial_test_data
import unittest
from pybreaks.break_adjust import TsRelBreakAdjust
from datetime import datetime, timedelta
from pybreaks.utils import dt_freq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

class Test_adjust_lmp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts = read_test_data(707393)
        cls.ts_full = ts.rename(columns={'CCI_44_COMBINED': 'candidate',
                                'MERRA2': 'reference'}).loc['2007-01-01':]
        breaktime = datetime(2012,7,1)
        timeframe = np.array([datetime(2010,1,15), datetime(2018,6,30)])
        # introduce a mean break of 0.1
        cls.ts_full.loc[breaktime:, 'candidate'] += 0.1

        cls.ts_frame = cls.ts_full.loc[timeframe[0]:timeframe[1]]

        test_kwargs = dict(test_resample=('M', 0.3),
                           test_check_min_data=3, test_check_spearR_sig=(0, 0.05),
                           alpha=0.01)

        adjmodel_kwargs = dict(filter=None, model_intercept=True, regress_resample=('M', 0.3))

        kwargs = dict(candidate_flags=None, timeframe=timeframe, bias_corr_method='cdf_match',
                 adjust_tf_only=False, adjust_group=0, input_resolution='D',
                 adjust_check_pearsR_sig=(0.5, 0.01), adjust_check_fix_temp_coverage=False,
                 adjust_check_min_group_range=365, adjust_check_coverdiff_max=None,
                 adjust_check_ppcheck=(True, False), create_model_plots=True,
                 test_kwargs=test_kwargs, adjmodel_kwargs=adjmodel_kwargs)

        cls.src = TsRelBreakAdjust(cls.ts_full['candidate'], cls.ts_full['reference'],
            breaktime, adjustment_method='LMP', **kwargs)

    def setUp(self):
        (res, freq) = dt_freq(self.src.df_original.index)
        assert (res, freq) == (1., 'D')
        self.src.test_and_adjust()


    def tearDown(self):
        plt.close('all')

    def test_plots(self):
        fig = self.src.plot_stats_ts(self.src.df_original, kind='line', stats=True)
        fig = self.src.plot_ts(self.src.df_frame)
        fig = self.src.plot_coll_fig()
        fig = self.src.plot_adj_ts()

    def test_adjusted_data(self):
        """
        The model is calculated from daily values.
        Corrections are derived for monthly resampled values and then interpolated
        to the target daily resolution of the values to adjust.
        """
        testresults_ifirst, models_ifirst, testresults_ilast, \
        models_ilast, group_stats, group_metrics, metrics_change, \
        checkstats = self.src.get_results()

        assert self.src.testresult['mean']['stats']['zval'] == testresults_ilast['zval_MEAN']

        (res, freq) = dt_freq(self.src.adjust_obj.df_adjust.index)
        assert (res, freq) == (1., 'M')

        # before correction
        assert testresults_ifirst['h_MEAN'] == 1.
        np.testing.assert_almost_equal(testresults_ifirst['zval_MEAN'], -7.5498487236321)
        np.testing.assert_almost_equal(testresults_ifirst['pval_MEAN'], 0.)
        assert testresults_ifirst['h_VAR'] == 0.
        np.testing.assert_almost_equal(testresults_ifirst['z_VAR'], 0.012002612)
        np.testing.assert_almost_equal(testresults_ifirst['pval_VAR'], 0.9127611636)
        assert testresults_ifirst['error_code_test'] == 0.
        assert testresults_ifirst['n0'] == 30
        assert testresults_ifirst['n1'] == 72
        np.testing.assert_almost_equal(testresults_ifirst['frame_spearmanR'], 0.58956409632967)
        np.testing.assert_almost_equal(testresults_ifirst['frame_corrPval'], 0.)

        # after correction
        assert testresults_ilast['h_MEAN'] == 0. # break removed
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']
        assert testresults_ilast['error_code_test'] == testresults_ifirst['error_code_test']
        assert testresults_ilast['n0'] == testresults_ifirst['n0']
        assert testresults_ilast['n1'] == testresults_ifirst['n1']

        # there was only one iteration, therefore first == last
        assert models_ifirst == models_ilast

        # model parameters
        m0 = models_ifirst['model0']
        m1 = models_ifirst['model1']

        np.testing.assert_almost_equal(m0['slope'], 0.49615028455)
        np.testing.assert_almost_equal(m0['inter'], 0.12478766038)
        np.testing.assert_almost_equal(m0['s02'], 0.0006490746439413)
        np.testing.assert_almost_equal(m0['std_error'], 0.0543443828232)
        np.testing.assert_almost_equal(m0['r_squared'], 0.748545649337)
        np.testing.assert_almost_equal(m0['p_value'], 0.)
        np.testing.assert_almost_equal(m0['sum_squared_residuals'], 0.0181740900303)
        np.testing.assert_almost_equal(m0['n_input'], testresults_ifirst['n0']) # both resampled

        np.testing.assert_almost_equal(m1['slope'], 0.45355634119)
        np.testing.assert_almost_equal(m1['inter'],  0.24941273901)
        np.testing.assert_almost_equal(m1['s02'], 0.000523507763)
        np.testing.assert_almost_equal(m1['std_error'], 0.0365909511)
        np.testing.assert_almost_equal(m1['r_squared'], 0.6870023018)
        np.testing.assert_almost_equal(m1['p_value'], 0.)
        np.testing.assert_almost_equal(m1['sum_squared_residuals'], 0.03664554343)
        np.testing.assert_almost_equal(m1['n_input'], testresults_ifirst['n1']) # both resampled

        # some group stats
        np.testing.assert_almost_equal(group_stats['mean_CAN_group0'], 0.2882447935744)
        np.testing.assert_almost_equal(group_stats['mean_CAN_group1'], 0.40933242)
        np.testing.assert_almost_equal(group_stats['mean_REF_group0'], 0.330211267833)
        np.testing.assert_almost_equal(group_stats['mean_REF_group1'], 0.3538935906)
        np.testing.assert_almost_equal(group_stats['median_REF_group0'], 0.3231581285)
        np.testing.assert_almost_equal(group_stats['median_REF_group1'], 0.36175487000)
        np.testing.assert_almost_equal(group_stats['median_CAN_group0'], 0.296414265)
        np.testing.assert_almost_equal(group_stats['median_CAN_group1'], 0.4234173049999)
        np.testing.assert_almost_equal(group_stats['median_CAN_FRAME'], 0.38348515499)
        np.testing.assert_almost_equal(group_stats['median_REF_FRAME'], 0.34560262847)
        # some group metrics
        should = group_stats['mean_CAN_group0'] - group_stats['mean_REF_group0']
        np.testing.assert_almost_equal(group_metrics['CAN_REF_mean_Diff_group0'], should)
        should = group_stats['median_CAN_group0'] - group_stats['median_REF_group0']
        np.testing.assert_almost_equal(group_metrics['CAN_REF_median_Diff_group0'], should)
        should = group_stats['median_CAN_group0'] - group_stats['median_REF_group0']
        np.testing.assert_almost_equal(group_metrics['CAN_REF_median_Diff_group0'], should)

