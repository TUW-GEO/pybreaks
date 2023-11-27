 # -*- coding: utf-8 -*-


from tests.helper_functions import read_test_data, compare_metrics
import unittest
import sys
from pybreaks.break_adjust import TsRelBreakAdjust
from datetime import datetime, timedelta
from pybreaks.utils import dt_freq
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import matplotlib
matplotlib.use('Agg')


def compare_stats(group_stats, group_metrics, metrics_change):
    # some group stats
    np.testing.assert_almost_equal(group_stats['mean_CAN_group0'], 0.2882447935744, 3)
    np.testing.assert_almost_equal(group_stats['mean_CAN_group1'], 0.40933242, 3)
    np.testing.assert_almost_equal(group_stats['mean_REF_group0'], 0.330211267833, 3)
    np.testing.assert_almost_equal(group_stats['mean_REF_group1'], 0.3538935906, 3)
    np.testing.assert_almost_equal(group_stats['median_REF_group0'], 0.3231581285, 3)
    np.testing.assert_almost_equal(group_stats['median_REF_group1'], 0.36175487000, 3)
    np.testing.assert_almost_equal(group_stats['median_CAN_group0'], 0.296414265, 3)
    np.testing.assert_almost_equal(group_stats['median_CAN_group1'], 0.4234173049999, 3)
    np.testing.assert_almost_equal(group_stats['median_CAN_FRAME'], 0.38348515499, 3)
    np.testing.assert_almost_equal(group_stats['median_REF_FRAME'], 0.34560262847, 3)

    return True

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
        fig = self.src.adjust_obj.plot_adjustments()
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
        np.testing.assert_almost_equal(testresults_ifirst['zval_MEAN'], -7.5498487236321, decimal=1)
        np.testing.assert_almost_equal(testresults_ifirst['pval_MEAN'], 0.)
        assert testresults_ifirst['h_VAR'] == 0.
        np.testing.assert_almost_equal(testresults_ifirst['z_VAR'], 0.0083928, 5)
        np.testing.assert_almost_equal(testresults_ifirst['pval_VAR'], 0.927005, 5)
        assert testresults_ifirst['error_code_test'] == 0.
        assert testresults_ifirst['n0'] == 30
        assert testresults_ifirst['n1'] == 72
        np.testing.assert_almost_equal(testresults_ifirst['frame_spearmanR'], 0.589643, 5)
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

        np.testing.assert_almost_equal(m0['slope'], 0.49615028455, 3)
        np.testing.assert_almost_equal(m0['inter'], 0.12478766038, 3)
        np.testing.assert_almost_equal(m0['s02'], 0.0006490746439413, 5)
        np.testing.assert_almost_equal(m0['std_error'], 0.0543443828232, 3)
        np.testing.assert_almost_equal(m0['r_squared'], 0.748545649337, 3)
        np.testing.assert_almost_equal(m0['p_value'], 0.)
        np.testing.assert_almost_equal(m0['sum_squared_residuals'], 0.0181740900303, 3)
        np.testing.assert_almost_equal(m0['n_input'], testresults_ifirst['n0']) # both resampled

        np.testing.assert_almost_equal(m1['slope'], 0.45355634119, 3)
        np.testing.assert_almost_equal(m1['inter'],  0.24941273901, 3)
        np.testing.assert_almost_equal(m1['s02'], 0.000523507763, 3)
        np.testing.assert_almost_equal(m1['std_error'], 0.0365909511, 3)
        np.testing.assert_almost_equal(m1['r_squared'], 0.6870023018, 3)
        np.testing.assert_almost_equal(m1['p_value'], 0., 3)
        np.testing.assert_almost_equal(m1['sum_squared_residuals'], 0.03664554343, 3)
        np.testing.assert_almost_equal(m1['n_input'], testresults_ifirst['n1']) # both resampled

        # some group stats
        assert compare_stats(group_stats, group_metrics, metrics_change)
        assert compare_metrics(group_stats, group_metrics, metrics_change)

        # some checkstats
        np.testing.assert_almost_equal(checkstats['n0'], testresults_ifirst['n0']) # both resampled
        np.testing.assert_almost_equal(checkstats['n1'], testresults_ifirst['n1']) # both resampled
        assert checkstats['error_code_adjust'] == 0
        assert checkstats['error_code_test'] == 0
        assert checkstats['THRES_R_pearson'] == 0.5
        # todo: this does not match, why? one from M, other from D?
        # np.testing.assert_almost_equal(checkstats['can_bias_diff'],
        #     group_metrics['CAN_REF_mean_Diff_group1']-group_metrics['CAN_REF_mean_Diff_group0'])

@unittest.skipIf(sys.version[0]=='2', 'lmoments3 only available for python 3')
class Test_adjust_hom(unittest.TestCase):

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

        adjmodel_kwargs = dict(regress_resample=None, filter=None, poly_orders=[1,2],
                               select_by='R', cdf_types=None)

        kwargs = dict(candidate_flags=None, timeframe=timeframe, bias_corr_method='cdf_match',
                 adjust_tf_only=False, adjust_group=0, input_resolution='D',
                 adjust_check_pearsR_sig=(0.5, 0.01), adjust_check_fix_temp_coverage=False,
                 adjust_check_min_group_range=365, adjust_check_coverdiff_max=None,
                 adjust_check_ppcheck=(True, False), create_model_plots=True,
                 test_kwargs=test_kwargs, adjmodel_kwargs=adjmodel_kwargs)

        cls.src = TsRelBreakAdjust(cls.ts_full['candidate'], cls.ts_full['reference'],
            breaktime, adjustment_method='HOM', **kwargs)

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
        fig = self.src.adjust_obj.plot_adjustments()


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

        (res, freq) = dt_freq(self.src.adjust_obj.df_original.index)
        assert (res, freq) == (1., 'D')

        # before correction
        assert testresults_ifirst['h_MEAN'] == 1.
        np.testing.assert_almost_equal(testresults_ifirst['zval_MEAN'], -7.5498487236321, decimal=1)
        np.testing.assert_almost_equal(testresults_ifirst['pval_MEAN'], 0.)
        assert testresults_ifirst['h_VAR'] == 0.
        np.testing.assert_almost_equal(testresults_ifirst['z_VAR'], 0.00839, 3)
        np.testing.assert_almost_equal(testresults_ifirst['pval_VAR'], 0.92700, 3)
        assert testresults_ifirst['error_code_test'] == 0.
        assert testresults_ifirst['n0'] == 30
        assert testresults_ifirst['n1'] == 72
        np.testing.assert_almost_equal(testresults_ifirst['frame_spearmanR'], 0.589, 3)
        np.testing.assert_almost_equal(testresults_ifirst['frame_corrPval'], 0.)

        # after correction
        assert testresults_ilast['h_MEAN'] == 0. # break removed
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']
        assert testresults_ilast['error_code_test'] == testresults_ifirst['error_code_test']
        assert testresults_ilast['n0'] == testresults_ifirst['n0']
        assert testresults_ilast['n1'] == testresults_ifirst['n1']

        # there was only one iteration, therefore first == last
        for k, v in models_ifirst.items():
            np.testing.assert_almost_equal(models_ilast[k], v)

        # model parameters
        np.testing.assert_almost_equal(models_ifirst['poly_order'], 2)
        np.testing.assert_almost_equal(models_ifirst['coef_0'],1.237, 3)
        np.testing.assert_almost_equal(models_ifirst['coef_1'], -1.131, 3)
        np.testing.assert_almost_equal(models_ifirst['inter'], 0.1198, 3)
        np.testing.assert_almost_equal(models_ifirst['r2'], 0.543, 3)
        np.testing.assert_almost_equal(models_ifirst['filter_p'], np.nan)
        np.testing.assert_almost_equal(models_ifirst['sse'], 2.211, 3)
        np.testing.assert_almost_equal(models_ifirst['n_input'], 1542) # both resampled

        # some group stats
        assert compare_stats(group_stats, group_metrics, metrics_change)
        assert compare_metrics(group_stats, group_metrics, metrics_change)

        # some checkstats
        np.testing.assert_almost_equal(checkstats['n0'], testresults_ifirst['n0']) # both resampled
        np.testing.assert_almost_equal(checkstats['n1'], testresults_ifirst['n1']) # both resampled
        assert checkstats['error_code_adjust'] == 0
        assert checkstats['error_code_test'] == 0
        assert checkstats['THRES_R_pearson'] == 0.5
        # todo: this does not match, why? one from M, other from D?
        # np.testing.assert_almost_equal(checkstats['can_bias_diff'],
        #     group_metrics['CAN_REF_mean_Diff_group1']-group_metrics['CAN_REF_mean_Diff_group0'])



class Test_adjust_qcm(unittest.TestCase):

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

        adjmodel_kwargs = dict(categories=4, first_last='formula', fit='mean')

        kwargs = dict(candidate_flags=None, timeframe=timeframe, bias_corr_method='cdf_match',
                 adjust_tf_only=False, adjust_group=0, input_resolution='D',
                 adjust_check_pearsR_sig=(0.5, 0.01), adjust_check_fix_temp_coverage=False,
                 adjust_check_min_group_range=365, adjust_check_coverdiff_max=None,
                 adjust_check_ppcheck=(True, False), create_model_plots=True,
                 test_kwargs=test_kwargs, adjmodel_kwargs=adjmodel_kwargs)

        cls.src = TsRelBreakAdjust(cls.ts_full['candidate'], cls.ts_full['reference'],
            breaktime, adjustment_method='QCM', **kwargs)

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
        fig = self.src.adjust_obj.plot_adjustments()


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

        (res, freq) = dt_freq(self.src.adjust_obj.df_original.index)
        assert (res, freq) == (1., 'D')

        # before correction
        assert testresults_ifirst['h_MEAN'] == 1.
        np.testing.assert_almost_equal(testresults_ifirst['zval_MEAN'], -7.5498487236321, decimal=1)
        np.testing.assert_almost_equal(testresults_ifirst['pval_MEAN'], 0.)
        assert testresults_ifirst['h_VAR'] == 0.
        np.testing.assert_almost_equal(testresults_ifirst['z_VAR'], 0.00839, 3)
        np.testing.assert_almost_equal(testresults_ifirst['pval_VAR'], 0.9270057, 3)
        assert testresults_ifirst['error_code_test'] == 0.
        assert testresults_ifirst['n0'] == 30
        assert testresults_ifirst['n1'] == 72
        np.testing.assert_almost_equal(testresults_ifirst['frame_spearmanR'], 0.58956409632967, 3)
        np.testing.assert_almost_equal(testresults_ifirst['frame_corrPval'], 0., 3)

        # after correction
        assert testresults_ilast['h_MEAN'] == 0. # break removed
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']
        assert testresults_ilast['error_code_test'] == testresults_ifirst['error_code_test']
        assert testresults_ilast['n0'] == testresults_ifirst['n0']
        assert testresults_ilast['n1'] == testresults_ifirst['n1']

        # there was only one iteration, therefore first == last
        for k, v in models_ifirst['model0'].items():
            np.testing.assert_almost_equal(models_ilast['model0'][k], v)
        for k, v in models_ifirst['model1'].items():
            np.testing.assert_almost_equal(models_ilast['model1'][k], v)

        m0 = models_ifirst['model0']
        # model parameters
        np.testing.assert_almost_equal(m0['n_quantiles'], 4)
        np.testing.assert_almost_equal(m0[0.125], -0.042636302715681, 3)
        np.testing.assert_almost_equal(m0[1.0], 0.196129642521334, 3)

        m1 = models_ifirst['model1']
        np.testing.assert_almost_equal(m1['n_quantiles'], 4)
        np.testing.assert_almost_equal(m1[0.125], 0.059842589741188, 3)
        np.testing.assert_almost_equal(m1[1.0], 0.30814550359433, 3)

        assert compare_stats(group_stats, group_metrics, metrics_change)
        assert compare_metrics(group_stats, group_metrics, metrics_change)

        # some checkstats
        np.testing.assert_almost_equal(checkstats['n0'], testresults_ifirst['n0']) # both resampled
        np.testing.assert_almost_equal(checkstats['n1'], testresults_ifirst['n1']) # both resampled
        assert checkstats['error_code_adjust'] == 0
        assert checkstats['error_code_test'] == 0
        assert checkstats['THRES_R_pearson'] == 0.5
        # todo: this does not match, why? one from M, other from D?
        # np.testing.assert_almost_equal(checkstats['can_bias_diff'],
        #     group_metrics['CAN_REF_mean_Diff_group1']-group_metrics['CAN_REF_mean_Diff_group0'])

if __name__ == '__main__':
    tests = Test_adjust_lmp()
    tests.setUpClass()
    tests.setUp()
    tests.test_adjusted_data()
