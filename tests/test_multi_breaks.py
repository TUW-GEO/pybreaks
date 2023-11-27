# -*- coding: utf-8 -*-

'''
Test the MultiBreakAdjustment module.
'''
import pytest

# TODO:
#   (+) Class for testing only
#---------
# NOTES:
#   -

from tests.helper_functions import read_test_data, compare_metrics
import unittest
import sys
from pybreaks.break_multi import TsRelMultiBreak
from datetime import datetime, timedelta
from pybreaks.utils import dt_freq
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import tempfile
import shutil

class Test_multibreak_adjust_lmp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts = read_test_data(707393)
        ts_full = ts.rename(columns={'CCI_44_COMBINED': 'candidate',
                                     'MERRA2': 'reference'}).loc['2007-01-01':].copy(True)
        ts_full['candidate_original'] = ts_full['candidate'] # keep original
        # introduce some breaks to correct
        cls.breaktimes = np.array([datetime(2012,7,1), datetime(2010,1,15)])
        can_biased = ts_full.loc[:, 'candidate'].copy(True)
        can_biased[cls.breaktimes[0]:] += 0.1 # first break
        can_biased.loc[:cls.breaktimes[1]] -= 0.1 # second break
        ts_full.loc[:, 'candidate'] = can_biased
        ts_full['flags'] = 0. # all are good in the example

        test_kwargs = dict([('test_resample', ('M', 0.3)),
                            ('mean_test', 'wilkoxon'),
                            ('var_test', 'scipy_fligner_killeen'),
                            ('alpha', 0.01),
                            ('test_check_min_data', 5),
                            ('test_check_spearR_sig', [0., 1.])])

        adjmodel_kwargs = dict([('regress_resample', ('M', 0.3)),
                                ('filter', None),
                                ('model_intercept', True)])
        adjfct_kwargs = {'corrections_from_core': True,
                         'values_to_adjust_freq': 'D',
                         'resample_corrections': True,
                         'interpolation_method': 'linear'}

        adjcheck_kwargs = {'adjust_check_fix_temp_coverage': False,
                           'adjust_check_min_group_range': 365,
                           'adjust_check_pearsR_sig': (0., 1.)}

        cls.ts_full = ts_full.copy(True)

        cls.src = TsRelMultiBreak(candidate=cls.ts_full['candidate'],
                             reference=cls.ts_full['reference'],
                             breaktimes=cls.breaktimes,
                             adjustment_method='LMP',
                             candidate_flags=(cls.ts_full['flags'], [0]),
                             full_period_bias_corr_method='cdf_match',
                             sub_period_bias_corr_method='linreg',
                             base_breaktime=None,
                             HSP_init_breaktest=True,
                             models_from_hsp=True,
                             adjust_within='breaks',
                             input_resolution='D',
                             test_kwargs=test_kwargs,
                             adjmodel_kwargs=adjmodel_kwargs,
                             adjcheck_kwargs=adjcheck_kwargs,
                             create_model_plots=True,
                             frame_ts_figure=True,
                             frame_tsstats_plots=True)

        (res, freq) = dt_freq(cls.src.df_original.index)
        assert (res, freq) == (1., 'D')
        cls.candidate_adjusted = cls.src.adjust_all(extended_reference=True,
                                                    **adjfct_kwargs)
        assert cls.src.candidate_has_changed()

    def setUp(self):
        pass

    def tearDown(self):
        plt.close('all')

    @pytest.mark.skip(reason="currently ignoring plots because code uses deprecated functions and needs to be updated")
    def test_plots(self):
        plot_path = tempfile.mkdtemp()
        self.src.plot_adjustment_ts_full(plot_path, prefix='testLMP')
        self.src.plot_frame_ts_figure(plot_path, prefix='testLMP')
        #todo: why is this failing?
        #self.src.plot_tsstats_figures(plot_path, prefix='testLMP')
        self.src.plot_models_figures(plot_path, prefix='testLMP')
        shutil.rmtree(plot_path)

    def test_init_testresults(self):
        """
        Test results that are used to create the adjustment frames from,
        if the HSP_init_breaktest option is selected.
        """
        init_res_1 = self.src.init_test_results[self.breaktimes[0]] #2012
        init_res_2 = self.src.init_test_results[self.breaktimes[1]] #2010

        # 2012
        assert init_res_1['error_code_test'] == 0.
        assert init_res_1['h_MEAN'] == 1
        assert init_res_1['h_VAR'] == 0
        assert init_res_1['n0'] == 30
        assert init_res_1['n1'] == 72
        #2010
        assert init_res_2['error_code_test'] == 0.
        assert init_res_2['h_MEAN'] == 1
        assert init_res_2['h_VAR'] == 0
        assert init_res_2['n0'] == 36
        assert init_res_2['n1'] == init_res_1['n0']

    def test_lmp_results(self):
        testresults_ifirst, testresults_ilast, models_ifirst, models_ilast, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2012,7,1))

        assert compare_metrics(group_stats, group_metrics, metrics_change)
        testresults_ifirst, testresults_ilast, models_ifirst, models_ilast, \
        group_stats, group_metrics, metrics_change, checkstats = \
        self.src.get_results(breaktime=datetime(2010, 1, 15))

        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ilast['h_MEAN'] == 0

@unittest.skipIf(sys.version[0]=='2', 'lmoments3 only available for python 3')
class Test_multibreak_adjust_hom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ts = read_test_data(707393)
        ts_full = ts.rename(columns={'CCI_44_COMBINED': 'candidate',
                                     'MERRA2': 'reference'}).loc['2007-01-01':].copy(True)
        ts_full['candidate_original'] = ts_full['candidate'] # keep original
        # introduce some breaks to correct
        cls.breaktimes = np.array([datetime(2012,7,1), datetime(2010,1,15)])
        can_biased = ts_full.loc[:, 'candidate'].copy(True)
        can_biased[cls.breaktimes[0]:] += 0.1 # first break
        can_biased.loc[:cls.breaktimes[1]] -= 0.1 # second break
        ts_full.loc[:, 'candidate'] = can_biased
        ts_full['flags'] = 0. # all are good in the example

        test_kwargs = dict([('test_resample', ('M', 0.3)),
                            ('mean_test', 'wilkoxon'),
                            ('var_test', 'scipy_fligner_killeen'),
                            ('alpha', 0.01),
                            ('test_check_min_data', 5),
                            ('test_check_spearR_sig', [0., 1.])])

        adjmodel_kwargs = dict([('regress_resample', ('M', 0.3)),
                                ('filter', None),
                                ('poly_orders', [1, 2]),
                                ('select_by', 'R'),
                                ('cdf_types', None)])

        adjfct_kwargs = {'alpha': 0.4,
                         'use_separate_cdf': False,
                         'from_bins': False}

        adjcheck_kwargs = {'adjust_check_fix_temp_coverage': False,
                           'adjust_check_min_group_range': 365,
                           'adjust_check_pearsR_sig': (0., 1.)}

        cls.ts_full = ts_full.copy(True)

        cls.src = TsRelMultiBreak(candidate=cls.ts_full['candidate'],
                              reference=cls.ts_full['reference'],
                              breaktimes=cls.breaktimes,
                              adjustment_method='HOM',
                              candidate_flags=(cls.ts_full['flags'], [0]),
                              full_period_bias_corr_method='cdf_match',
                              sub_period_bias_corr_method='linreg',
                              base_breaktime=None,
                              HSP_init_breaktest=True,
                              models_from_hsp=True,
                              adjust_within='breaks',
                              input_resolution='D',
                              test_kwargs=test_kwargs,
                              adjmodel_kwargs=adjmodel_kwargs,
                              adjcheck_kwargs=adjcheck_kwargs,
                              create_model_plots=False,
                              frame_ts_figure=True,
                              frame_tsstats_plots=True)

        (res, freq) = dt_freq(cls.src.df_original.index)
        assert (res, freq) == (1., 'D')
        cls.candidate_adjusted = cls.src.adjust_all(**adjfct_kwargs)
        assert cls.src.candidate_has_changed()

    def setUp(self):
        pass

    def tearDown(self):
        plt.close('all')

    @pytest.mark.skip(reason="currently ignoring plots because code uses deprecated functions and needs to be updated")
    def test_plots(self):
        plot_path = tempfile.mkdtemp()
        self.src.plot_adjustment_ts_full(plot_path, prefix='testHOM')
        self.src.plot_frame_ts_figure(plot_path, prefix='testHOM')
        # todo: why is this failing?
        #self.src.plot_tsstats_figures(plot_path, prefix='testHOM')
        self.src.plot_models_figures(plot_path, prefix='testHOM')
        shutil.rmtree(plot_path)

    def test_init_testresults(self):
        """
        Test results that are used to create the adjustment frames from,
        if the HSP_init_breaktest option is selected.
        """
        init_res_1 = self.src.init_test_results[self.breaktimes[0]]
        init_res_2 = self.src.init_test_results[self.breaktimes[1]]

        # 2012
        assert init_res_1['error_code_test'] == 0.
        assert init_res_1['h_MEAN'] == 1
        assert init_res_1['h_VAR'] == 0
        assert init_res_1['n0'] == 30
        assert init_res_1['n1'] == 72
        # 2010
        assert init_res_2['error_code_test'] == 0.
        assert init_res_2['h_MEAN'] == 1
        assert init_res_2['h_VAR'] == 0
        assert init_res_2['n0'] == 36
        assert init_res_2['n1'] == init_res_1['n0']

    def test_lmp_results(self):
        testresults_ifirst, testresults_ilast, models_ifirst, models_ilast, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2012,7,1))
        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ilast['h_MEAN'] == 0
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']
        assert compare_metrics(group_stats, group_metrics, metrics_change)

        testresults_ifirst, testresults_ilast, models_ifirst, models_ilast, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2010,1,15))

        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ilast['h_MEAN'] == 0
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']


class Test_multibreak_adjust_qcm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ts = read_test_data(707393)
        ts_full = ts.rename(columns={'CCI_44_COMBINED': 'candidate',
                                     'MERRA2': 'reference'}).loc['2007-01-01':].copy(True)
        ts_full['candidate_original'] = ts_full['candidate'] # keep original
        # introduce some breaks to correct
        cls.breaktimes = np.array([datetime(2012,7,1), datetime(2010,1,15)])
        can_biased = ts_full.loc[:, 'candidate'].copy(True)
        can_biased[cls.breaktimes[0]:] += 0.1 # first break
        can_biased.loc[:cls.breaktimes[1]] -= 0.1 # second break
        ts_full.loc[:, 'candidate'] = can_biased
        ts_full['flags'] = 0. # all are good in the example

        test_kwargs = dict([('test_resample', ('M', 0.3)),
                            ('mean_test', 'wilkoxon'),
                            ('var_test', 'scipy_fligner_killeen'),
                            ('alpha', 0.01),
                            ('test_check_min_data', 5),
                            ('test_check_spearR_sig', [0., 1.])])

        adjmodel_kwargs = dict([('categories', 12),
                                ('first_last', 'formula'),
                                ('fit', 'mean')])

        adjfct_kwargs = {'interpolation_method': 'cubic'}

        adjcheck_kwargs = {'adjust_check_fix_temp_coverage': False,
                           'adjust_check_min_group_range': 365,
                           'adjust_check_pearsR_sig': (0., 1.)}

        cls.ts_full = ts_full.copy(True)

        cls.src = TsRelMultiBreak(candidate=cls.ts_full['candidate'],
                              reference=cls.ts_full['reference'],
                              breaktimes=cls.breaktimes,
                              adjustment_method='QCM',
                              candidate_flags=(cls.ts_full['flags'], [0]),
                              full_period_bias_corr_method='cdf_match',
                              sub_period_bias_corr_method='linreg',
                              base_breaktime=None,
                              HSP_init_breaktest=True,
                              models_from_hsp=True,
                              adjust_within='breaks',
                              input_resolution='D',
                              test_kwargs=test_kwargs,
                              adjmodel_kwargs=adjmodel_kwargs,
                              adjcheck_kwargs=adjcheck_kwargs,
                              create_model_plots=True,
                              frame_ts_figure=True,
                              frame_tsstats_plots=True)

        (res, freq) = dt_freq(cls.src.df_original.index)
        assert (res, freq) == (1., 'D')
        cls.candidate_adjusted = cls.src.adjust_all(extended_reference=True,
                                                    **adjfct_kwargs)
        assert cls.src.candidate_has_changed()

    def setUp(self):
        pass

    def tearDown(self):
        plt.close('all')

    @pytest.mark.skip(reason="currently ignoring plots because code uses deprecated functions and needs to be updated")
    def test_plots(self):
        plot_path = tempfile.mkdtemp()
        self.src.plot_adjustment_ts_full(plot_path, prefix='testQCM')
        self.src.plot_frame_ts_figure(plot_path, prefix='testQCM')
        # todo: why is this failing?
        #self.src.plot_tsstats_figures(plot_path, prefix='testQCM')
        self.src.plot_models_figures(plot_path, prefix='testQCM')
        shutil.rmtree(plot_path)

    def test_init_testresults(self):
        """
        Test results that are used to create the adjustment frames from,
        if the HSP_init_breaktest option is selected.
        """
        init_res_1 = self.src.init_test_results[self.breaktimes[0]]
        init_res_2 = self.src.init_test_results[self.breaktimes[1]]

        # 2012
        assert init_res_1['error_code_test'] == 0.
        assert init_res_1['h_MEAN'] == 1
        assert init_res_1['h_VAR'] == 0
        assert init_res_1['n0'] == 30
        assert init_res_1['n1'] == 72
        # 2010
        assert init_res_2['error_code_test'] == 0.
        assert init_res_2['h_MEAN'] == 1
        assert init_res_2['h_VAR'] == 0
        assert init_res_2['n0'] == 36
        assert init_res_2['n1'] == init_res_1['n0']

    def test_lmp_results(self):
        testresults_ifirst, testresults_ilast, models_ifirst, models_ilast, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2012,7,1))
        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ilast['h_MEAN'] == 0
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']
        assert compare_metrics(group_stats, group_metrics, metrics_change)

        testresults_ifirst, testresults_ilast, models_ifirst, models_ilast, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2010,1,15))

        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ilast['h_MEAN'] == 0
        assert testresults_ilast['h_VAR'] == testresults_ifirst['h_VAR']



class Test_multibreak_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ts = read_test_data(707393)
        ts_full = ts.rename(columns={'CCI_44_COMBINED': 'candidate',
                                     'MERRA2': 'reference'}).loc['2007-01-01':].copy(True)
        ts_full['candidate_original'] = ts_full['candidate'] # keep original
        # introduce some breaks to correct
        cls.breaktimes = np.array([datetime(2012,7,1), datetime(2010,1,15)])
        can_biased = ts_full.loc[:, 'candidate'].copy(True)
        can_biased[cls.breaktimes[0]:] += 0.1 # first break
        can_biased.loc[:cls.breaktimes[1]] -= 0.1 # second break
        ts_full.loc[:, 'candidate'] = can_biased
        ts_full['flags'] = 0. # all are good in the example

        test_kwargs = dict([('test_resample', ('M', 0.3)),
                            ('mean_test', 'wilkoxon'),
                            ('var_test', 'scipy_fligner_killeen'),
                            ('alpha', 0.01),
                            ('test_check_min_data', 5),
                            ('test_check_spearR_sig', [0., 1.])])

        cls.ts_full = ts_full.copy(True)

        cls.src = TsRelMultiBreak(candidate=cls.ts_full['candidate'],
                                  reference=cls.ts_full['reference'],
                                  breaktimes=cls.breaktimes,
                                  adjustment_method=None,
                                  candidate_flags=(cls.ts_full['flags'], [0]),
                                  full_period_bias_corr_method='cdf_match',
                                  sub_period_bias_corr_method='linreg',
                                  base_breaktime=None,
                                  HSP_init_breaktest=False,
                                  models_from_hsp=False,
                                  adjust_within='frames',
                                  input_resolution='D',
                                  test_kwargs=test_kwargs,
                                  adjmodel_kwargs={},
                                  adjcheck_kwargs={},
                                  create_model_plots=False,
                                  frame_ts_figure=False,
                                  frame_tsstats_plots=False)

        (res, freq) = dt_freq(cls.src.df_original.index)
        assert (res, freq) == (1., 'D')
        cls.src.test_all()
        assert cls.src.candidate_has_changed() == False

    def test_results(self):
        testresults_ifirst, _, _, _, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2012,7,1))
        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ifirst['h_VAR'] == 0
        assert compare_metrics(group_stats, group_metrics, metrics_change)

        testresults_ifirst, _, _, _, \
        group_stats, group_metrics, metrics_change, checkstats = \
            self.src.get_results(breaktime=datetime(2010,1,15))

        assert testresults_ifirst['h_MEAN'] == 1
        assert testresults_ifirst['h_VAR'] == 0

if __name__ == '__main__':
    unittest.main()
