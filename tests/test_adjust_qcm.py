# -*- coding: utf-8 -*-
"""
TODO:
    -
"""
import matplotlib # necessary in py2 but not in py 3
matplotlib.use('Agg')

from tests.helper_functions import read_test_data, create_artificial_test_data
import unittest
from pybreaks.adjust_freq_quantile_matching import QuantileCatMatch, _init_rank
from datetime import datetime, timedelta
from pybreaks.utils import dt_freq
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd


class Test_qcm_realdata(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts = read_test_data(654079)
        ts.rename(columns={'CCI': 'can',
                           'REF': 'ref'}, inplace=True)
        cls.ts_full = ts  # this can be used to test if values to adjust are not same as for model
        breaktime = datetime(2012, 7, 1)
        start = datetime(2010, 1, 15)
        ts_frame = ts.loc[start:, :]

        qcm_kwargs = dict(categories=4, first_last='formula', fit='mean')

        qcm = QuantileCatMatch(ts_frame['can'],
                                 ts_frame['ref'],
                                 breaktime,
                                 bias_corr_method='linreg',
                                 adjust_group=0,
                                 **qcm_kwargs)
        cls.qcm = qcm

    def setUp(self):
        (res, freq) = dt_freq(self.qcm.df_original.index)
        assert (res, freq) == (1., 'D')

        self.values_to_adjust = self.ts_full.loc[
                           datetime(2000, 1, 1):self.qcm.breaktime, 'can'].dropna()
        self.can_adjusted = self.qcm.adjust(self.values_to_adjust, interpolation_method='cubic')

    def tearDown(self):
        plt.close('all')

    def test_plots(self):
        # todo: compare to reference plots
        self.qcm.plot_models()
        self.qcm.plot_adjustments()
        self.qcm.plot_cdf_compare()
        self.qcm.plot_ts(self.qcm.df_original)
        self.qcm.plot_emp_dist_can_ref()
        self.qcm.plot_pdf_compare()
        self.qcm.plot_stats_ts(self.qcm.df_original, kind='line', stats=True)

    def test_correct_resample_interpolate(self):
        assert self.can_adjusted.index.size == self.values_to_adjust.index.size

        corrections = self.qcm.adjust_obj.adjustments
        assert all(self.can_adjusted == (self.values_to_adjust + corrections))

        # test plots
        models = self.qcm.get_model_params() # the 1 model that counts
        m0 = models['model0']
        m1 = models['model1']

        nq_should = 4.
        assert m0['n_quantiles'] == m1['n_quantiles'] == nq_should
        # should be QCM + first + last + n_quantile_field = 7
        assert len(m0.keys()) == len(m0.keys()) == nq_should + 3
        np.testing.assert_almost_equal(m0[0.0], -2.34232735533)
        np.testing.assert_almost_equal(m0[0.125], -2.092327355335855)
        np.testing.assert_almost_equal(m0[0.375], -1.50420157216)
        np.testing.assert_almost_equal(m0[0.625], -0.792750240071)
        np.testing.assert_almost_equal(m0[0.875], 2.08102467237)
        np.testing.assert_almost_equal(m0[1.], 2.3310246723)

        np.testing.assert_almost_equal(m1[0.0], -2.01589693272)
        np.testing.assert_almost_equal(m1[0.125], -1.765896932729)
        np.testing.assert_almost_equal(m1[0.375], -1.006022746441)
        np.testing.assert_almost_equal(m1[0.625], 0.154860835)
        np.testing.assert_almost_equal(m1[0.875], 3.7751659653)
        np.testing.assert_almost_equal(m1[1.], 4.02516596535)

class Test_qcm_synthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts, breaktime, timeframe = create_artificial_test_data('norm_mean')
        ts.rename(columns={'candidate': 'can',
                           'reference': 'ref'}, inplace=True)
        cls.ts_full = ts  # this can be used to test if values to adjust are not same as for model

        qcm_kwargs = dict(categories=1, first_last='equal', fit='mean')

        qcm = QuantileCatMatch(ts['can'],
                               ts['ref'],
                               breaktime,
                               bias_corr_method='linreg',
                               adjust_group=0,
                               **qcm_kwargs)
        cls.qcm = qcm

    def setUp(self):
        (res, freq) = dt_freq(self.qcm.df_original.index)
        assert (res, freq) == (1., 'D')

        self.values_to_adjust = self.ts_full.loc[:self.qcm.breaktime, 'can'].dropna()
        self.can_adjusted = self.qcm.adjust(self.values_to_adjust, interpolation_method='cubic')

    def tearDown(self):
        plt.close('all')

    def test_plots(self):
        self.qcm.plot_models()
        self.qcm.plot_adjustments()
        self.qcm.plot_cdf_compare()
        self.qcm.plot_ts(self.qcm.df_original)
        self.qcm.plot_emp_dist_can_ref()
        self.qcm.plot_pdf_compare()
        self.qcm.plot_stats_ts(self.qcm.df_original, kind='line', stats=True)

    def test_correct_resample_interpolate(self):
        assert self.qcm.candidate_col_name == 'can'
        assert self.qcm.reference_col_name == 'ref'

        assert self.can_adjusted.index.size == self.values_to_adjust.index.size

        corrections = self.qcm.adjust_obj.adjustments
        assert all(self.can_adjusted == (self.values_to_adjust + corrections))

        # test plots
        models = self.qcm.get_model_params() # the 1 model that counts
        m0 = models['model0']
        m1 = models['model1']

        nq_should = 1.
        assert m0['n_quantiles'] == m1['n_quantiles'] == nq_should
        # should be QCM + first + last + n_quantile_field = 7
        assert len(m0.keys()) == len(m0.keys()) == nq_should + 3
        should = -5.011489733999822
        np.testing.assert_almost_equal(m0[0.0], should)
        np.testing.assert_almost_equal(m0[0.5], should)
        np.testing.assert_almost_equal(m0[1.0], should)

        should = 4.942394907872
        np.testing.assert_almost_equal(m1[0.0], should)
        np.testing.assert_almost_equal(m1[0.5], should)
        np.testing.assert_almost_equal(m1[1.0], should)


def test_init_rank():
    n = 100
    data = pd.DataFrame(index=range(n), data={'data': range(100)})
    cf_data, data_cf = _init_rank(data, 'data')

    np.testing.assert_equal(cf_data.index.values, np.arange(1.,101.,1))
    np.testing.assert_almost_equal(data_cf.index.values, np.arange(0.01,1.01,0.01))
    assert all(cf_data['F_data'] == 1)
    assert all(cf_data['norm_F_data'] == 1./n)
    assert all(cf_data['rank_data'] == range(1,n+1))


if __name__ == '__main__':
    unittest.main()
