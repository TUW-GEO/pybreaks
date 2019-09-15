# -*- coding: utf-8 -*-
"""
TODO:
    - A Matrix for mMt values is not yet used because of the used test data
    - Test also the plotting
    - Test the corrections from core option
"""

import matplotlib # necessary in py2 but not in py 3
matplotlib.use('Agg')

from tests.helper_functions import read_test_data, create_artificial_test_data
import unittest
from pybreaks.adjust_linear_model_pair_fitting import RegressPairFit, PairRegressMatchAdjust
from datetime import datetime, timedelta
from pybreaks.utils import dt_freq
import numpy as np
import matplotlib.pyplot as plt

class Test_lmp_realdata_model_m(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts = read_test_data(654079)
        ts.rename(columns={'CCI': 'can',
                           'REF': 'ref'}, inplace=True)
        cls.ts_full = ts  # this can be used to test if values to adjust are not same as for model
        breaktime = datetime(2012, 7, 1)
        start = datetime(2010, 1, 15)
        ts_frame = ts.loc[start:, :]

        lmp_kwargs = dict(regress_resample=('M', 0.3))

        lmp = RegressPairFit(ts_frame['can'],
                             ts_frame['ref'],
                             breaktime,
                             candidate_freq='D',
                             bias_corr_method='linreg',
                             filter=('both', 5),
                             adjust_group=0,
                             model_intercept=True,
                             **lmp_kwargs)
        cls.lmp = lmp

    def setUp(self):
        (res, freq) = dt_freq(self.lmp.df_original.index)
        assert (res, freq) == (1., 'D')

        self.values_to_adjust = self.ts_full.loc[
                           datetime(2000, 1, 1):self.lmp.breaktime, 'can']
        # correction from core has only impact if values to adjust are not the same
        # as used to create the first model
        self.can_adjusted = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=True,
            interpolation_method='linear', values_to_adjust_freq='D')  # interpolation from M to D

        self.can_adjusted_noresample = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=False,
            interpolation_method='linear', values_to_adjust_freq='D')

    def tearDown(self):
        plt.close('all')

    def test_plots(self):
        fig = self.lmp.plot_models()
        fig = self.lmp.plot_stats_ts(self.lmp.df_original, kind='line', stats=True)
        fig = self.lmp.plot_adjustments()

    def test_correct_resample_interpolate(self):
        """
        The model is calculated from daily values.
        Corrections are derived for monthly resampled values and then interpolated
        to the target daily resolution of the values to adjust.
        """
        (res, freq) = dt_freq(self.lmp.df_adjust.index)
        assert (res, freq) == (1., 'M')

        assert self.can_adjusted.index.size == self.values_to_adjust.index.size

        corrections_interpolated = self.lmp.adjust_obj.adjustments  # interpolated M to D
        assert corrections_interpolated.index.size == 366.

        m0 = self.lmp.get_model_params(0)
        m1 = self.lmp.get_model_params(1)

        np.testing.assert_almost_equal(m0['slope'], 0.867307299)
        np.testing.assert_almost_equal(m0['inter'], 1.12528662)
        np.testing.assert_almost_equal(m1['slope'], 0.9175243965)
        np.testing.assert_almost_equal(m1['inter'], 1.392644610)

    def test_correct_direct(self):
        # do not resample for the models
        (res, freq) = dt_freq(self.lmp.df_adjust.index)
        assert (res, freq) == (1., 'M')

        values_to_adjust = self.ts_full.loc[
                           datetime(2000, 1, 1):self.lmp.breaktime, 'can']



        assert self.can_adjusted_noresample.index.size == values_to_adjust.index.size

        corrections_interpolated = self.lmp.adjust_obj.adjustments  # D, not interpolated
        assert corrections_interpolated.index.size == 366.

        m0 = self.lmp.get_model_params(0)  # models stay the same as for the previous test
        m1 = self.lmp.get_model_params(1)

        np.testing.assert_almost_equal(m0['slope'], 0.867307299)
        np.testing.assert_almost_equal(m0['inter'], 1.12528662)
        np.testing.assert_almost_equal(m1['slope'], 0.9175243965)
        np.testing.assert_almost_equal(m1['inter'], 1.392644610)


class Test_lmp_realdata_model_d(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts = read_test_data(654079)
        ts.rename(columns={'CCI': 'can',
                           'REF': 'ref'}, inplace=True)
        cls.ts_full = ts  # this can be used to test if values to adjust are not same as for model
        breaktime = datetime(2012, 7, 1)
        start = datetime(2010, 1, 15)
        ts_frame = ts.loc[start:, :]

        lmp_kwargs = dict(regress_resample=None)

        lmp = RegressPairFit(ts_frame['can'],
                             ts_frame['ref'],
                             breaktime,
                             candidate_freq='D',
                             bias_corr_method='linreg',
                             filter=('both', 5),
                             adjust_group=0,
                             model_intercept=True,
                             **lmp_kwargs)
        cls.lmp = lmp

    def setUp(self):
        (res, freq) = dt_freq(self.lmp.df_original.index)
        assert (res, freq) == (1., 'D')

        self.values_to_adjust = self.ts_full.loc[
                           datetime(2000, 1, 1):self.lmp.breaktime, 'can']
        # correction from core has only impact if values to adjust are not the same
        # as used to create the first model
        self.can_adjusted = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=True,
            interpolation_method='linear', values_to_adjust_freq='D')  # interpolation from M to D

        self.can_adjusted_noresample = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=False,
            interpolation_method='linear', values_to_adjust_freq='D')

    def tearDown(self):
        plt.close('all')

    def test_plots(self):
        fig = self.lmp.plot_models()
        fig = self.lmp.plot_stats_ts(self.lmp.df_original, kind='line', stats=True)
        fig = self.lmp.plot_adjustments()


    def test_correct_resample_interpolate(self):
        """
        The model is calculated from daily values.
        Corrections are derived for monthly resampled values and then interpolated
        to the target daily resolution of the values to adjust.
        """

        (res, freq) = dt_freq(self.lmp.df_adjust.index)
        assert (res, freq) == (1., 'D')

        assert self.can_adjusted.index.size == self.values_to_adjust.index.size

        corrections_interpolated = self.lmp.adjust_obj.adjustments  # interpolated M to D
        assert corrections_interpolated.index.size == 366.

        m0 = self.lmp.get_model_params(0)
        m1 = self.lmp.get_model_params(1)

        np.testing.assert_almost_equal(m0['slope'], 0.79388864747)
        np.testing.assert_almost_equal(m0['inter'], 2.126399268)
        np.testing.assert_almost_equal(m1['slope'], 1.009376751)
        np.testing.assert_almost_equal(m1['inter'], -0.1921845042)

    def test_correct_direct(self):
        # do not resample for the models
        (res, freq) = dt_freq(self.lmp.df_adjust.index)
        assert (res, freq) == (1., 'D')

        assert self.can_adjusted_noresample.index.size == self.values_to_adjust.index.size

        corrections_interpolated = self.lmp.adjust_obj.adjustments  # D, not interpolated
        assert corrections_interpolated.index.size == 366.

        m0 = self.lmp.get_model_params(0)  # models stay the same as for the previous test
        m1 = self.lmp.get_model_params(1)

        np.testing.assert_almost_equal(m0['slope'], 0.79388864747)
        np.testing.assert_almost_equal(m0['inter'], 2.126399268)
        np.testing.assert_almost_equal(m1['slope'], 1.009376751)
        np.testing.assert_almost_equal(m1['inter'], -0.1921845042)


class Test_lmp_synthetic_model_d(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ts, breaktime, timeframe = create_artificial_test_data('mean')

        ts.rename(columns={'candidate': 'can',
                           'reference': 'ref'}, inplace=True)
        cls.ts_full = ts  # this can be used to test if values to adjust are not same as for model
        start = timeframe[0]
        ts_frame = ts.loc[start:, :]

        lmp_kwargs = dict(regress_resample=None)

        lmp = RegressPairFit(ts_frame['can'],
                             ts_frame['ref'],
                             breaktime,
                             candidate_freq='D',
                             bias_corr_method='linreg',
                             filter=None, # cannot filter the test data, otherwise empty
                             adjust_group=0,
                             model_intercept=True,
                             **lmp_kwargs)
        cls.lmp = lmp

    def setUp(self):
        (res, freq) = dt_freq(self.lmp.df_original.index)
        assert (res, freq) == (1., 'D')

        self.values_to_adjust = self.ts_full.loc[
                           datetime(2000, 1, 1):self.lmp.breaktime, 'can']
        # correction from core has only impact if values to adjust are not the same
        # as used to create the first model

        self.can_adjusted = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=True,
            interpolation_method='linear', values_to_adjust_freq='D')  # interpolation from M to D

        self.can_adjusted_noresample = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=False,
            interpolation_method='linear', values_to_adjust_freq='D')

        self.can_adjusted_direct = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=True, resample_corrections=False,
            interpolation_method='linear', values_to_adjust_freq='D')

        self.can_adjusted_nocore = self.lmp.adjust(
            self.values_to_adjust, corrections_from_core=False, resample_corrections=False,
            interpolation_method='linear', values_to_adjust_freq='D')

    def tearDown(self):
        plt.close('all')

    def test_plots(self):
        fig = self.lmp.plot_models()
        fig = self.lmp.plot_stats_ts(self.lmp.df_original, kind='line', stats=True)
        fig = self.lmp.plot_adjustments()

    def test_correct_resample_interpolate(self):
        (res, freq) = dt_freq(self.lmp.df_adjust.index)
        assert (res, freq) == (1., 'D')
        assert self.can_adjusted.index.size == self.values_to_adjust.index.size

        # candidate before correction
        np.testing.assert_equal(np.unique(self.values_to_adjust.values), np.array([10.,11.]))
        # candidate after correction
        np.testing.assert_equal(np.unique(self.can_adjusted.values), np.array([50.,51.]))

        corrections_interpolated = self.lmp.adjust_obj.adjustments  # interpolated M to D
        assert corrections_interpolated.index.size == self.values_to_adjust.index.size
        assert all(corrections_interpolated == 40.) # 50-40 resp. 51-11

        m0 = self.lmp.get_model_params(0)
        m1 = self.lmp.get_model_params(1)

        np.testing.assert_almost_equal(m0['slope'], 1.0)
        np.testing.assert_almost_equal(m0['inter'], -20.10928961)
        np.testing.assert_almost_equal(m1['slope'], 1.0)
        np.testing.assert_almost_equal(m1['inter'], 19.8907103)

        # also get corrections from interpolated M values - must be same for the
        # synthetic case.
        assert np.alltrue(np.equal(self.can_adjusted_direct.values, self.can_adjusted.values))
        assert np.alltrue(np.equal(self.can_adjusted_nocore.values, self.can_adjusted.values))


if __name__ == '__main__':
    unittest.main()