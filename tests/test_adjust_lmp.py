# -*- coding: utf-8 -*-
"""
"""

from helper_functions import read_test_data, create_artificial_test_data
import unittest
from pybreaks.adjust_linear_model_pair_fitting import RegressPairFit, PairRegressMatchAdjust
from datetime import datetime, timedelta
from pybreaks.utils import dt_freq
import numpy as np
from pprint import pprint


class TestLinearRegress(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def tearDown(self):
        # called after a test (also if it fails)
        pass

    def create_object(self, realdata=True, **lmp_kwargs):
        if realdata:
            ts = read_test_data(654079)
            ts.rename(columns={'CCI': 'can',
                               'REF': 'ref'}, inplace=True)
            self.ts_full = ts # this can be used to test if values to adjust are not same as for model
            breaktime = datetime(2012,7,1)
            start = datetime(2010,1,15)
            ts_frame = ts.loc[start:, :]
        else:
            ts_frame, breaktime, timeframe = create_artificial_test_data('mean')

        lmp = RegressPairFit(ts_frame['can'],
                             ts_frame['ref'],
                             breaktime,
                             candidate_freq='D',
                             bias_corr_method='linreg',
                             filter=('both', 5),
                             adjust_group=0,
                             model_intercept=True,
                             **lmp_kwargs)
        return lmp

    # def test_lmp_realdata_resample(self):
    #     lmp = self.create_object(realdata=True, regress_resample=('M', 0.3))
    #
    #     (res, freq) = dt_freq(lmp.df_original.index)
    #     assert (res, freq) ==(1., 'D')
    #     (res, freq) = dt_freq(lmp.df_adjust.index)
    #     assert (res, freq) ==(1., 'M')
    #
    #   values_to_adjust = self.ts_full[datetime(2000, 1, 1):lmp.breaktime]

    # resampled correction only have inpact if input NOT already resampled!
    #     can_adjusted = lmp.adjust(
    #         values_to_adjust, corrections_from_core=True, resample_corrections=False,
    #         interpolation_method='linear', values_to_adjust_freq='D')
    #
    #     lmp.adjust_obj.adjustments # monthly adjustments
    #     lmp.plot_adjustments()
    #
    #     m0 = lmp.get_model_params(0)
    #     m1 = lmp.get_model_params(1)

        # resampled correction only have inpact if input NOT already resampled!
    #     can_adjusted = lmp.adjust(
    #         values_to_adjust, corrections_from_core=False, resample_corrections=False,
    #         interpolation_method='linear', values_to_adjust_freq='D')
    #
    #     lmp.adjust_obj.adjustments # monthly adjustments
    #     lmp.plot_adjustments()
    #
    #     m0 = lmp.get_model_params(0)
    #     m1 = lmp.get_model_params(1)


    def test_lmp_realdata_noresample(self):
        # do not resample for the models
        lmp = self.create_object(realdata=True, regress_resample=None)

        (res, freq) = dt_freq(lmp.df_original.index)
        assert (res, freq) ==(1., 'D')
        (res, freq) = dt_freq(lmp.df_adjust.index)
        assert (res, freq) ==(1., 'D')

        values_to_adjust = self.ts_full[datetime(2000,1,1):lmp.breaktime]
        # resample for the corrections.
        # correction from core has only impact if values to adjust are not the same
        # as used to create the first model
        can_adjusted = lmp.adjust(
            values_to_adjust, corrections_from_core=True, resample_corrections=True,
            interpolation_method='linear', values_to_adjust_freq='D')

        lmp.adjust_obj.adjustments # monthly adjustments
        lmp.plot_adjustments()

        m0 = lmp.get_model_params(0)
        m1 = lmp.get_model_params(1)


        can_adjusted = lmp.adjust(
            values_to_adjust, corrections_from_core=True, resample_corrections=False,
            interpolation_method='linear', values_to_adjust_freq='D')

        lmp.adjust_obj.adjustments # monthly adjustments
        lmp.plot_adjustments()

        m0 = lmp.get_model_params(0)
        m1 = lmp.get_model_params(1)

if __name__ == '__main__':
    unittest.main()