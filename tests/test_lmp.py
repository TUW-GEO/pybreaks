# -*- coding: utf-8 -*-
"""
"""

from helper_functions import read_test_data, create_artificial_test_data
import unittest
from pybreaks.adjust_linear_model_pair_fitting import RegressPairFit, PairRegressMatchAdjust
from datetime import datetime, timedelta
import numpy as np





class TestLinearRegress(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def tearDown(self):
        # called after a test (also if it fails)
        pass

    @staticmethod
    def create_object(realdata=True):
        if realdata:
            ts, breaktime = read_test_data(654079)
            ts.rename(columns={'CCI_45_COMBINED': 'can',
                               'MERRA2': 'ref'}, inplace=True)
            breaktime = datetime(2012,7,1)
        else:
            ts, breaktime = create_artificial_test_data('mean')

        lmp = RegressPairFit(ts_frame[canname],
                             ts_frame[refname],
                             breaktime,
                             candidate_freq='D',
                             regress_resample=('M', 0.3),
                             bias_corr_method='linreg',
                             filter=('both', 5),
                             adjust_group=0,
                             model_intercept=True)
        return lmp

    def test_lmp_realdata(self):
        lmp = self.create_object(realdata=True)

        values_to_adjust = lmp.df_original['can'].loc[:lmp.breaktime]
        can_adjusted = lmp.adjust(
            values_to_adjust, corrections_from_core=True, resample_corrections=True,
            interpolation_method='linear', values_to_adjust_freq='D')
        obj.plot_adjustments()

        m0 = lmp.get_model_params(0)
        m1 = lmp.get_model_params(1)
