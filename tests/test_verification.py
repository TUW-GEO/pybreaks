# -*- coding: utf-8 -*-

'''
Test the verification module that is used within the break correction
'''

from pybreaks.horizontal_errors import HorizontalVal, compare
from tests.helper_functions import create_artificial_test_data
import unittest
import numpy as np


class TestHorizontalErrors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.df, breaktime, timeframe = create_artificial_test_data(type='const')
        self.test = HorizontalVal(candidate=self.df.candidate,
                                  reference=self.df.reference,
                                  breaktime=breaktime)

    def tearDown(self):
        pass

    def test_run(self):
        comparison_method = 'AbsDiff'

        df_change = self.test.run(comparison_method=comparison_method)
        df_groupstats = self.test.df_group_stats

        col = 'group0_group1'
        df_change = df_change[col]

        mean_diff = df_change['{}_mean_Diff'.format(comparison_method)]
        median_diff = df_change['{}_median_Diff'.format(comparison_method)]
        min_diff = df_change['{}_min_Diff'.format(comparison_method)]
        max_diff = df_change['{}_max_Diff'.format(comparison_method)]

        assert median_diff == min_diff == max_diff == 0.8
        np.testing.assert_almost_equal(mean_diff, 0.8) # not exactly

        bias_diff = df_change['{}_bias'.format(comparison_method)]

        np.testing.assert_almost_equal(bias_diff, mean_diff)

        rmsd_diff = df_change['{}_rmsd'.format(comparison_method)]
        nrmsd_diff = df_change['{}_nrmsd'.format(comparison_method)]

        assert rmsd_diff == nrmsd_diff == 0

        can_0_min = df_groupstats.loc['min_candidate', 'group0']
        can_1_min = df_groupstats.loc['min_candidate', 'group1']

        ref_0_min = df_groupstats.loc['min_reference', 'group0']
        ref_1_min = df_groupstats.loc['min_reference', 'group1']

        can_0_max = df_groupstats.loc['max_candidate', 'group0']
        can_1_max = df_groupstats.loc['max_candidate', 'group1']

        ref_0_max = df_groupstats.loc['max_reference', 'group0']
        ref_1_max = df_groupstats.loc['max_reference', 'group1']

        ref_frame_mean = df_groupstats.loc['mean_reference', 'FRAME']
        can_frame_med = df_groupstats.loc['median_candidate', 'FRAME']

        assert can_0_min == can_0_max == 0.1
        assert can_1_min == can_1_max == 0.9
        assert ref_1_min == ref_0_min == 0.5
        assert ref_1_max == ref_0_max == 0.5
        assert ref_frame_mean == 0.5
        assert can_frame_med == 0.9

    @staticmethod
    def test_compare():
        d = compare(1., 2., 'AbsDiff')
        assert d == 1
        d = compare(1., 2., 'Ratio')
        assert d == 0.5
        d = compare(1., 0., 'Ratio') # Div0
        assert np.isnan(d)
        d = compare(1., 2., 'Diff')
        assert d == -1

if __name__ == '__main__':
    unittest.main()