# -*- coding: utf-8 -*-
"""
Test the input and output verifications for break detection and correction.
"""


def test_calc_bias():
    ''' Test bias calculation function'''
    base = test_base('const')
    g0_data = base.get_group_data(0, base.df_original, 'all')
    g1_data = base.get_group_data(1, base.df_original, 'all')

    bias0 = base._calc_RMSD(g0_data, 'candidate', 'reference')
    bias1 = base._calc_RMSD(g1_data, 'candidate', 'reference')

    nptest.assert_almost_equal(bias0, 0.4)
    nptest.assert_almost_equal(bias1, 0.4)

if __name__ == '__main__':
    test_calc_bias()