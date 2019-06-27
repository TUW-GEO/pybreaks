# -*- coding: utf-8 -*-

from helper_funcions import create_test_data, read_test_data
from breakadjustment.adjust_linear_model_pair_fitting import RegressPairFit
import numpy.testing as nptest
import math

def pair_regress_object(type, resample=None, bias_corr=None, filter=None,
                        adjust_group=0):

    if type == 'real':
        df, breaktime = read_test_data(431790)
        df = df[['CCI_41_COMBINED', 'merra2']].\
            rename(columns={'CCI_41_COMBINED': 'candidate',
                            'merra2': 'reference'}).dropna()
    else:
        df, breaktime = create_test_data(type)

    obj = RegressPairFit(candidate=df['candidate'], reference=df['reference'],
                         breaktime=breaktime, regress_resample=resample,
                         bias_corr_method=bias_corr, bias_corr_group=None,
                         filter=filter, adjust_group=adjust_group)

    return obj


def test_pair_regress_adjust_daily_no_filter():
    '''
    Create models from daily values and apply daily adjustment values.
    '''

    #adjust data BEFORE break
    pra = pair_regress_object('mean', None, None, None, 0)

    model0params = pra.get_model_params(0, True)
    model1params = pra.get_model_params(1, True)

    nptest.assert_almost_equal(model0params['slope'], 1)
    nptest.assert_almost_equal(model0params['inter'], -20)
    nptest.assert_almost_equal(model0params['s02'], 0)
    nptest.assert_almost_equal(model0params['median_squared_residuals'], 0)
    nptest.assert_almost_equal(model0params['r_value'], 1)
    nptest.assert_almost_equal(model0params['p_value'], 0)

    nptest.assert_almost_equal(model1params['slope'], 1)
    nptest.assert_almost_equal(model1params['inter'], 20)
    nptest.assert_almost_equal(model1params['s02'], 0)
    nptest.assert_almost_equal(model1params['median_squared_residuals'], 0)
    nptest.assert_almost_equal(model1params['r_value'], 1)
    nptest.assert_almost_equal(model1params['p_value'], 0)

    can_adj = pra.adjust(adjust_param='both', interpolation_method=False)
    # mean break removed
    nptest.assert_almost_equal(can_adj.dropna().values, [50.,51.] * 91)


    # adjust data AFTER break
    pra = pair_regress_object('mean', None, None, None, 1)
    can_adj = pra.adjust(adjust_param='both', interpolation_method=False)
    # mean break removed
    nptest.assert_almost_equal(can_adj.dropna().values, [10., 11.] * 92)

def test_pair_regress_adjust_monthly_interpol_no_filter():
    '''
    Create models from monthly values, create monthly adjustments, interpolate.
    '''

    #adjust data BEFORE break
    pra = pair_regress_object('mean', ('M', 0.1), None, None, 0)

    model0params = pra.get_model_params(0, True)
    model1params = pra.get_model_params(1, True)

    nptest.assert_almost_equal(model0params['slope'], 1)
    nptest.assert_almost_equal(model0params['inter'], -20)
    nptest.assert_almost_equal(model0params['s02'], 0)
    nptest.assert_almost_equal(model0params['median_squared_residuals'], 0)
    nptest.assert_almost_equal(model0params['r_value'], 1)
    nptest.assert_almost_equal(model0params['p_value'], 0)

    nptest.assert_almost_equal(model1params['slope'], 1)
    nptest.assert_almost_equal(model1params['inter'], 20)
    nptest.assert_almost_equal(model1params['s02'], 0)
    nptest.assert_almost_equal(model1params['median_squared_residuals'], 0)
    nptest.assert_almost_equal(model1params['r_value'], 1)
    nptest.assert_almost_equal(model1params['p_value'], 0)

    method = ['linear', 'nearest', 'polynomial', 'spline']
    order = [None, None, 5, 4]

    # adjust data BEFORE the  break
    for m, o in zip(method, order):
        can_adj = pra.adjust(adjust_param='both', interpolation_method=m, order=o)
        # mean break removed
        nptest.assert_almost_equal(can_adj.dropna().values, [50.,51.] * 91)


    # adjust data AFTER break
    pra = pair_regress_object('mean', ('M', 0.1), None, None, 1)

    for m, o in zip(method, order):
        can_adj = pra.adjust(adjust_param='both', interpolation_method=m, order=o)
        # mean break removed
        nptest.assert_almost_equal(can_adj.dropna().values, [10., 11.] * 92)


def test_filter():
    #adjust data BEFORE break
    pra_filtered = pair_regress_object('asc', None, None, ('both', 10), 0)
    pra_unfiltered = pair_regress_object('asc', None, None, None, 0)

    # check if 10% have been removed
    n_should = int(pra_unfiltered.model0.df_model.index.size * 0.9) # round down
    n_is = pra_filtered.model0.df_model.index.size
    assert(n_should == n_is)

    n_should = int(pra_unfiltered.model1.df_model.index.size * 0.9) # round down
    n_is = pra_filtered.model1.df_model.index.size
    assert(n_should == n_is)


def test_pair_regress_adjust_daily_filter():
    '''
    Create models from daily values and apply daily adjustment values.
    '''
    pra = pair_regress_object('real', None, 'linreg', ('both', 10), 0)
    adjusted = pra.adjust('both', None)

    # group 0 was adjusted --> changed
    can = pra.get_group_data(0, pra.df_original, 'candidate')
    adj = pra.get_group_data(0, pra.df_original, 'candidate_adjusted')
    assert(all(can.values != adj.values))

    # group 1 was not adjusted --> kept
    can = pra.get_group_data(1, pra.df_original, 'candidate')
    adj = pra.get_group_data(1, pra.df_original, 'candidate_adjusted')
    assert(all(can.values == adj.values))

    cols = ['candidate', 'reference', 'candidate_adjusted']
    group, vert, hor = pra.get_validation_stats(frame=pra.df_original,
                                                columns=cols,
                                                comp_meth='AbsDiff')

    delta_mean_can = hor['candidate_reference_AbsDiff_bias']
    delta_mean_adj = hor['candidate_adjusted_reference_AbsDiff_mean_Diff']
    assert(delta_mean_adj < delta_mean_can) # data means were fit




def test_pair_regress_adjust_monthly_interpol_filter():
    pra = pair_regress_object('real', ('M', 0.3), 'linreg', ('both', 10), 0)
    adjusted = pra.adjust('both', 'linear')

    # group 0 was adjusted --> changed
    can = pra.get_group_data(0, pra.df_original, 'candidate')
    adj = pra.get_group_data(0, pra.df_original, 'candidate_adjusted')
    assert(all(can.values != adj.values))

    # group 1 was not adjusted --> kept
    can = pra.get_group_data(1, pra.df_original, 'candidate').values
    adj = pra.get_group_data(1, pra.df_original, 'candidate_adjusted').values
    assert(all(can == adj))

    # filtering reduces the number of model input values
    assert(pra.model0.df_model.index.size < pra.model0.df_original.index.size)
    assert(pra.model1.df_model.index.size < pra.model1.df_original.index.size)

    cols = ['candidate', 'reference', 'candidate_adjusted']
    group, vert, hor = pra.get_validation_stats(frame=pra.df_original,
                                                columns=cols,
                                                comp_meth='AbsDiff')

    delta_mean_can = hor['candidate_reference_AbsDiff_bias']
    delta_mean_adj = hor['candidate_adjusted_reference_AbsDiff_mean_Diff']
    # adjustment fits at least the change in means
    assert(delta_mean_adj < delta_mean_can) # data means were fit




if __name__ == '__main__':
    test_pair_regress_adjust_daily_no_filter()
    test_pair_regress_adjust_monthly_interpol_no_filter()
    test_filter()
    test_pair_regress_adjust_monthly_interpol_filter()
    test_pair_regress_adjust_daily_filter()



