# -*- coding: utf-8 -*-


from breakadjustment.model_lin_regress import LinearRegression
import pandas as pd
import numpy as np
from pytesmo.scaling import linreg_stored_params, linreg_params
from helper_funcions import read_test_data
from datetime import datetime
import numpy.testing as nptest
from sklearn import datasets

def linear_model_test_data(type):
    '''
    Create test data
    Parameters
    ----------
    type : str
        Set real for real data of gpi 325278 or 'artificial' for
        continuous test data.

    Returns
    -------
    ts : pandas.DataFrame
        Time series data for testing
    breaktime : datetime
        Time of potential break in the test data
    '''

    if type == 'real':
        ts, breaktime = read_test_data(325278)
        ts = ts.rename(columns={'CCI_41_COMBINED': 'candidate', 'merra2': 'reference'})
    elif type == 'artificial':
        index = pd.DatetimeIndex(start='1998-01-01', end='2007-01-01', freq='D')
        breaktime = datetime(2002, 6, 19)
        ts = pd.DataFrame(index=index, data={'candidate': range(3288),
                                             'reference': range(10, 3288+10)})
    else:
        raise Exception('Unknown type selected')

    return ts, breaktime



def init_model_unfiltered(stats_model=True, type='real'):

    ts, breaktime = linear_model_test_data(type)
    if type == 'real':
        slope, inter = linreg_params(ts.dropna().candidate, ts.dropna().reference)
        ts.candidate = linreg_stored_params(ts.candidate, slope, inter) # scale

    obj = LinearRegression(ts[['candidate']].loc['1998-01-01':'2007-01-01'],
                      ts[['reference']].loc['1998-01-01':'2007-01-01'],
                           filter_p=None,
                           stats_regress_implementation=stats_model)

    return obj


def init_model_filtered(stats_model=True, type='real'):
    ts, breaktime = linear_model_test_data(type)

    if type == 'real':
        slope, inter = linreg_params(ts.dropna().candidate, ts.dropna().reference)
        ts.candidate = linreg_stored_params(ts.candidate, slope, inter) # scale


    obj = LinearRegression(ts[['candidate']].loc['1998-01-01':'2007-01-01'],
                      ts[['reference']].loc['1998-01-01':'2007-01-01'],
                           filter_p=5,
                           stats_regress_implementation=stats_model)

    return obj


def test_lin_model_no_filter():
    model = init_model_unfiltered('real')

    # When no resampling is done the df is just a simple copy
    check = model.df_model.candidate.dropna() == \
            model.df_original.candidate.dropna() # type: pd.DataFrame

    assert(all(check) == True)

    check = model.df_model.reference.dropna() == \
        model.df_original.reference.dropna()    # type: pd.DataFrame
    assert(all(check) == True)

    assert(model.df_model.loc['2000-01-01','Q'] ==
           (model.df_model.loc['2000-01-01','candidate'] -
            model.df_model.loc['2000-01-01','reference']))

def test_params_no_filter():
    model = init_model_unfiltered(type='artificial')
    params = model.get_model_params(True)
    assert(np.round(params['slope'], 5) == 1.)
    assert(np.round(params['p_value'],5) == 0.)
    assert(np.round(params['inter'],5) == -10.)
    assert(np.round(params['median_squared_residuals'], 5) == 0.)
    assert(np.round(params['r_value'], 5) == 1.)
    assert(np.round(params['std_error'], 5) == 0.)
    assert(np.round(model.sse(), 5) == 0)


def test_2_linreg_functions():
    # check if own implementation has same results as scipy.stats one.
    model_stats = init_model_unfiltered(stats_model=True, type='real')
    model_other = init_model_unfiltered(stats_model=False, type='real')

    # single values
    model_stats_params = model_stats.get_model_params(True)
    model_other_params = model_other.get_model_params(True)

    assert(np.round(model_stats_params['slope'], 5) ==
           np.round(model_other_params['slope'], 5))
    assert(np.round(model_stats_params['inter'], 5) ==
           np.round(model_other_params['inter'], 5))
    assert(np.round(model_stats_params['s02'], 5) ==
           np.round(model_other_params['s02'], 5))
    assert(np.round(model_stats_params['median_squared_residuals'], 5) ==
           np.round(model_other_params['median_squared_residuals'], 5))


    assert (model_stats_params['r_value'] is not None) # supported
    assert (model_stats_params['std_error'] is not None) # supported
    assert (model_stats_params['p_value'] is not None) # supported

    assert (model_other_params['r_value'] is None)  # not supported
    assert (model_other_params['std_error'] is None)  # not supported
    assert (model_other_params['p_value'] is None)  # not supported


    assert(np.round(model_stats.sse(), 5) == np.round(model_other.sse(), 5))

    # continuous values
    model_stats_params = model_stats.get_model_params(False)
    model_other_params = model_other.get_model_params(False)

    nptest.assert_almost_equal(model_stats_params['median_squared_residuals'],
                               model_other_params['median_squared_residuals'], 5)
    nptest.assert_almost_equal(model_stats_params['candidate_modeled'],
                               model_other_params['candidate_modeled'], 5)
    nptest.assert_almost_equal(model_stats_params['residuals'].values,
                               model_other_params['residuals'].values, 5)


def test_lin_models_filter():
    model = init_model_filtered(type='real')
    # 5% of data was dropped
    assert(int(model.df_model.candidate.dropna().index.size) ==
           int(model.df_original.candidate.dropna().index.size * 0.95))

    assert(int(model.df_model.reference.dropna().index.size) ==
           int(model.df_original.reference.dropna().index.size * 0.95))

    assert (model.df_model.loc['2000-01-01', 'Q'] ==
            (model.df_model.loc['2000-01-01', 'candidate'] - \
             model.df_model.loc['2000-01-01', 'reference']))


def test_2_linreg_functions_filter():
    # check if own implementation has same results as scipy.stats one.
    model_stats = init_model_filtered(stats_model=True, type='real')
    model_other = init_model_filtered(stats_model=False, type='real')

    # single values
    model_stats_params = model_stats.get_model_params(True)
    model_other_params = model_other.get_model_params(True)

    assert(np.round(model_stats_params['slope'], 5) ==
           np.round(model_other_params['slope'], 5))
    assert(np.round(model_stats_params['inter'], 5) ==
           np.round(model_other_params['inter'], 5))
    assert(np.round(model_stats_params['s02'], 5) ==
           np.round(model_other_params['s02'], 5))
    assert(np.round(model_stats_params['median_squared_residuals'], 5) ==
           np.round(model_other_params['median_squared_residuals'], 5))


    assert (model_stats_params['r_value'] is not None) # supported
    assert (model_stats_params['std_error'] is not None) # supported
    assert (model_stats_params['p_value'] is not None) # supported

    assert (model_other_params['r_value'] is None)  # not supported
    assert (model_other_params['std_error'] is None)  # not supported
    assert (model_other_params['p_value'] is None)  # not supported


    assert(np.round(model_stats.sse(), 5) == np.round(model_other.sse(), 5))

    # continuous values
    model_stats_params = model_stats.get_model_params(False)
    model_other_params = model_other.get_model_params(False)

    nptest.assert_almost_equal(model_stats_params['median_squared_residuals'],
                               model_other_params['median_squared_residuals'], 5)
    nptest.assert_almost_equal(model_stats_params['candidate_modeled'],
                               model_other_params['candidate_modeled'], 5)
    nptest.assert_almost_equal(model_stats_params['residuals'].values,
                               model_other_params['residuals'].values, 5)
    

def test_compare_all_3_implementations():
    ''' compare results for scipy, sklearn and the default model'''
    testdata = datasets.load_diabetes()
    X = testdata.data[:,0]
    y = testdata.target

    rng = pd.date_range('2000-01-01', periods=X.size, freq='D')
    candidate = pd.DataFrame(index=rng, data={'CAN' : y})
    reference = pd.DataFrame(index=rng, data={'REF' : X})

    model = LinearRegression(candidate=candidate, reference=reference, fit_intercept=True,
                             force_implementation='default')
    std_params = model.get_model_params(True)

    model = LinearRegression(candidate=candidate, reference=reference, fit_intercept=True,
                             force_implementation='stats')
    stats_params = model.get_model_params(True)

    model = LinearRegression(candidate=candidate, reference=reference, fit_intercept=True,
                             force_implementation='sklearn')
    sklearn_params = model.get_model_params(True)

    df_concat=[]
    df_concat.append(pd.DataFrame(index=['std_params'], data=std_params))
    df_concat.append(pd.DataFrame(index=['stats_params'], data=stats_params))
    df_concat.append(pd.DataFrame(index=['sklearn_params'], data=sklearn_params))

    df = pd.concat(df_concat, axis=0)

    nptest.assert_almost_equal(df.at['std_params', 'inter'], df.at['stats_params', 'inter'])
    nptest.assert_almost_equal(df.at['std_params', 'inter'], df.at['sklearn_params', 'inter'])

    nptest.assert_almost_equal(df.at['std_params', 's02'], df.at['stats_params', 's02'])
    nptest.assert_almost_equal(df.at['std_params', 's02'], df.at['sklearn_params', 's02'])

    nptest.assert_almost_equal(df.at['std_params', 'slope'], df.at['stats_params', 'slope'])
    nptest.assert_almost_equal(df.at['std_params', 'slope'], df.at['sklearn_params', 'slope'])

    nptest.assert_almost_equal(df.at['std_params', 'sum_squared_residuals'],
                               df.at['stats_params', 'sum_squared_residuals'])
    nptest.assert_almost_equal(df.at['std_params', 'sum_squared_residuals'],
                               df.at['sklearn_params', 'sum_squared_residuals'])

    nptest.assert_almost_equal(df.at['stats_params', 'p_value'], df.at['sklearn_params', 'p_value'])
    nptest.assert_almost_equal(df.at['stats_params', 'r_squared'], df.at['sklearn_params', 'r_squared'])

if __name__ == '__main__':
    test_compare_all_3_implementations()
    test_lin_model_no_filter()
    test_params_no_filter()
    test_2_linreg_functions()
    test_lin_models_filter()
    test_lin_model_no_filter()
    test_2_linreg_functions_filter()



