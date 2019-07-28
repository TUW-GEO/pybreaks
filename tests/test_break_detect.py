# -*- coding: utf-8 -*-
"""
Created on Jul 28 19:18 2019

@author: wolfgang
"""


def test_sse_around_breaktime():
    ''' Test calculation of SSE around the break time'''

    base = test_base('mean')
    sse, min_date = base.sse_around_breaktime(base.df_original,
                                              n_margin=10)

    assert(min_date== datetime(2000,6,30))

    base = test_base('var')
    sse, min_date = base.sse_around_breaktime(base.df_original,
                                              n_margin=10)

    assert(min_date== datetime(2000,6,30))


def test_calc_resiudals_autocorr():
    '''
    Test the calculation of the autocorrelation function for the single
    regression model residuals.
    '''

    base = test_base('mean')
    autocorr = base.calc_residuals_auto_corr(lags=range(30))

    # MEAN BREAK
    assert(autocorr.iloc[0] == 1.) # first value of autocorrelation always 1
    # autocorrelation for a mean break is not affected, therefore linear
    for i in range(autocorr.index[-1]):
        assert(autocorr.iloc[i] > autocorr.iloc[i+1])

    base = test_base('var')     # VAR BREAK
    autocorr = base.calc_residuals_auto_corr(lags=range(30))
    assert(autocorr.iloc[0] == 1.)

    # autocorrelation for a var break is affected, signs have to alternate
    for i in range(autocorr.index[-2]):
        assert(np.sign(autocorr.iloc[i]) == np.sign(autocorr.iloc[i+2]))
        if i % 2 == 0:
            assert(autocorr.iloc[i] > autocorr.iloc[i+2])
        else:
            assert(autocorr.iloc[i] < autocorr.iloc[i+2])



if __name__ == '__main__':
    test_sse_around_breaktime()

