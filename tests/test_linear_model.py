# -*- coding: utf-8 -*-


from pybreaks.model_lin_regress import LinearRegression
import pandas as pd
import numpy as np
from pytesmo.scaling import linreg_stored_params, linreg_params
from tests.helper_functions import read_test_data, create_artificial_test_data
import numpy.testing as nptest
import unittest
from datetime import datetime

class TestLinearRegress(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def tearDown(self):
        # called after a test (also if it fails)
        pass

    @staticmethod
    def create_model(realdata=True, force_implementation=None, filter_p=None):
        if realdata:
            ts = read_test_data(325278)
            start, end = datetime(1998, 1, 1), datetime(2007, 1, 1)
            breaktime = datetime(2002, 6, 19)
            ts = ts[start:end]
            ts.rename(columns={'CCI_41_COMBINED': 'candidate',
                               'merra2': 'reference'}, inplace=True)
        else:
            ts, breaktime, [start, end] = create_artificial_test_data('asc2')

        if realdata: # bias correction
            slope, inter = linreg_params(ts.dropna().candidate, ts.dropna().reference)
            ts.candidate = linreg_stored_params(ts.candidate, slope, inter)  # scale

        regress = LinearRegression(ts['candidate'].loc[start:end],
                                   ts['reference'].loc[start:end],
                                   filter_p=filter_p, fit_intercept=True,
                                   force_implementation=force_implementation)
        return regress

    def test_lin_model_no_filter(self):
        # When no resampling is done the df is just a simple copy
        regress = self.create_model(True, None)
        check = regress.df_model.candidate.dropna().eq(
                regress.df_original.candidate.dropna())

        assert(all(check) == True)

        check = regress.df_model.reference.dropna().eq(
                regress.df_original.reference.dropna())
        assert(all(check) == True)

        assert(regress.df_model.loc['2000-01-01','Q'] ==
              (regress.df_model.loc['2000-01-01','candidate'] -
               regress.df_model.loc['2000-01-01','reference']))


    def test_2_linreg_functions(self):
        # check if own implementation has same results as scipy.stats one.
        model_stats = self.create_model(True, 'lsq_stats')
        model_other = self.create_model(True, 'lsq_default')

        # single values
        model_stats_params = model_stats.get_model_params(True)
        model_other_params = model_other.get_model_params(True)

        assert (np.round(model_stats_params['slope'], 5) ==
                np.round(model_other_params['slope'], 5))
        assert (np.round(model_stats_params['inter'], 5) ==
                np.round(model_other_params['inter'], 5))
        assert (np.round(model_stats_params['s02'], 5) ==
                np.round(model_other_params['s02'], 5))
        assert (np.round(model_stats_params['median_squared_residuals'], 5) ==
                np.round(model_other_params['median_squared_residuals'], 5))

        assert (model_stats_params['r_squared'] is not None)  # supported
        assert (model_stats_params['std_error'] is not None)  # supported
        assert (model_stats_params['p_value'] is not None)  # supported

        assert (model_other_params['r_squared'] is None)  # not supported
        assert (model_other_params['std_error'] is None)  # not supported
        assert (model_other_params['p_value'] is None)  # not supported

        assert (np.round(model_stats.sse(), 5) == np.round(model_other.sse(), 5))

        # continuous values
        model_stats_params = model_stats.get_model_params(False)
        model_other_params = model_other.get_model_params(False)

        nptest.assert_almost_equal(model_stats_params['median_squared_residuals'],
                                   model_other_params['median_squared_residuals'], 5)
        nptest.assert_almost_equal(model_stats_params['candidate_modeled'],
                                   model_other_params['candidate_modeled'], 5)
        nptest.assert_almost_equal(model_stats_params['residuals'].values,
                                   model_other_params['residuals'].values, 5)


    def test_lin_models_filter(self):
        model = self.create_model(True, filter_p=5.)
        # 5% of data was dropped
        assert (int(model.df_model.candidate.dropna().index.size) ==
                int(model.df_original.candidate.dropna().index.size * 0.95))

        assert (int(model.df_model.reference.dropna().index.size) ==
                int(model.df_original.reference.dropna().index.size * 0.95))

        assert (model.df_model.loc['2000-01-01', 'Q'] ==
                (model.df_model.loc['2000-01-01', 'candidate'] -
                 model.df_model.loc['2000-01-01', 'reference']))


    def test_2_linreg_functions_filter(self):
        # check if own implementation has same results as scipy.stats one.
        model_stats = self.create_model(True, 'lsq_stats')
        model_other = self.create_model(True, 'lsq_default')

        # single values
        model_stats_params = model_stats.get_model_params(True)
        model_other_params = model_other.get_model_params(True)

        assert (np.round(model_stats_params['slope'], 5) ==
                np.round(model_other_params['slope'], 5))
        assert (np.round(model_stats_params['inter'], 5) ==
                np.round(model_other_params['inter'], 5))
        assert (np.round(model_stats_params['s02'], 5) ==
                np.round(model_other_params['s02'], 5))
        assert (np.round(model_stats_params['median_squared_residuals'], 5) ==
                np.round(model_other_params['median_squared_residuals'], 5))

        assert (model_stats_params['r_squared'] is not None)  # supported
        assert (model_stats_params['std_error'] is not None)  # supported
        assert (model_stats_params['p_value'] is not None)  # supported

        assert (model_other_params['r_squared'] is None)  # not supported
        assert (model_other_params['std_error'] is None)  # not supported
        assert (model_other_params['p_value'] is None)  # not supported

        assert (np.round(model_stats.sse(), 5) == np.round(model_other.sse(), 5))

        # continuous values
        model_stats_params = model_stats.get_model_params(False)
        model_other_params = model_other.get_model_params(False)

        nptest.assert_almost_equal(model_stats_params['median_squared_residuals'],
                                   model_other_params['median_squared_residuals'], 5)
        nptest.assert_almost_equal(model_stats_params['candidate_modeled'],
                                   model_other_params['candidate_modeled'], 5)
        nptest.assert_almost_equal(model_stats_params['residuals'].values,
                                   model_other_params['residuals'].values, 5)

    def test_compare_all_3_implementations(self):
        ''' compare results for scipy, sklearn and the default model'''

        model = self.create_model(True, 'lsq_default')
        std_params = model.get_model_params(True)

        model = self.create_model(True, 'lsq_stats')
        stats_params = model.get_model_params(True)

        model = self.create_model(True, 'lsq_sklearn')
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


    def test_params_no_filter(self):
        model = self.create_model(realdata=False)

        params = model.get_model_params(True)
        assert(np.round(params['slope'], 5) == 1.)
        assert(np.round(params['p_value'],5) == 0.)
        assert(np.round(params['inter'],5) == -10.)
        assert(np.round(params['median_squared_residuals'], 5) == 0.)
        assert(np.round(params['r_squared'], 5) == 1.)
        assert(np.round(params['std_error'], 5) == 0.)
        assert(np.round(model.sse(), 5) == 0)

    def test_stats(self):
        regress = self.create_model(True, filter_p=None)

        lags = [0, 100, 200, 300, 400, 500]
        autocorr = regress.residuals_autocorr(lags=lags)
        assert autocorr.values[0] == 1.
        assert autocorr.values.size == len(lags)

        np.testing.assert_almost_equal(regress.sse(), 69478.8302937324)
        np.testing.assert_almost_equal(regress.mse(True), 13.040270937220757)


    def test_plot(self):
        pass


if __name__ == '__main__':
    ## done
    unittest.main()



