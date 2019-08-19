# -*- coding: utf-8 -*-


from pybreaks.model_poly_regress import HigherOrderRegression
from pytesmo.scaling import linreg_stored_params, linreg_params
from tests.helper_functions import read_test_data, create_artificial_test_data
import unittest
import numpy as np
from datetime import datetime

class TestLinearRegress(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def tearDown(self):
        # called after a test (also if it fails)
        pass

    @staticmethod
    def create_model(realdata=True, poly_order=2, filter_p=None):
        if realdata:
            ts = read_test_data(325278)
            start, end = datetime(1998, 1, 1), datetime(2007, 1, 1)
            ts.rename(columns={'CCI_41_COMBINED': 'can',
                               'merra2': 'ref'}, inplace=True)
            ts_drop = ts.dropna()
            slope, inter = linreg_params(ts_drop['can'], ts_drop['ref'])
            ts['can'] = linreg_stored_params(ts['can'], slope, inter)  # scale
        else:
            ts, breaktime, [start, end] = create_artificial_test_data('asc2')

        regress = HigherOrderRegression(ts['can'].loc[start:end],
                                        ts['ref'].loc[start:end],
                                        poly_order=poly_order, filter_p=filter_p)
        return regress

    def test_poly_model_no_filter(self):
        p=2
        regress_nf = self.create_model(True, poly_order=p, filter_p=None)
        check = regress_nf.df_model['can'].dropna().eq(
                regress_nf.df_original['can'].dropna())

        assert(all(check) == True)

        check = regress_nf.df_model['ref'].dropna().eq(
                regress_nf.df_original['ref'].dropna())
        assert(all(check) == True)

        assert(regress_nf.df_model.loc['2000-01-01','Q'] ==
              (regress_nf.df_model.loc['2000-01-01','can'] -
               regress_nf.df_model.loc['2000-01-01','ref']))


        params_nf = regress_nf.get_model_params()

        np.testing.assert_almost_equal(params_nf['coef_0'], -0.3407021042)
        np.testing.assert_almost_equal(params_nf['coef_1'],  0.0139705460)
        assert np.isnan(params_nf['filter_p'])
        np.testing.assert_almost_equal(params_nf['inter'], 30.700283984)
        np.testing.assert_almost_equal(params_nf['mse'],   27.6561612998)
        assert params_nf['n_input'] == regress_nf.df_model.index.size
        assert params_nf['poly_order'] == p
        np.testing.assert_almost_equal(params_nf['r2'],  0.572561591)
        np.testing.assert_almost_equal(params_nf['sse'], 59543.71527864)

        ax = regress_nf.plot()


    def test_poly_model_filter(self):
        # Test if values are dropped
        p = 2
        filter_p = 5.
        regress_f = self.create_model(True, poly_order=p, filter_p=filter_p)
        assert regress_f.df_model.index.size  == 2044

        params_f = regress_f.get_model_params()

        np.testing.assert_almost_equal(params_f['coef_0'], -0.2741787363)
        np.testing.assert_almost_equal(params_f['coef_1'], 0.013642449537)
        assert params_f['filter_p'] == filter_p
        np.testing.assert_almost_equal(params_f['inter'], 28.4998018692)
        np.testing.assert_almost_equal(params_f['mse'],  23.61528847616)
        assert params_f['n_input'] == regress_f.df_model.index.size
        assert params_f['poly_order'] == p
        np.testing.assert_almost_equal(params_f['r2'],  0.62507165674)
        np.testing.assert_almost_equal(params_f['sse'], 48269.6496452)

        ax = regress_f.plot()


    def test_stats(self):
        regress_stats = self.create_model(True, poly_order=2, filter_p=None)

        lags = [0, 100, 200, 300, 400, 500]
        autocorr = regress_stats.residuals_autocorr(lags=lags)
        assert autocorr[0] == 1.
        assert autocorr.index.size == len(lags)

        np.testing.assert_almost_equal(regress_stats.me(True), -0.2646658840)
        np.testing.assert_almost_equal(regress_stats.mse(True), 11.12884641529)
        np.testing.assert_almost_equal(regress_stats.rmse(), 5.2589125586)
        np.testing.assert_almost_equal(regress_stats.r2(), 0.57256159130)


    def test_plot(self):
        # compare the output plots?
        pass

if __name__ == '__main__':
    ## done
    unittest.main()



