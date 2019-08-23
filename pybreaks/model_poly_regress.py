# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from pybreaks.utils import filter_by_quantiles, autocorr
from pybreaks.base import TsRelBreakBase
import pandas as pd
import matplotlib.lines as mlines
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class HigherOrderRegression(TsRelBreakBase):
    '''
    Module to create linear and non-linear model between reference and candidate
    data after filtering the input data with respect to differences between
    candidate and reference data.
    '''

    def __init__(self, candidate, reference, poly_order=2, filter_p=None):

        '''
        Class for calculating linear models between the candidate and reference data.

        Parameters
        ----------
        candidate : pandas.Series
            The candidate (dependent) values, which will be filtered
        reference : pandas.Series
            The reference (independent) values
        poly_order : int, optional (default: 2)
            Degree of the polynomial kernel function (‘poly’) that we fit
        filter_p : float, optional (default:None)
            (between 0 and 100) or None to turn it off
            Drop the passed p% of worst (i.e difference compared to reference)
            candidate values before calculating the linear models.
            i.e. all values where Q is below this percentile.
        '''
        self.candidate = candidate.to_frame('can').copy(True)
        self.reference = reference.to_frame('ref').copy(True)


        self.filter_p = filter_p
        self.poly_order = int(poly_order)

        TsRelBreakBase.__init__(self, self.candidate, self.reference, breaktime=None,
                                bias_corr_method=None, dropna=True)

        self.df_model = self._filter()
        self.model = self._calc_poly_model(fit_inter=True)


    def plot(self, plot_data=True, plot_model=True, ax=None, vmax=None, plot_stats=True,
             scatter_style = ('o', 0.4, 'blue'), model_style = ('solid', 3, 'red'),
             oneoneline=True, label=None, label_scatter=False, label_model=False):
        """
        Plot data and model as scatter plot

        Prameters
        -------
        axs : matplotlib.Axes.axes
            Axes object that is used for plotting

        Returns
        -------
        ax : matplotlib.Axes.axes
            The plot axes
        """
        if not ax:
            self.fig = plt.figure(figsize=(5, 5), facecolor='w', edgecolor='k')
            ax = self.fig.add_subplot(1, 1, 1)

        df_subset = self.get_group_data(None, self.df_model, ['candidate_modeled',
                                                              self.candidate_col_name,
                                                              self.reference_col_name])
        if plot_data:
            ax.scatter(x=df_subset[self.reference_col_name].values,
                       y=df_subset[self.candidate_col_name].values,
                       alpha=scatter_style[1], color=scatter_style[2],
                       marker=scatter_style[0], label=label if (label and label_scatter) else None)


        if not vmax:
            vmax = df_subset.max().max()

        vmax = vmax + vmax * 0.15

        if oneoneline:
            line = mlines.Line2D([0, vmax], [0, vmax], color='black', linestyle='--',
                                 linewidth=1)
            ax.add_line(line)

        if plot_model:
            df_subset = df_subset.sort_values(self.reference_col_name)
            ax.plot(df_subset[self.reference_col_name].values,
                    df_subset['candidate_modeled'].values,
                    color=model_style[2], linewidth=model_style[1],
                    linestyle=model_style[0], label=label if (label and label_model) else None)

        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)

        ax.set_xlabel(self.reference_col_name)
        ax.set_ylabel(self.candidate_col_name)

        if plot_stats:
            textbox = r'N:{}, $\alpha:{:.2f}$'.format(self.n, self.inter)
            greek = ['beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta']
            for i, c in enumerate(self.coef):
                textbox+=r', $\{}:{:.2f}$'.format(greek[i], c)

            xlim = ax.get_xlim()
            pos_x = (xlim[0] + xlim[1]) / 2.
            pos_y = 0 + (vmax * 0.04)
            ax.annotate(textbox, fontsize=10, xy=(pos_x, pos_y),xycoords='data',
                        ha='center', va='bottom',
                        bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3, 'edgecolor':model_style[2]})


        return ax


    def _filter(self):
        """
        Performs percentile based filtering based on differences between
        candidate and reference.
        Stores the filtered values in the model data frame (self.df_model)

        Returns
        -------
        df_model : pd.DataFrame
            The filtered (or unfiltered) dataframe.
        """
        df_model = self.df_original.copy(True)
        df_model['Q'] = self.calc_diff(df_model)
        df_model, _ = self._frame_q_filter_(frame=df_model)

        return df_model

    def _frame_q_filter_(self, frame):
        """
        Filter data in the selected frame based on the difference values
        (Q-column) and the current filter settings.

        Parameters
        ----------
        frame : pandas.DataFrame
            The frame, that will be filtered based on Q

        Returns
        -------
        df_filtered : pd.DataFrame
            The filtered input data (input data that hast filter mask value 0)
        masked_values : pd.DataSeries
            Series of flags according to the objects input data
        """

        data = self.get_group_data(None, frame, 'all')

        if self.filter_p:
            # Take the absolute values, and drop the x% highest values in the
            # candidate series.
            abs_values_to_filter = self.get_group_data(None, frame, ['Q']).abs()

            # upper percentile is 100% minus the passed amount
            upper_p = 1. - (self.filter_p / 100.)
            # dont drop values that are close to 0 (candidate and reference are
            # similar) in the absolute time series, therefore this is always 0!!
            lower_p = 0.

            filter_mask = pd.concat([filter_by_quantiles(abs_values_to_filter,
                                                         'Q',
                                                         lower_p,
                                                         upper_p)],
                                    axis=0)
        else:
            filter_mask = pd.Series(index=self.get_group_data(None, frame, None),
                                    data=0).to_frame()


        data['diff_flag'] = filter_mask
        masked_values = data['diff_flag']
        data_filtered = data.loc[data['diff_flag'] == 0]

        df_filtered = data_filtered.drop(axis=1, labels='diff_flag')

        return df_filtered, masked_values

    def _calc_poly_model(self, fit_inter=True):
        """
        Calculate the actual (non) linear model for the candidate and
        reference data of the object using support vector regression from sklearn.

        Parameters
        -------
        C : float, optional (default = 1.)
            Penalty parameter C of the error term.

        Returns
        -------
        model : sklearn.svm.SVR
            Model between candidate and reference
        """

        data_group = self.get_group_data(
            None, self.df_model, [self.candidate_col_name, self.reference_col_name])

        X = data_group[self.reference_col_name].values.reshape(-1,1)
        y = data_group[self.candidate_col_name].values

        self.poly_features = PolynomialFeatures(degree=self.poly_order)
        X_train_poly = self.poly_features.fit_transform(X)
        model = LinearRegression(fit_intercept=fit_inter)
        model.fit(X_train_poly, y)

        self.n, c = X_train_poly.shape

        self.inter = model.intercept_
        self.coef = model.coef_[1:]

        self.valr2 = model.score(X_train_poly, y)

        canname = self.candidate_col_name
        self.df_model['candidate_modeled'] = model.predict(X_train_poly)
        self.df_model['errors'] = self.df_model[canname] - self.df_model['candidate_modeled']

        p_value = None
        # todo: find the significance


        return model


    def residuals_autocorr(self, lags):
        """
        Calculate auto-correlation function for the errors of the current model.

        Parameters
        -------
        lags : list
            Time lags used for the auto correlation function.
            E.g: [0,1,2] returns the auto-correlation values with for shifts over
            0, 1 and 2 time units.

        Returns
        -------
        autocorr : pandas.Series
            Auto correlation function for the current residuals with the selected
            lag(s).
        """
        residuals = self.df_model['errors'].copy(True)
        return pd.Series(index=lags, data=autocorr(residuals, lags))

    def r2(self):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0."""

        return self.valr2

    def rmse(self):
        """
        Calculate the root mean squared error from the residuals between the
        observed and predicted values.

        Returns
        -------
        rmse : float
            Root mean squared errors
        """
        residuals2 = self.df_model['errors'].copy(True) ** 2
        return np.sqrt(np.nanmean(residuals2.values))

    def sse(self):
        """
        Calculate the sum of squared errors of the current model

        Returns
        -------
        sse : float
            Sum of squared residuals

        """
        residuals2 = self.df_model['errors'].copy(True)**2
        return residuals2.sum()

    def me(self, median=False):
        """
        Calculate the median/mean of errors of the current model

        Parameters
        -------
        median : bool
            Use the median instead of the mean of the residuals

        Returns
        -------
        mse : float
            Median of the squared residuals

        """
        residuals = self.df_model['errors'].copy(True)
        if median:
            return np.nanmedian(residuals.values)
        else:
            return np.nanmean(residuals.values)


    def mse(self, median=False):
        """
        Calculate the median/mean of squared errors of the current model

        Parameters
        -------
        median : bool
            Use the median instead of the mean of the residuals

        Returns
        -------
        mse : float
            Median or mean of the squared residuals

        """
        residuals = self.df_model['errors'].copy(True)
        if median:
            return np.nanmedian((residuals ** 2).values)
        else:
            return np.nanmean((residuals ** 2).values)

    def get_model_params(self):
        '''
        Get parameters that define the model

        Returns
        -------
        model_params: dict
            Dictionary of the current model parameters

        '''
        #todo: return more values
        model_params = {}

        model_params['poly_order'] = self.poly_order
        for i, coef in enumerate(self.coef):
            model_params['coef_{}'.format(i)] = coef
        model_params['inter'] = self.inter


        model_params['r2'] = self.valr2
        model_params['n_input'] = self.n
        model_params['sse'] = self.sse()
        model_params['mse'] = self.mse()
        model_params['filter_p'] = self.filter_p if self.filter_p is not None else np.nan

        return model_params



if __name__ == '__main__':

    from cci_timeframes import CCITimes
    from io_data.otherfunctions import smart_import

    gpi = 431790

    canname = 'CCI_44_COMBINED'
    refname = 'MERRA2'

    times = CCITimes('CCI_44_COMBINED', min_set_days=None,
                     skip_breaktimes=[1, 3]).get_times(gpi=gpi)
    breaktimes, timeframes = times['breaktimes'], times['timeframes']

    ts_full, plotpath = smart_import(gpi, canname, refname)


    ts = ts_full.loc['1991-08-15':'1998-01-01']
    obj = HigherOrderRegression(ts[canname], ts[refname], filter_p=5, poly_order=2)
    obj.plot()

    corr = obj.residuals_autocorr(lags=range(10))


    ts = ts_full.loc['1991-08-15':'1998-01-01']
    obj = HigherOrderRegression(ts[canname], ts[refname], poly_order=2, filter_p=5)

    obj.plot()







