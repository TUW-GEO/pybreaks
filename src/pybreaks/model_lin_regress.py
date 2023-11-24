# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.linear_model as sklearnmodel
from pybreaks.utils import filter_by_quantiles
from pybreaks.base import TsRelBreakBase
import pandas as pd
from pybreaks.utils import autocorr
import matplotlib.lines as mlines

'''
This module implements multiple ways to create linear regression models for
different packages. 

TODO #################################
(+) Implement the Theil-Sen linear regression
(+) Implement the RANSAC linear regression

NOTES ################################
- The model_poly_regress.py module also implements a linear regression model, 
but this module is more specific and allows different options to do it.
'''


class LinearRegression(TsRelBreakBase):
    """
    Module to create a linear model between reference and candidate data after
    filtering the input data with respect to differences between candidate and
    reference data.
    """

    def __init__(self, candidate, reference, filter_p=None, fit_intercept=True,
                 force_implementation=None):

        """
        Class for calculating linear models between the candidate and reference data.

        Parameters
        ----------
        candidate : pd.Series
            The candidate observations, which will be filtered
        reference : pd.Series
            The reference observations
        filter_p : float or None, optional (default: None)
            Between 0 and 100
            Drop the passed p% of worst (i.e difference compared to reference)
            candidate values before calculating the linear models.
            i.e. all values where Q is below this percentile.
        fit_intercept : bool, optional (default: True)
            Set to False to only model the slope paramter and set intercept to 0
        force_implementation : str, optional (default: None)
            Force a specific implementation (might not work with all input combinations,
            eg not modelling the intecept does only work with sklearn)
           'lsq_default' for default implementation (matrix algebra, not used)
           'lsq_sklearn' for the sklearn implementation
           'lsq_stats' for the scipy stats implementation
           'theilsen' NOT IMPLEMENTED : calculate a TheilSen model.
           'ransac' NOT IMPLEMENTED : calculate a RANSAC model.
           If None is passed, we choose based on other passed params.
        """

        if force_implementation:
            implementation = force_implementation
        else:
            # we choose between scipy and sklearn if user does not specify.
            implementation = 'lsq_stats' if fit_intercept else 'lsq_sklearn'

        self.fit_intercept = fit_intercept
        self.filter_p = filter_p

        TsRelBreakBase.__init__(self, candidate, reference, breaktime=None,
                                bias_corr_method=None, dropna=True)

        self.df_model = self._filter()
        self._calc_regress_model(implementation)

    def _calc_regress_model(self, implementation):
        """
        Calls the chosen implementation of the linear regression model with the
        properties that were chosen for this class.

        Parameters
        -------
        implementation : str
            Identifies the implementation for the model calculation
        """

        data_group = self.get_group_data(
            None, self.df_model, [self.candidate_col_name, self.reference_col_name])

        subset_candidate = data_group[self.candidate_col_name]
        subset_reference = data_group[self.reference_col_name]

        if implementation == 'lsq_stats':
            if not self.fit_intercept:
                raise ValueError(implementation, 'Method cannot ignore the intercept')
            B, n, r_value, p_value, std_err, can_model, residuals = \
                self._stats_regression_model(subset_candidate, subset_reference)
        elif implementation == 'lsq_default':
            if not self.fit_intercept:
                raise ValueError(implementation, 'Method cannot ignore the intercept')
            B, n, r_value, p_value, std_err, can_model, residuals = \
                self._default_regression_model(subset_candidate, subset_reference)
        elif implementation == 'lsq_sklearn':
            B, n, r_value, p_value, std_err, can_model, residuals = \
                self._sklearn_regression_model(subset_candidate, subset_reference,
                                               fit_intercept=self.fit_intercept)
        elif implementation == 'theilsen':
            raise NotImplementedError
        elif implementation == 'ransac':
            raise NotImplementedError
        else:
            raise ValueError(implementation, 'Unknown implementation')

        self.residuals = residuals
        self.s02 = np.dot(np.transpose(residuals), residuals) / (n - 2)
        self.candidate_modeled = can_model
        self.std_error = std_err
        self.p_value = p_value
        self.r_value = r_value
        self.n = n

        self.df_model.loc[subset_candidate.index, 'residuals'] = residuals
        self.df_model.loc[subset_candidate.index, 'candidate_modeled'] = can_model

        self.B = [B[0], B[1]]  # B = [k, d]

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

            masked = filter_by_quantiles(
                df_in=abs_values_to_filter, filter_col='Q', lower=lower_p,
                upper=upper_p)
            filter_mask = pd.concat([masked], axis=0)
        else:
            filter_mask = pd.Series(
                index=self.get_group_data(None, frame, None), data=0).to_frame()

        data['diff_flag'] = filter_mask
        masked_values = data['diff_flag']
        data_filtered = data.loc[data['diff_flag'] == 0]

        df_filtered = data_filtered.drop(axis=1, labels='diff_flag')

        return df_filtered, masked_values

    @staticmethod
    def _ransac_regression_model(candidate_data, reference_data):
        """
        Implement a RANSAC based linear regression estimation
        """
        raise NotImplementedError('Ransanc model not implemented yet.')

    @staticmethod
    def _theilsen_regression_model(candidate_data, reference_data):
        """
        Implement a Theil-Sen based linear regression estimation
        """
        raise NotImplementedError('Theil-Sen regression not implemented yet.')

    @staticmethod
    def _default_regression_model(candidate_data, reference_data):
        """
        Calculates the regression model between candidate and reference

        Parameters
        ----------
        candidate_data : np.array
            Candidate values as array
        reference_data : np.array
            Reference data as array

        Returns
        -------
        b : np.array
            Model parameters (intercept, slope)
        n : int
            Number of elemets
        placeholder1 : None
            This is just so the outputs are the same as for the scipy version.
        placeholder2 : None
            This is just so the outputs are the same as for the scipy version.
        placeholder3 : None
            This is just so the outputs are the same as for the scipy version.
        candidate_modeled : np.array
            Application of the linear model to the reference data yields the
            modeled candidate
        residuals : np.array
            Residuals between candidate and modeled candidate
        """

        n_obs = len(candidate_data)
        x = np.transpose(np.stack((reference_data, np.ones(n_obs))))

        n = np.dot(np.transpose(x), x)
        try:
            b = np.dot(np.dot(np.linalg.inv(n), np.transpose(x)), candidate_data)
        except np.linalg.LinAlgError:
            raise ValueError('6: N Matrix singular')

        candidate_modeled = np.dot(b, np.transpose(x))

        # Residuals =  candidate - candidate_modeled
        residuals = np.dot(-1 * x, b) + candidate_data

        return b, n_obs, None, None, None, candidate_modeled, residuals

    @staticmethod
    def _stats_regression_model(candidate_data, reference_data):
        """
        Calculates the regression model between candidate and reference based
        on the scipy.stats module.

        Parameters
        ----------
        candidate_data : np.array
            Candidate values as array
        reference_data : np.array
            Reference data as array

        Returns
        -------
        b : np.array
            Model parameters (intercept, slope)
        n : int
            Number of elemets
        r_value : float
            Correlation coefficient
        p_value : float
            Two-sided p-value for a hypothesis test whose null hypothesis is
            that the slope is zero.
        std_err : float
            Standard error of the estimate
        candidate_modeled : np.array
            Application of the linear model to the reference data yields the
            modeled candidate
        residuals : np.array
            Residuals between candidate and modeled candidate
        """

        n = candidate_data.index.size
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            reference_data, candidate_data)

        b = np.array([slope, intercept])

        x = np.transpose(np.stack((reference_data, np.ones(n))))
        candidate_modeled = np.dot(b, np.transpose(x))

        residuals = candidate_data - candidate_modeled  # Residuals =  candidate - candidate_modeled

        return b, n, r_value, p_value, std_err, candidate_modeled, residuals

    @staticmethod
    def _sklearn_regression_model(candidate_data, reference_data, fit_intercept):
        """
        Calculates the regression model between candidate and reference based
        on the scipy.stats module.

        Parameters
        ----------
        candidate_data : np.array
            Candidate values as array
        reference_data : np.array
            Reference data as array
        fit_intercept : bool
            Fit the intercept also.

        Returns
        -------
        b : np.array
            Model parameters (intercept, slope)
        n : int
            Number of elemets
        r_value : float
            Correlation coefficient
        p_value : float
            Two-sided p-value for a hypothesis test whose null hypothesis is
            that the slope is zero.
        std_err : float
            Standard error of the estimate
        candidate_modeled : np.array
            Application of the linear model to the reference data yields the
            modeled candidate
        residuals : np.array
            Residuals between candidate and modeled candidate
        """

        n = candidate_data.index.size

        model = sklearnmodel.LinearRegression(fit_intercept=fit_intercept,
                                              copy_X=True)

        X = reference_data.values.reshape(-1, 1)
        y = candidate_data.values.reshape(-1, 1)
        model = model.fit(X, y, sample_weight=None)

        inter = model.intercept_[0] if fit_intercept else model.intercept_
        slope = model.coef_[0][0]

        b = np.array([slope, inter])

        candidate_modeled = model.predict(X)
        r_value = np.sqrt(model.score(X, y))

        # Residuals =  candidate - candidate_modeled
        residuals = y[:, 0] - candidate_modeled[:, 0]
        residuals = pd.Series(index=candidate_data.index, data=residuals)

        sse = np.sum((y - candidate_modeled) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                       for i in range(sse.shape[0])])
        t = b[0] / se

        p_value = 2 * (1 - stats.t.cdf(np.abs(t), n - X.shape[1]))[0][0]

        return b, n, r_value, p_value, None, candidate_modeled[:, 0], residuals

    def plot(self, plot_data=True, plot_model=True, ax=None, vmax=None, plot_stats=True,
             scatter_style=('o', 0.4, 'blue'), model_style=('solid', 3, 'red'),
             oneoneline=True):
        """
        Plot data and model as scatter plot

        Parameters
        -------
        plot_data : bool, optional (default: True)
            Add the data points as a scatter plot.
        plot_model : bool, optional (default: True)
            Add the candidate predictions from the model as a line plot (linear)
        ax : matplotlib.Axes.axes
            Axes object that is used for plotting
        vmax : float, optional (defaul: None)
            Set the maximum (we add 15%) of the 2 axes that are plotted
        plot_stats : bool, optional (default: True)
            Plot the stats of the model (N, slope, int) at the bottom in a box.
        scatter_style : (str, float, str), optional (default: ('o', 0.4, 'blue'))
            (marker_style, marker_alpha, marker_color)
            Styling for the scatter plot part of the figure (the observations).
        model_style : (str, float, str), optional (default: ('o', 0.4, 'blue'))
            (linestyle, linewidth, color)
            Styling for the line plot part of the figure (the predictions).
        oneoneline: bool, optional (default: True)
            Plot the one by one line

        Returns
        -------
        ax : matplotlib.Axes.axes
            The plot axes
        """
        if not ax:
            fig = plt.figure(figsize=(4, 4), facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)

        cols = ['candidate_modeled', self.candidate_col_name, self.reference_col_name]
        df_subset = self.get_group_data(group_no=None, frame=self.df_model,
                                        columns=cols)
        model_params = self.get_model_params()

        if plot_data:
            ax.scatter(df_subset[self.reference_col_name].values,
                       df_subset[self.candidate_col_name].values,
                       marker=scatter_style[0],
                       alpha=scatter_style[1], color=scatter_style[2])

        if not vmax:
            vmax = df_subset.max().max()

        vmax = vmax + vmax * 0.15

        if oneoneline:
            line = mlines.Line2D([0, vmax], [0, vmax], color='black',
                                 linestyle='--', linewidth=1)
            ax.add_line(line)

        if plot_model:
            ax.plot(df_subset[self.reference_col_name].values,
                    df_subset['candidate_modeled'].values,
                    color=model_style[2], linewidth=model_style[1],
                    linestyle=model_style[0])

        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)

        ax.set_xlabel(self.reference_col_name)
        ax.set_ylabel(self.candidate_col_name)
        # ax.axis('equal')

        # Add a text box with stats
        if plot_stats:
            textbox = r'N:%i, $\alpha$:%.2f, $\beta$:%.2f' \
                      % (df_subset.index.size, model_params['inter'],
                         model_params['slope'])

            xlim = ax.get_xlim()
            pos_x = (xlim[0] + xlim[1]) / 2.
            pos_y = 0 + (vmax * 0.04)
            ax.annotate(textbox, fontsize=12, xy=(pos_x, pos_y), xycoords='data',
                        ha='center', va='bottom',
                        bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 3})

        return ax

    def get_model_params(self, only_1d=True):
        """
        Get the model parameters of the linear model for the according group of the current iteration

        Parameters
        ----------
        only_1d : bool, optional (default: True)
            Exclude time series parameters (like residuals)

        Returns
        -------
        model_params: dict
            Dictionary of the current model parameters
        """

        model_params = {}

        slope, intercept = self.B[0], self.B[1]
        model_params['slope'] = slope
        model_params['inter'] = intercept
        model_params['n_input'] = self.n
        model_params['s02'] = self.s02
        model_params['std_error'] = self.std_error
        model_params['r_squared'] = self.r_value ** 2 if self.r_value else None
        model_params['p_value'] = self.p_value
        model_params['sum_squared_residuals'] = self.sse()
        model_params['mean_squared_residuals'] = self.mse(median=False)
        model_params['median_squared_residuals'] = self.mse(median=True)

        if not only_1d:
            model_params['residuals'] = self.residuals
            model_params['candidate_modeled'] = self.candidate_modeled

        return model_params

    def residuals_autocorr(self, lags=0):
        """
        Calculate autocorrelation function for the residuals of the current model.

        Parameters
        -------
        lags : list or int, optional (default: 0)
            Time lags used for the auto correlation function

        Returns
        -------
        autocorr : pandas.Series
            Auto correlation function for the current residuals with the selected
            lag(s).
        """
        return pd.Series(index=lags, data=autocorr(self.residuals, lags))

    def sse(self):
        """
        Calculate the sum of squared errors (residuals) the the current model

        Returns
        -------
        sse : float
            Sum of squared residuals
        """
        residuals2 = self.df_model['residuals'].copy(True) ** 2
        return residuals2.sum()

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
        if median:
            return np.nanmedian((self.residuals ** 2).values)
        else:
            return np.nanmean((self.residuals ** 2).values)

def usecase():
    from io_data.otherfunctions import smart_import
    from pprint import pprint

    gpi = 431790

    canname = 'CCI_44_COMBINED'
    refname = 'MERRA2'

    ts_full, plotpath = smart_import(gpi, canname, refname)

    ts_full = ts_full.rename(columns={'CCI_44_COMBINED': 'candidate',
                                      'MERRA2': 'reference'})

    ts = ts_full.loc['1991-08-15':'1998-01-01'].dropna()
    obj = LinearRegression(ts[['candidate']], ts[['reference']], filter_p=None,
                           fit_intercept=True, force_implementation=None)

    obj.plot()
    for implementation in ['lsq_stats', 'lsq_default', 'lsq_sklearn']:
        obj = LinearRegression(ts[['candidate']], ts[['reference']], filter_p=5,
                               force_implementation=implementation)
        pprint(obj.get_model_params())
        pprint(obj.sse())
        pprint(obj.mse())
        pprint(obj.residuals_autocorr(lags=range(5)))
        obj.plot()


if __name__ == '__main__':
    usecase()
