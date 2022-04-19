# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pybreaks.break_test import TsRelBreakBase
import pandas as pd
import numpy as np
from datetime import datetime
from pybreaks.utils import df_conditional_temp_resample
from pybreaks.model_poly_regress import HigherOrderRegression
from scipy.interpolate import interp1d
import statsmodels.api as sm
from scipy.stats import ttest_rel, pearsonr
import os
from pybreaks.lmoments_ks import FitCDF


'''
Quanitify differences in 2 parts of a time series using the Higher Order Moments
method (Della-Marta et al., 2005). In the according correction class the
quantification is then used to match the 2 parts of the time series accordingly.
The implementation of the method allows using different values for the
quantification than for the correction, i.e. we assume that the corrections can
be applied to an extended set of input values.
'''

# TODO:
#   (++) OBS-PRED leads to negative corrections (must be substraced), use PRED-OBS to
#       derive corrections that can be added.
#   (+) WAK and GPA seem not to work properly in the lmoments package.
#   (+) lookup in CDF fails if the quantiles are very different (synthetic use case)

# NOTE:
#   - Selection of poly order could be based on comparing distributions of residuals
#     with models of different order (ttest, significantly lower mean of residuals)


def quant_bin(cdf, ds, n_bins=None):
    """
    Use the passed cdf to quantize the passed observations respectively.

    Parameters
    -------
    cdf : FitCDF
        CDF object that is used to look up the quantiles for ds
    ds : pd.Series
        Values that are looked up
    n_bins : int, optional (default : None)
        Number of equally sized bins that the the quantized values are distributed into

    Returns
    -------
    df_binned : pd.DataFrame
        Dataframe with the quantile and bin for each passed value
    """

    # use the cdf from can_train to calculate the quantiles for can_pred
    x = ds.values
    quant = cdf.calc_quantiles(x)

    df = ds.to_frame('can_pred')
    df['quant'] = quant

    if n_bins is not None:
        # bin them by deciles
        bins = np.linspace(0.1, 1., n_bins)
        binned = np.digitize(quant, bins, right=True).astype(float) / 10.
        df['bin'] = binned

    return df


class HigherOrderMoments(TsRelBreakBase):
    """
    This module adjusts 1 part of a time series to the other part based on
    the HOM method as proposed by P.M. Della-Marta et al. (2005).
    It uses non linear models to estimate the relationship between the CAN and REF
    series. The differences between the modeled and the observed CAN are binned
    by the deciles of the CAN CDF they fall in to, to define adjustments per
    cdf_deciles, which should remove the break between the 2 periods.

    This is independent of any (or none) breaks in the time series.
    Only together with the BreakTest class (as implemented in TsRelBreakAdjust)
    we can use this method to adjust breaks in a time series.
    """

    # Properties
    _ref_regress = None # Reference time period regression fit
    _obs_regress = None # Other time period regression fit


    def __init__(self, candidate, reference, breaktime,
                 regress_resample=None, bias_corr_method='linreg',
                 filter=None, adjust_group=0, poly_orders=[1, 2],
                 select_by='R', cdf_types=None):

        """
        Initialize the HOM adjustment object and find the best fitting model
        wrt the candidate and reference data in the group that is NOT adjusted.
        Before that apply bias correction to the passed values and resample and
        filter the model inputs.

        Parameters
        ----------
        candidate : pd.Series
            Pandas Series containing the candidate time series
        reference : pd.Series
            Pandas Series containing the reference time series
        breaktime : datetime
            Time that separates the 2 homogeneous sub periods
        regress_resample : tuple, optional (default: None)
             e.g.: ('M', 0.3))
             time period and minimum data coverage in this period for resampling
             candidate and reference before generating regression models from
             the resampled data.
        bias_corr_method : str, optional (default: 'linreg')
            Perform bias correction on the reference before modelling.
            Method for bias correction of reference data, as defined in pytesmo
        filter : tuple or None (default = None)
            e.g.: ('both', 5)
            (parts to filter, quantile)
                parts to filter : which part of the input data (wrt to break time)
                is being filtered
                ('first', 'last', None, or 'both')
            percent threshold : the amount of values that will be dropped
            (based on difference to reference)
        adjust_group : int, optional (default = 0)
            What part of the candidate should be adjusted
            (0 = before breaktime, 1 = after breaktime)
        poly_orders : list or int, optional (default: [1,2])
            Poly. degrees that are tested to find the best fitting model in terms
            of minimizing the RSS.
        select_by : str or None, optional (default : 'R')
            Method to use to find the best fitting model for
            Either ttest or R or None (requires single option for poly_orders).
        cdf_types : list or None, optional (default: None)
            List of potential CDFs that are tested, if None are passed, all
            implemented ones are used. This should correspond with the Adjustment
            Class.
        """

        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method, dropna=True)

        self.filter = (None, None) if filter is None else filter
        self.cdf_types = cdf_types

        self.adj_group, self.ref_group = self._check_group_no(adjust_group)

        self.df_original = self.df_original.rename(
            columns={self.candidate_col_name: 'can', self.reference_col_name: 'ref'})

        self.regress_resample = regress_resample

        best_p = self._init_ref_regress(select_by, poly_orders)
        # this is optional
        self._obs_regress = self._calc_model(group=self.adj_group, poly_order=best_p)

        # candidate data that was used for model generation
        self.obs = self.get_group_data(self.adj_group, self.df_original, 'all')
        self.train = self.get_group_data(self.ref_group, self.df_original, 'all')

        self.cdf_can_train = FitCDF(np.array(self.train['can'].values, dtype=np.float64),
                                    self.cdf_types)
        # this is optional
        self._cdf_can_obs = FitCDF(np.array(self.obs['can'].values, dtype=np.float64),
                                   self.cdf_types)
        # create predictions of the candidate with the reference using the model
        self.obs['can_pred'] = self._init_make_predictions(
            regress=self._ref_regress, ds=self.obs['ref'])

        self._cdf_can_pred = FitCDF(np.array(self.obs['can_pred'].values, dtype=np.float64),
                                    self.cdf_types)
        # find differences between the predictions and the observations in the core
        self.obs['residuals'] = self.obs['can'] - self.obs['can_pred']

        self.adjusted_col_name = None  # not yet adjusted

    def _init_make_predictions(self, regress, ds):
        """
        Predict the candidate for the adjust period using the reference for the
        adjust period and the regression model.

        Parameters:
        regress : HigherOrderRegression
            The regression object that contains the model to use to make predictions
        ds : pd.Series
            Reference observations that are used to create predictions with the model

        Returns
        -------
        pred : pd.Series
            The predicted candidate values
        """
        ds = ds.dropna()
        df = pd.DataFrame(index=ds.index, data={'input': ds.values})
        X = df.dropna()['input'].values
        X = X.reshape(-1, 1)
        can_pred = regress.model.predict(regress.poly_features.fit_transform(X))
        df['pred'] = can_pred

        return can_pred

    def _init_ref_regress(self, select_by, poly_orders):
        if select_by == 'ttest':
            # find the model with the smallest median squared error from the passed
            # ensemble of possible polynomials if the mean error is significantly lower.
            self._ref_regress, best_p, ref_perf = \
                self._best_model_ttest(group=self.ref_group, poly_orders=poly_orders,
                    alpha=0.05)
        elif select_by == 'R':
            # Use linear correlation to decide whether a linear or quadratic model
            # fits better.
            if poly_orders != [1,2]:
                raise ValueError(poly_orders, 'Deciding by R needs poly_orders=[1,2]')
            self._ref_regress, best_p, ref_perf = \
                self._best_model_R(group=self.ref_group, thresR=0.8, thresP=0.05)
        elif not select_by:
            # Force using the passed regression order.
            if isinstance(poly_orders, int):
                poly_orders = [poly_orders]
            if len(poly_orders) > 1:
                raise ValueError(poly_orders,
                    "Either specify the poly order or select a method for selection")
            best_p = poly_orders[0]
            self._ref_regress = self._calc_model(group=self.ref_group,
                                                poly_order=best_p)
        else:
            raise ValueError(select_by, "Unknown method to detect the polynomial regression order.")

        return best_p

    def _best_model_R(self, group, thresR=0.8, thresP=0.05):
        """
        Fit regression models of degree 1 or 2 to the data, based on the
        correlation find the type that fits the data best.

        Parameters
        -------
        group : int
            Group number (0 or 1 = before or after break) for which to find the
            model.
        poly_orders : list, optional (default: [1,2])
            List of poly degrees that are created and compared in terms of sse.
        thresR : float, optional (default : 0.8)
            Threshold for the linear correlation. If R_p is above this
            one, we select a linear model, else a non-linear one.
        thresR : float, optional (default : 0.05)
            Threshold for the linear correlation p value. If p is above this
            one, we select a non-linear model.

        Returns
        -------
        best_regress : HigherOrderRegression
            The regression obect of the model that was fitting best
        best_p : int
            The poly order of the best fitting model
        metric : dict
            the selected metric and how large it was for the best fitting model
        """

        data_group = self.get_group_data(group, self.df_original, ['can', 'ref'])
        can, ref = data_group.dropna()['can'].values, data_group.dropna()['ref'].values
        (R, p) = pearsonr(can, ref)
        if R >= thresR and p <= thresP:
            best_p = 1
        else:
            best_p = 2

        best_regress = self._calc_model(group=group, poly_order=best_p)

        min_m = best_regress.me()
        residuals = best_regress.df_model['errors'].dropna().values

        return best_regress, best_p, dict(metric=min_m)


    def _best_model_ttest(self, group, poly_orders=[1, 2, 3], alpha=0.05):
        """
        Fit regression models of degree 1, 2 and 3 to the data, based on the
        distribution of the residuals find the type that fits the data best.

        Parameters
        -------
        group : int
            Group number (0 or 1 = before or after break) for which to find the
            model.
        poly_orders : list, optional (default: [1,2,3])
            List of poly degrees that are created and compared in terms of sse.
        alpha : float, optional (default: 0.05)
            Compare the residual distributions and in case the metric got smaller
            check if the mean of the residuals got smaller and if it did, check
            if the decrease is statistically significant with p < alpha.

        Returns
        -------
        best_regress : HigherOrderRegression
            The regression obect of the model that was fitting best
        best_p : int
            The poly order of the best fitting model
        metric : dict
            the selected metric and how large it was for the best fitting model
        """

        min_m = np.inf
        best_p = None
        best_regress = None
        best_model_residuals = None

        if any([p > 3 for p in poly_orders]):
            raise ValueError('Only polynomials up to degree 3 are supported')

        for p in sorted(poly_orders):
            # calculate the required model (from data that is NOT adjusted)
            regress = self._calc_model(group=group, poly_order=p)
            m = regress.me()
            residuals = regress.df_model['errors'].dropna().values

            if best_model_residuals is None:  # always accept if there's nothing to compare to
                min_m = m
                best_p = p
                best_regress = regress
                best_model_residuals = residuals
            else:
                # test if the mean of the residuals is really different
                t_stat, p = ttest_rel(best_model_residuals, residuals)

                if p / 2 <= alpha:  # they are different, here use p/2 as we are only
                    # interested in the cases that the mean got smaller.
                    if np.less(m, min_m):  # test if the mean error is lower
                        # Accept Ha, reject H0 --> the higher order model has
                        # significantly lower mean of residuals
                        min_m = m
                        best_p = p
                        best_regress = regress
                        best_model_residuals = residuals
                    else:
                        print('Mean of residuals did not decrease, use the lower order model')
                        pass
                else:
                    print('No significant difference in means of the residuals, use lower order model.')
                    pass

        return best_regress, best_p, dict(metric=min_m)

    def _resample(self, rescale=True):
        """
        Resample the main data frame to the selected temporal resolution.
        Rescale then the reference column again.

        Parameters
        -------
        rescale : bool, optional (default: True)
            Re-scale the resampled time series, to correct for biases different
            to the original series.

        Returns
        -------
        df_adjust_resampled : pandas.DataFrame
            The resampled, re-scaled data frame.
        """

        df_adjust_resampled = \
            df_conditional_temp_resample(self.df_original,
                                         self.regress_resample[0],
                                         self.regress_resample[1])
        # if data was resampled, do bias correction again?
        if self.bias_corr_method and rescale:
            df_adjust_resampled[self.reference_col_name] = \
                self._reference_bias_correction(df_adjust_resampled,
                                                self.bias_corr_method)

        return df_adjust_resampled

    @property
    def ref_regress(self):
        return self._ref_regress

    @property
    def obs_regress(self):
        return self._obs_regress

    def plot_adjustments(self, *args, **kwargs):
        """
        (Box) Plot of the (binned) residuals and the LOWESS fit

        Parameters
        -------
        box : bool, optional (default:True)
            Plot the quantile binned boxes
        ax : plt.Axes, optional (default: None)
            Use this axis object instead to create the plot in
        raw_plot : bool
            Create a plot without labels and legend
        """
        try:
            return self.adjust_obj.plot_adjustments(*args, **kwargs)
        except AttributeError:
            raise AttributeError('No adjustments are yet calculated')

    def plot_models(self, plot_both=True, image_path=None, axs=None,
                    supress_title=False):
        """
        Create a figure of the 2 regression models, save it if a path is passed.

        Parameters
        -------
        plot_both : bool, optional (default: True)
            Plot also model for the group that is predicted (model not used)
        image_path : str, optional (default: None)
            Path where the figure is stored at, if None is passed no fig is stored.
        axs : list, optional (default: None)
            To use preset axes, pass them here, must contain exactly 2 axes
        supress_title : bool, optional (default: False)
            Do not create plot title if this is set to True.

        Returns
        -------
        plot_coll_figure : plt.figure
            The figure object
        """
        if axs != None:
            if len(axs) != 2:
                raise IOError('Wrong number of axes passed')
            else:
                if image_path:
                    raise ValueError('Cannot store the plot if axs are passed')
                else:
                    plot_coll_fig = None
        else:
            plot_coll_fig = plt.figure(figsize=(3, 5),
                                       facecolor='w', edgecolor='k')

            axs = [plot_coll_fig.add_subplot(1, 2, 1), plot_coll_fig.add_subplot(1, 2, 2)]

        # Create scatter plots for the 2 groups
        model_ax = axs[self.ref_group]
        model_ax = self._ref_regress.plot(ax=model_ax)
        title = self._model_plots_title_first_row(group_no=self.ref_group)
        if not supress_title:
            model_ax.set_title(title, fontsize=10)

        if plot_both:
            model_ax = axs[self.adj_group]
            model_ax = self._obs_regress.plot(ax=model_ax)
            if not supress_title:
                title = self._model_plots_title_first_row(group_no=self.adj_group)
                model_ax.set_title(title, fontsize=10)

        if image_path:
            file_path = os.path.join(image_path, '%s_model_plots.png'
                                     % str(self.breaktime.date()))

            if not os.path.isdir(image_path):
                os.mkdir(image_path)

            plot_coll_fig.tight_layout()
            plot_coll_fig.savefig(file_path, dpi=300)
            plt.close()

        return plot_coll_fig

    def _model_plots_title_first_row(self, group_no):
        """
        Returns titles for scatter plot first row visualizations, depending if
        the models are calculated from filtered or unfiltered, resampled or
        un-resampled data, and for which group.

        Parameters
        ----------
        group_no : int
            Group number (0 or 1) of group to create scatter plot title for

        Returns
        -------
        title : str
            Title for the plot with the current class configurations
        """

        if self.regress_resample:
            rs_prefix = ' %s-Resampled' % self.regress_resample[0]
        else:
            rs_prefix = 'Unresampled'

        if self.filter[0] in ['first', 'both']:
            g0 = 'Filtered Group'
            g1 = g0 if self.filter[0] == 'both' else 'Unfiltered Group'
        elif self.filter[0] in ['last', 'both']:
            g1 = 'Filtered Group'
            g0 = g1 if self.filter[0] == 'both' else 'Unfiltered Group'
        else:
            g0, g1 = 'Unfiltered Group', 'Unfiltered Group'

        if self.adj_group == group_no:
            used = 'HSPa'
        else:
            used = 'HSPr'

        if group_no == 0:
            title = '%s, %s %s, %s' % (rs_prefix, g0, '(before break)', used)
        else:
            title = '%s, %s %s, %s' % (rs_prefix, g1, '(after break)', used)

        return title

    def _calc_model(self, group, poly_order=1):
        """
        Create regression models for data in the the passed groups

        Parameters
        -------
        group : int
            Create the regression model for this group (0 = before, 1=after break)
        poly_order : int
            Order of the polynom that is fitted.

        Returns
        -------
        regress : HigherOrderRegression
            The regression model object
        """

        # choose the right filter value for the group
        if (group == 0) and (self.filter[0] in ['first', 'both']):
            filter_p = self.filter[1]
        elif (group == 1) and (self.filter[0] in ['last', 'both']):
            filter_p = self.filter[1]
        else:
            filter_p = None

        data_group = self.get_group_data(group, self.df_original, ['can', 'ref'])

        if self.regress_resample is not None:
            to, thres = self.regress_resample[0], self.regress_resample[1]
            data_group = df_conditional_temp_resample(data_group, to, thres)
            data_group = data_group.dropna()

        subset_candidate = data_group['can']
        subset_reference = data_group['ref']

        regress = HigherOrderRegression(subset_candidate, subset_reference,
                                        poly_order, filter_p)

        return regress

    def _ts_props(self):
        ''' Specific for each child class '''
        props = {'isbreak': None,
                 'breaktype': None,
                 'candidate_name': self.candidate_col_name,
                 'reference_name': self.reference_col_name,
                 'adjusted_name': self.adjusted_col_name,
                 'adjust_failed': False}

        return props

    def adjust(self, values_to_adjust, use_separate_cdf=False, **adjustment_kwargs):
        """
        Adjusts the candidate column for the selected period (before, after
        break time) relatively to the reference column, so that the regression
        properties fit, then recalculates the regression model with the new
        values.

        Parameters
        -------
        values_to_adjust : pd.Series
            Must contain 2 columns, 1 for the candidate and 1 for the reference
            The names of the columns must be the same as in self.df_adjust
            If None is passed, the init. passed values are used.
        use_separate_cdf : bool, optional (default: False)
            Do not use the CDF of the candidate training set but find another
            CDF for all values_to_adjust, to which the quantile corrections
            from the core frame are applied to.

        ------adjustment kwargs--------
        alpha : float, optional (default: 0.6)
            smoothing factor for adjustment from residuals or bins
        from_bins : bool, optional (default: False)
            Interpolate the bin means for the corrections and not the residuals
            directly.

        Returns
        -------
        candidate_adjusted : pd.Series
            The adjusted input values to adjust
        """
        values_to_adjust = values_to_adjust.dropna()

        self.adjusted_col_name = self.candidate_col_name + '_adjusted'

        adjust_obj = HigherOrderMomentsAdjust(df_obs=self.obs,
                                              df_train=self.train,
                                              cdf_can_train=self.cdf_can_train,
                                              regress=self._ref_regress,
                                              **adjustment_kwargs)

        adjusted_values = adjust_obj.adjust(values_to_adjust,
                                            use_separate_cdf=use_separate_cdf,
                                            separate_cdf_types=self.cdf_types)

        group_index = self.get_group_data(self.adj_group, self.df_original, None)
        common_index = adjusted_values.index.intersection(group_index)

        self.df_original[self.adjusted_col_name] = self.df_original['can']
        self.df_original.loc[common_index, self.adjusted_col_name] = adjusted_values

        self.adjust_obj = adjust_obj

        # check if no data got lost
        assert(len(adjusted_values.values) == len(values_to_adjust.values))
        return adjusted_values

    def get_model_params(self):
        """
        Get the model parameters of the linear model for the according group
        of the current iteration.

        Returns
        -------
        model_params: dict
            Dictionary of model parameters of the selected model
        """

        model_params = self._ref_regress.get_model_params()
        try:
            cdf_type = self.adjust_obj.cdf_can_train.name
            lut = {'nor': 1, 'pe3': 2, 'gno': 3, 'gpa': 4, 'gev': 5, 'wak': 6}
            model_params['cdf_type'] = lut[cdf_type]
        except:
            pass

        # todo: return metadata

        return model_params

    def plot_cdf_compare(self, plot_empirical=True, ax=None):
        """
        Plots a collection of the CDF for the predicted and the training candidate
        data (reference period) and the observed data (adjusted period).Plots
        are combined in one subplot here.

        Parameters
        -------
        plot_empirical : bool, optional (default: True)
            Add the observations as points to the plot of the theoretical, fitted CDF.
        ax : plt.axes, optional (default: None)
            An axes object to use instead of creating a new figure and ax.

        Returns
        -------
        fig : plt.figure or None
            The figure, if one was created, else None
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        ax = self.cdf_can_train.plot_cdf(plot_empirical=plot_empirical, name='TRAIN',
                                         ax=ax, style='b-')
        ax = self._cdf_can_pred.plot_cdf(plot_empirical=plot_empirical, name='PRED',
                                         ax=ax, style='g--')
        ax = self._cdf_can_obs.plot_cdf(plot_empirical=plot_empirical, name='OBS',
                                        ax=ax, style='r:')
        ax.set_title('CDF fit before/after break')

        return fig

    def plot_pdf_compare(self, plot_empirical=False):
        """
        Plots a collection of the PDF for the predicted and the training candidate
        data (reference period) and the observed data (adjusted period). Plots
        are combined in one subplot here.

        Parameters
        -------
        plot_empirical : bool, optional (default: True)
            Add the observations as points to the plot of the theoretical, fitted CDF.
        ax : plt.axes, optional (default: None)
            An axes object to use instead of creating a new figure and ax.

        Returns
        -------
        fig : plt.figure or None
            The figure, if one was created, else None
        """
        fig, ax = plt.subplots()
        ax = self.cdf_can_train.plot_pdf(plot_empirical=plot_empirical, name='TRAIN',
                                         ax=ax, style='b-')
        ax = self._cdf_can_pred.plot_pdf(plot_empirical=plot_empirical, name='PRED',
                                         ax=ax, style='g--')
        ax = self._cdf_can_obs.plot_pdf(plot_empirical=plot_empirical, name='OBS',
                                        ax=ax, style='r:')
        ax.set_title('PDF fit before/after break')

        return fig

    def plot_cdf_compare_separate(self, plot_empirical=True):
        """
        Plots a collection of the PDF for the predicted and the training candidate
        data (reference period) and the observed data (adjusted period). Plots
        are separated in subplots here.

        Parameters
        -------
        plot_empirical : bool, optional (default: True)
            Add the observations as points to the plot of the theoretical, fitted CDF.

        Returns
        -------
        fig : plt.figure or None
            The figure that was created
        """
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1 = self.cdf_can_train.plot_cdf(plot_empirical=plot_empirical,
                                          name='TRAIN', ax=ax1, style='b-')
        ax1.set_title('CDF CAN Training')
        ax2 = fig.add_subplot(1, 3, 2)
        ax2 = self._cdf_can_pred.plot_cdf(plot_empirical=plot_empirical,
                                          name='PRED', ax=ax2, style='g--')
        ax2.set_ylabel('')
        ax2.set_title('CDF CAN Predicted')
        ax3 = fig.add_subplot(1, 3, 3)
        ax3 = self._cdf_can_obs.plot_cdf(plot_empirical=plot_empirical,
                                         name='OBS', ax=ax3, style='r:')
        ax3.set_title('CDF CAN Observed')
        ax3.set_ylabel('')

        plt.tight_layout()

        return fig


class HigherOrderMomentsAdjust(object):
    """
    Class that adjusts the candidate using the reference (for the same time period)
    and a regression model (from candidate and reference of reference period).
    """
    def __init__(self, df_obs, df_train, cdf_can_train, regress, alpha=0.6,
                 from_bins=False):
        """
        Parameters
        ----------
        df_obs : pd.DataFrame
            The homogeneous data in the period that is adjusted/predicted
        df_train : pd.DataFrame
            The homogeneous data in the period that is used to find the model
            to create predictions in the obs. period.
        cdf_train : FitCDF
            The cdf that is used to quantize the values to adjust, so that the
            corrections can be applied. This is calculated from the candidate
            observations in the period where the model comes from.
        regress : HigherOrderRegression
            regression object the contains the model that was used to make predictions.
        df_quant_ref : pd.DataFrame
            The quantized residuals and corrections to apply to the values to adjust.
        alpha : float, optional (default: 0.4)
            smoothing factor the the LOWESS fit (between 0 and 1)
        from_bins : bool, optional (default: False)
            Calculate the corrections from binned residuals instead of all residuals.
        """
        self.df_obs = df_obs
        self.df_train = df_train

        self.cdf_can_train = cdf_can_train
        self.regress = regress

        # bin the predictions based on the cdf of the training data (10 bins)
        df_obs_quant = quant_bin(cdf=self.cdf_can_train, ds=self.df_obs['can_pred'],
                                 n_bins=10)

        self.df_obs['bin'] = df_obs_quant['bin']
        df_obs_quant['residuals'] = self.df_obs['residuals']
        df_obs_quant['date'] = self.df_obs.index
        df_obs_quant = df_obs_quant.set_index('quant').sort_index()
        self.df_obs_quant = df_obs_quant

        # interpolate the residuals to get corrections for all frequencies
        self.from_bins = from_bins
        self.fadjust = self._calc_adjustments(alpha=alpha)

        self.df_obs_quant['adjustments'] = self.fadjust(self.df_obs_quant.index.values)

        self.df_obs['adjustments'] = pd.Series(index=self.df_obs_quant['date'].values,
                                               data=self.df_obs_quant['adjustments'].values)

        # Calculate this later
        self.df_adjust = None # will be calculated in function

    def plot_adjustments(self, ax=None, raw_plot=False):
        """
        (Box) Plot of the (binned) residuals and the LOWESS fit. Boxes are only
        shown when corrections come from the binned values.

        Parameters
        -------
        ax : plt.Axes, optional (default: None)
            Use this axis object instead to create the plot in
        raw_plot : bool
            Create a plot without labels and legend

        Returns
        -------
        fig : plt.figure or None
            The figure that was created
        """
        df = self.df_obs.loc[:, ('residuals', 'bin')]
        N = df['residuals'].index.size
        bins = np.unique(df['bin'].values)

        data = []
        widths = []
        for bin in bins:
            bindat = df.loc[df['bin'] == bin]
            data.append(bindat['residuals'].values.reshape(-1, 1))
            widths.append((float(bindat.index.size) / float(N)))

        # with of each box should be relative to the number of values in it
        max_width, norm_widths = max(widths), []
        for width in widths:
            norm_widths.append(width / max_width)

        if ax is None:
            fig = plt.figure(figsize=(7.5, 6))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        x = self.df_obs_quant.index.values
        ax.scatter(x, -1* self.df_obs_quant['residuals'].values, # todo: change -1
                   alpha=0.3, label='Residuals', color='grey')
        ax.plot(x, -1 * self.df_obs_quant['adjustments'], label='adjustments', color='blue',
                linestyle='--') # todo: change -1

        if not raw_plot:
            ax.set_xlabel('Quantile')
            ax.set_ylabel('SM Residuals (PRED - OBS)')

            if not self.from_bins:
                ax.set_title('HOM Adjustments', y=1.15)

        if self.from_bins:
            ax2 = ax.twiny()
            ax2.grid(False)
            x = np.linspace(0, 1, 100)
            adjustments = self.fadjust(x)
            medianprops = dict(linestyle='-', linewidth=2.5, color='black')
            ax2.boxplot([-1 * d for d in data], sym='', showmeans=False, widths=norm_widths, # todo: change -1
                        positions=np.linspace(0.5, 9.5, len(bins)).tolist(),
                        medianprops=medianprops,
                        labels=range(1, len(bins)+1))

            ax2.set_xlabel('Quantile Bin' if bins.size != 10 else 'Decile Bin')
            if not raw_plot:
                ax2.set_title('(Binned) Residuals and Adjustments', y=1.15)

            ax.set_xlim(0, 1.1)
            ax2.set_xlim(0, 11)
        plt.tight_layout()

        ax.legend()

        return fig

    def _calc_adjustments(self, alpha):
        """
        Interpolate the residuals with a linear LOESS (all values not just bin
        means)

        Parameters
        -------
        alpha : float
            Between 0 and 1. Smoothing parameter for the LOWESS fit.
            The fraction of the data used when estimating each y-value.

        Returns
        -------
        f : interp1d
            Function to calculate adjustments
        """
        df = self.df_obs_quant.loc[:, ('residuals', 'bin')]

        # frac is the smoothing parameter
        if self.from_bins:
            df = df.groupby(['bin']).mean()
            df.at[1.0] = np.nan
            z = sm.nonparametric.lowess(df['residuals'].values, df.index.values,
                                        frac=alpha)
        else:
            z = sm.nonparametric.lowess(df['residuals'].values, df.index.values,
                                        frac=alpha)

        x = list(zip(*z))[0]
        y = list(zip(*z))[1]

        f = interp1d(x, y, fill_value='extrapolate', bounds_error=False)

        return f

    def adjust(self, values_to_adjust, use_separate_cdf=False, separate_cdf_types=None):
        """
        Takes a set of values to adjust and applies corrections based on the
        corrections that were found for the frame from the residuals.
        The cdf of the candidate training part is used to apply the corrections to
        if not chosen to calculate a new CDF for the values to adjust.

        Parameters
        -------
        values_to_adjust : pd.Series
            Observations that are corrected
        use_separate_cdf : bool, optional (default: False)
            Do not use the CDF of the candidate training set but find a separate
            L-moments CDF for the values to adjust to apply the quantile corrections
            to.
        separate_cdf_types : list or None, optional (default: None)
            List of potential CDFs that are tested, if None are passed, all
            implemented ones are used. This should correspond with the Adjustment
            Class.

        Returns
        -------
        ds_adjusted : pd.Series
            The adjusted input values to adjust.
        """
        if use_separate_cdf:
            # lookup_cdf is used to find corresponding observations for the corrections
            lookup_cdf = FitCDF(values_to_adjust.values, types=separate_cdf_types)
        else:
            lookup_cdf = self.cdf_can_train

        values_to_adjust = pd.DataFrame(index=values_to_adjust.index,
                                        data={'can': values_to_adjust})
        df_adjust = quant_bin(lookup_cdf, values_to_adjust['can'], n_bins=None)

        df_adjust['date'] = df_adjust.index
        df_adjust = df_adjust.set_index('quant').sort_index()
        # find adjustments for the core only, this is not necessary
        df_adjust['adjustments'] = self.fadjust(df_adjust.index.values)

        self.df_adjust = df_adjust
        # find the adjustment for each observation accordingly
        values_to_adjust['adjustments'] = df_adjust.set_index('date').sort_index()['adjustments']

        self.adjustments = values_to_adjust['adjustments'] # TODO: *-1 to add them?

        values_to_adjust['adjusted'] = values_to_adjust['can'] - self.adjustments # todo: + instead of -



        return values_to_adjust['adjusted']


if __name__ == '__main__':
    pass