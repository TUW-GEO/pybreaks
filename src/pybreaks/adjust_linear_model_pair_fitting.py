# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pybreaks.base import TsRelBreakBase
from pybreaks.model_lin_regress import LinearRegression

import pandas as pd
import numpy as np
from datetime import datetime
from pybreaks.utils import df_conditional_temp_resample, \
    days_in_month, mid_month_target_values
import os
from scipy.interpolate import interp1d
import calendar
from collections.abc import Iterable

'''
Module implements the Linear Model Pair adjustment, based on the suggestions
by Su (2016, SI). It uses the differences between model parameters of 2 linear
regression models to find corrections for daily/monthly observations that are 
interpolated and upsampled to be applied to an extended set of values to adjust.

TODO #################################
(+) find way to automatically detect the candidate and values to adjust 

NOTES ################################
-
'''


class RegressPairFit(TsRelBreakBase):
    '''
    This module adjusts 1 part of a time series to the other part based on
    2 regression models, so that afterwards the regression parameters match.
    This is independent of any (or none) breaks in the time series.
    Only together with the BreakTest class (as implemented in TsRelBreakAdjust)
    we can use this method to adjust breaks in a time series.
    '''

    def __init__(self, candidate, reference, breaktime, candidate_freq='D',
                 regress_resample=None, bias_corr_method='linreg',
                 filter=('both', 5), adjust_group=0,
                 model_intercept=True):

        """
        Initialize the LMP break adjustment object

        Parameters
        ----------
        candidate : pd.Series or pd.DataFrame
            Pandas object containing the candidate time series
        reference : pd.Series or pd.DataFrame
            Pandas object containing the reference time series
        breaktime : datetime
            Time to test for a break
        candidate_freq : str, optional (default: D)
            Temporal resolution of candidate input (M or D) of this class
        regress_resample : tuple, optional (default: None)
             Time period and minimum data coverage in this period for resampling
             candidate and reference before generating regression models from
             the resampled data. eg ('M', 0.3)
        bias_corr_method : str or None, optional (default: 'linreg')
            Name of the method that is used to correct the bias between
            candidate and reference. Currently supported methods are
            'linreg' (for scaling the reference to the candidate based on
            fitting slope and intercept of the 2 series) and 'cdf_match'
            (for scaling the reference to the candidate based on fitting
            percentiles 0, 5, 10, 30, 50, 70, 90, 95 and 100 of the reference
            to the candidate with linear interpolation.)
        filter : tuple or None, optional (default: ('both', 5))
            (parts to filter , percentile)
                parts to filter : which part of the input data (wrt to the break time)
                is being filtered
                    ('first', 'last', None, or 'both')
            percent threshold : 0 = No filtering, 100 = all removed
            (based on difference to reference)
        adjust_group : int
            What part of the candidate should be adjusted
            (0 = before breaktime, 1=after breaktime)
        model_intercept : bool, optional (default: True)
            Create a 2 parameter linear model, if this is False, only the slope
            will be modeled (intercept is 0) and matched.
        -------
        """
        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method, dropna=True)

        self.filter = (None, None) if filter is None else filter

        self.regress_resample = regress_resample

        self.adjust_group, self.other_group = self._check_group_no(adjust_group)

        # if chosen, perform temporal resampling
        if self.regress_resample:
            self.df_adjust = self._resample(rescale=True)
            self.df_adjust.freq = self.regress_resample[0]
        else:
            self.df_adjust = self.df_original.copy(True)
            self.df_adjust.freq = candidate_freq

        # calculate the 2 models
        self.model0, self.model1 = self._calc_regress_models(model_intercept)

        # will be created later:
        self.adjust_obj = None
        self.adjusted_col_name = None  # this indicates if adjustment was perfomed

    def _resample(self, rescale=True):
        """
        Resample the main data frame to the selected temporal resolution.
        Rescale then the reference column again.

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
            rs_prefix = '%s-Resampled' % self.regress_resample[0]
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
        if group_no == 0:
            title = '%s, %s %s' % (rs_prefix, g0, '(before break)')
        else:
            title = '%s, %s %s' % (rs_prefix, g1, '(after break)')

        return title

    def _calc_regress_models(self, model_intercept=True):
        """
        Model objects input data before/after break time via linear model.

        Parameters
        -------
        model_intercept : bool
            Set True to model slope and inter, False to model slope only
            (inter is then 0, and so is inter diff between the models)
        Returns
        -------
        model0 : LinearRegression
            The linear model for the FIRST part (before break)
        model1 : LinearRegression
            The linear model for the SECOND part (after break)

        """
        model0 = None
        model1 = None

        for i in [0, 1]:
            if (i == 0) and (self.filter[0] in ['first', 'both']):
                filter_p = self.filter[1]
            elif (i == 1) and (self.filter[0] in ['last', 'both']):
                filter_p = self.filter[1]
            else:
                filter_p = None

            data_group = self.get_group_data(i, self.df_adjust,
                                             [self.candidate_col_name,
                                              self.reference_col_name])

            subset_candidate = data_group[self.candidate_col_name]
            subset_reference = data_group[self.reference_col_name]

            model = LinearRegression(subset_candidate, subset_reference, filter_p,
                                     fit_intercept=model_intercept)

            if i == 0:
                model0 = model
            else:
                model1 = model

        return model0, model1

    def _ts_props(self):
        """Specific for each child class"""
        props = {'isbreak': None,
                 'breaktype': None,
                 'candidate_name': self.candidate_col_name,
                 'reference_name': self.reference_col_name,
                 'adjusted_name': self.adjusted_col_name,
                 'adjust_failed': False}

        return props

    def plot_adjustments(self, *args, **kwargs):
        """
        Crate a plot of the fitted adjustments (interpolated residuals from
        before and after applying model correction factors) that are finally
        applied to the observations.

        Parameters
        -------
        ax : plt.Axes, optional (default: None)
            Use this axis object instead to create the plot in

        Returns
        -------
        fig : plt.figure or None
            The figure that was created
        """
        try:
            return self.adjust_obj.plot_adjustments(*args, **kwargs)
        except AttributeError:
            raise AttributeError('No adjustments are yet calculated')


    def plot_models(self, image_path=None, axs=None, supress_title=True):
        """
        Create a figure of the 2 linear models, save it if a path is passed.

        Parameters
        -------
        image_path : str
            File that the figure is saved as
        axs : list
            To use preset axes, pass them here, must contain 2 axes
        supress_title : bool
            Do not plot a title

        Returns
        -------
        plot_coll_figure : plt.figure
            The figure object
        axs : plt.axes
            Axes of the figure
        """
        models = [self.model0, self.model1]

        if axs is not None:
            if not (isinstance(axs, Iterable) and (len(axs) == len(models))):
                raise Exception('Wrong number of axes passed')
            else:
                if image_path:
                    raise Exception('Cannot store the plot if axs are passed')
                else:
                    plot_coll_fig = None
        else:
            plot_coll_fig = plt.figure(figsize=(10, 4),
                                       facecolor='w', edgecolor='k')

        vmax = -999
        for model in models:
            vmax = model.df_model.max().max()

        # Create scatter plots for the 2 groups
        for i, model in enumerate(models):
            if plot_coll_fig is not None:
                ax = plot_coll_fig.add_subplot(1, len(models), i + 1)
            else:
                ax = axs[i]

            _ = models[i].plot(ax=ax, vmax=vmax)

            if not supress_title:
                ax.set_title(self._model_plots_title_first_row(group_no=i), fontsize=12)

        if image_path:
            file_path = os.path.join(image_path, '%s_model_plots.png'
                                     % str(self.breaktime.date()))

            if not os.path.isdir(image_path):
                os.mkdir(image_path)

            # plot_coll_fig.tight_layout()
            plot_coll_fig.savefig(file_path)
            plt.close()

        return plot_coll_fig

    def get_model_params(self, model_no=None, only_1d=True):
        """
        Get the model parameters of the linear model for the according group
        of the current iteration.

        Parameters
        ----------
        model_no : int (0 or 1) or None
            Number of the model (0=before break, 1=after break)
        only_1d : bool
            Exclude time series values (like residuals)

        Returns
        -------
        model_params: dict
            Dictionary of model parameters of the selected model
        """

        if model_no == 0:
            return self.model0.get_model_params(only_1d)
        elif model_no == 1:
            return self.model1.get_model_params(only_1d)
        else:
            return {'model0': self.model0.get_model_params(only_1d),
                    'model1': self.model1.get_model_params(only_1d)}

    @staticmethod
    def _apply_adjustments(values_to_adjust, adjustments):
        """
        Apply adjustments to values with the respective dates.
        Parameters
        ----------
        values_to_adjust : pd.Series
            Values where the adjustments are applied
        adjustments : pd.Series
            Adjustments that are applied

        Returns
        -------
        adjusted_values : pd.Series
            Values with applied adjustments
        """
        comm_index = adjustments.index.intersection(values_to_adjust.index)

        if not all(list(np.in1d(values_to_adjust.index, comm_index))):
            raise ValueError('Adjustments dont fit to passed values')

        df = values_to_adjust.to_frame('vals')
        df['adjustments'] = adjustments.loc[comm_index]

        df['adjusted'] = df['vals'] + df['adjustments']
        return df['adjusted']

    def adjust(self, values_to_adjust=None, corrections_from_core=True,
               values_to_adjust_freq='D', resample_corrections=True,
               interpolation_method='linear'):
        """
        Adjusts the passed values based on differences in the linear models of the
        object.

        Parameters
        -------
        values_to_adjust : pd.Series, optional (default: None)
            Values to which the additive (intercept diff) and multiplicative
            (slope ratio) correction parameters are applied.
        corrections_from_core : bool, optional (default: True)
            Apply the model corrections to the values from the homogeneous core
            instead to all values_to_adjust. Then applies the corrections
            to all values_to_adjust. If this is not selected the corrections are
            applied directly and the adjustments are not repeated to the extended
            time frame.
        values_to_adjust_freq : str, optional (default: D)
            Temporal resolution of the values that should be adjusted (D or M)
        resample_corrections : bool, optional (default: True)
            Adjustments are M-resampled then interpolation to match the
            temporal resolution of the daily values_to_adjust.
        interpolation_method : str, optional (default: linear)
            To fit constant, linear or a spline pass the method as for scipy.interp1d
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
            To fit a least squares polynomial, pass "polyN" where N is the order of
            the polynomial.

        Returns
        -------
        adjusted_values : pd.Series
            The passed values after applying the corrections from to model
            differences.
        """
        candidate_col_name = self.candidate_col_name
        self.adjusted_col_name = self.candidate_col_name + '_adjusted'

        if self.regress_resample:
            extended = True
        else:
            extended = False

        if corrections_from_core:
            # get the core values, that are homogeneous
            can_adj = self.get_group_data(self.adjust_group, self.df_adjust,
                                          candidate_col_name, extended)
            if values_to_adjust is None:  # adjust the core values if noting else is passed
                values_to_adjust = can_adj.copy(True)
        else:
            if values_to_adjust is None:
                raise ValueError('If corrections_from_core is not selected, '
                                 'values_to_adjust must be passed')
            else:
                # use the passed values
                can_adj = values_to_adjust

        assert values_to_adjust is not None
        assert can_adj is not None

        if self.adjust_group == 0:
            adj_model = self.model0
            ref_model = self.model1
        else:
            adj_model = self.model1
            ref_model = self.model0

        self.adjust_obj = PairRegressMatchAdjust(can_adj=can_adj,
                                                 adj_model=adj_model,
                                                 ref_model=ref_model)

        adjustments = self.adjust_obj.calc_adjustments(
            startdate=values_to_adjust.index[0].to_pydatetime(),
            enddate=values_to_adjust.index[-1].to_pydatetime(),
            resample_corrections=resample_corrections,
            interpolation_method=interpolation_method,
            values_to_adjust_freq=values_to_adjust_freq)

        adjusted_values = self._apply_adjustments(values_to_adjust, adjustments)

        common_index = self.df_original.index.intersection(adjusted_values.index)

        self.df_original[self.adjusted_col_name] = self.df_original[self.candidate_col_name]
        self.df_original.loc[common_index, self.adjusted_col_name] = adjusted_values

        return adjusted_values


class PairRegressMatchAdjust(object):
    """
    Class that adjusts the candidate data based on the passed slope and intercept
    adjustment values of 2! regression models (before and after a break), which
    are matched for the candidate group.
    Takes a set of values, to which the model corrections are then applied
    Optionally the adjustments can be upsampled, eg when monthly values are passed
    to get daily adjustments (this needs a start and end date for extrapolation)
    """

    def __init__(self, can_adj, adj_model, ref_model):
        """
        Parameters
        ----------
        can_adj : pandas.Series
            Values that are used to calculate adjustments from. eg monthly
            observations with interpolation to daily or directly daily
            observations without interpolation.
            To these values the adjustment parameters are then applied.
        adj_model : LinearRegression
            Parameters of the model for the part that will be adjusted,
            must contain a key 'slope' and a key 'inter'
        ref_model : LinearRegression
            Parameters of the model for the part that will not be adjusted,
            (adj is fit to) must contain a key 'slope' and a key 'inter'
        """

        df_adjust = pd.DataFrame(data={'can': can_adj.copy(True)})

        self.df_adjust = df_adjust  # type: pd.DataFrame

        # Calculate the correction parameters
        self.slope_ratio, self.inter_diff = self._corr_params(adj_model,
                                                              ref_model)

        self.correction_params = {'slope_ratio': self.slope_ratio,
                                  'intercept_diff': self.inter_diff}

        # will be calculated later
        self.samples, self.adjustments = None, None

    @staticmethod
    def _corr_params(adj_model, ref_model):
        """
        Calculate differences in the model parameters for the 2 passed models.

        Parameters
        ----------
        adj_model : LinearRegression
            Model parameters of the model from data that is adjusted
            must contain 'slope' and 'inter' as keys and valid values.
        ref_model : LinearRegression
            Model parameters of the model for data that is not adjusted
            must contain 'slope' and 'inter' as keys and valid values.

        Returns
        -------
        slope_ratio : float
            Ratio of the slopes of the 2 models based on which model is to be
            corrected later.
        inter_diff : float
            Difference in the models intercepts, based on which model is to be
            corrected later.
        """
        adj_model_params = adj_model.get_model_params(True)
        ref_model_params = ref_model.get_model_params(True)

        slope_ratio = ref_model_params['slope'] / adj_model_params['slope']
        inter_diff = ref_model_params['inter'] - slope_ratio * adj_model_params['inter']

        return slope_ratio, inter_diff

    @staticmethod
    def _interpolate(start, end, samples, interpolation_method):
        # old and not used anymore, daily adjustments from upsampled M makes no sense
        """
        Create daily values from monthly ones by interpolation

        Parameters
        ----------
        start : datetime
            First date of the interpolated series
        end : datetime
            Last date of the interpolted series
        samples : pd. Series
            Some values and dates that are used as nodes for the interpolation
        interpolation_method :
            To fit constant, linear or a spline pass the method as for scipy.interp1d
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
            To fit a least squares polynomial, pass "polyN" where N is the order of
            the polynomial.

        Returns
        -------

        """
        index = pd.date_range(start=start, end=end, freq='D')
        # add the samples
        df = pd.DataFrame(index=index)
        df['x'] = range(len(df.index.values))
        df['samples'] = samples

        if 'poly' not in interpolation_method:
            f = interp1d(df.dropna()['x'].values, df.dropna()['samples'].values,
                         kind=interpolation_method,
                         fill_value="extrapolate")

            x_new = df['x'].values
            f_new = f(x_new)
            df['inter'] = f_new
        else:
            deg = interpolation_method[4:]
            z = np.polyfit(df.dropna()['x'].values, df.dropna()['samples'].values,
                           deg=deg)
            p = np.poly1d(z)

            df['inter'] = p(df['x'].values)

        return df['inter']

    def _interpol_samples(self, samples, interpolation_method):
        start = datetime(samples.index[0].year, samples.index[0].month, 1)
        end = datetime(samples.index[-1].year, samples.index[-1].month,
                       int(days_in_month(samples.index[-1].month, samples.index[-1].year)))
        interpolated = self._interpolate(start=start, end=end,
                                         samples=samples,
                                         interpolation_method=interpolation_method)
        return interpolated

    def calc_adjustments(self, startdate, enddate, resample_corrections=True,
                         interpolation_method='linear', values_to_adjust_freq='D'):
        """
        TODO: clean up

        Adjusts the candidate column for the selected period (before, after
        break time) relatively to the reference column, so that the regression
        properties would fit.

        Parameters
        -------
        startdate : datetime
            First datetime for which we find adjustmetns
        enddate : datetime
            Last date for which we find adjustments
        resample_corrections: bool
            Adjustments are M-resampled before interpolation/before application to the input
        interpolation_method : str or None, optional (default: poly2)
            To fit constant, linear or a spline pass the method as for scipy.interp1d
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
            To fit a least squares polynomial, pass "polyN" where N is the order of
            the polynomial.

        Returns
        -------
        adjustments : pd.Series
            Adjustments to apply to the observations.
        """
        if values_to_adjust_freq == 'M':
            resample_corrections = True

        df = self.df_adjust.copy(True)  # type: pd.DataFrame

        df['dd'] = self.correction_params['intercept_diff']
        df['cc'] = np.abs(self.correction_params['slope_ratio'])

        data_adjusted = df['cc'] * df['can'] + df['dd']

        # These are the adjustments for each value
        adjustments = data_adjusted - df['can']

        if resample_corrections:
            # Create a MOY series from adjustments (mean of each unique month)
            m_samples = adjustments.groupby(adjustments.index.month).mean()
            first_m, last_m = m_samples.index[0], m_samples.index[-1]
            # the year 1904 is just a placeholder here (it is a leap year)
            index = pd.date_range(
                start = datetime(1904, first_m, 1),
                end = datetime(1904, last_m, days_in_month(last_m, 1904, int)),
                freq='D')

            samples = pd.Series(index=index)
            for m in m_samples.index:
                str_m = str(m).zfill(2) if len(str(m)) == 1 else str(m)
                samples.loc['1904-{}-15'.format(str_m)] = m_samples.loc[m]

            if m_samples.index[-1] == 12:
                samples.loc[samples.index[-1]] = m_samples.loc[m_samples.index[0]]  # last day same as first day
            if m_samples.index[1] == 11:
                samples.loc[samples.index[0]] = m_samples.loc[m_samples.index[0]]  # last day same as first day
        else:
            # Do not resample, but find a DOY series (mean of each unique DOY)
            samples = adjustments.groupby([adjustments.index.month, adjustments.index.day]).mean()
            index = pd.DataFrame({'month': samples.index.get_level_values(0),
                                  'day': samples.index.get_level_values(1),
                                  'year': 1904})
            samples.index = pd.to_datetime(index).values

        if resample_corrections:
            try:  # try the midmonth target values (only possible for 12 months)
                s = samples.dropna()
                i_f, fst = s.index[0], s.iloc[0]
                i_l, lst = s.index[-1], s.iloc[-1]
                s = s[1:-1]
                s = pd.Series(index=s.index, data=mid_month_target_values(s))
                s.at[i_f] = fst
                s.at[i_l] = lst
            except ValueError:
                s = samples.dropna()

                i = 0
                while s.index.month[0] > 1:  # fill
                    if i > 100:
                        break
                    nd = datetime(1904, s.index.month[0] - 1, 15)
                    s[nd] = np.nan
                    s = s.sort_index()
                    i += 1

                i = 0
                while s.index.month[-1] != 12:  # fill
                    if i > 100:
                        break
                    nd = datetime(1904, s.index.month[-1] + 1, 15)
                    s[nd] = np.nan
                    s = s.sort_index()
                    i += 1

                if s.index.month[-1] == 12:
                    dom = days_in_month(12, 1904, int)
                    nd = datetime(1904, 12, dom)
                    s[nd] = np.nan
                    s = s.sort_index()
                if s.index.month[0] == 1:
                    nd = datetime(1904, 1, 1)
                    s[nd] = np.nan
                    s = s.sort_index()

                idx = pd.date_range(start='1904-01-01',
                                    end='1904-12-31',
                                    freq='D')
                s = s.reindex(idx, fill_value=np.nan)
                samples = s.sort_index()
        # if there are nans in the DOY or MOY series, interpolate them
        # always linear here, this is just a gap filling and there should
        # not be many values missing
        adjustments = self._interpol_samples(
            samples, interpolation_method=interpolation_method)

        self.samples = samples.copy(True)
        self.adjustments = adjustments.copy(True)  # store before filling the whole period

        # re-distribute to the full time range necessary for applying
        df_years = []
        first_y, last_y = startdate.year, enddate.year
        for year in range(first_y, last_y + 1):
            adj = adjustments.copy(True)
            if not calendar.isleap(year):  # drop feb 29th
                adj.drop(labels=adj.loc[adj.index == '1904-02-29'].index,
                         axis=0, inplace=True)
            try:
                index = pd.DataFrame({'year': year, 'month': adj.index.month, 'day': adj.index.day})
            except AttributeError:
                index = pd.DataFrame({'year': year, 'month': adj.index.values,
                                      'day': days_in_month(adj.index.values, year, float)})

            i = pd.to_datetime(index).values

            df_years.append(pd.Series(index=i, data=adj.values))
        adjustments = pd.concat(df_years, axis=0)

        return adjustments

    def plot_adjustments(self, ax=None):
        """
        Crate a plot of the fitted adjustments (interpolated residuals from
        before and after applying model correction factors) that are finally
        applied to the observations.

        Parameters
        -------
        ax : plt.Axes, optional (default: None)
            Use this axis object instead to create the plot in

        Returns
        -------
        fig : plt.figure or None
            The figure that was created
        """

        if ax is None:
            fig = plt.figure(figsize=(7.5, 6))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = None

        self.samples.plot(style='o', ax=ax)
        self.adjustments.plot(ax=ax)

        return fig


def usecase():
    from tests.helper_functions import read_test_data
    from pybreaks.break_test import TsRelBreakTest
    gpi = 654079  # bad: 395790,402962

    ts_full, breaktime = read_test_data(gpi)
    canname, refname = 'CCI', 'REF'
    ts_full = ts_full[[canname, refname]]
    ts_full['original'] = ts_full[canname].copy(True)

    adjust_group = 0

    breaktime = datetime(2012,7,1)
    timeframe = np.array([datetime(2010,7,1), datetime(2018,6,30)])


    ts_frame = ts_full[datetime(2002, 6, 19):timeframe[1]].copy(True)

    testds = TsRelBreakTest(ts_frame[canname], ts_frame[refname],
                            breaktime=breaktime,
                            bias_corr_method='linreg')

    isbreak, breaktype, testresult, errorcode = testds.run_tests()
    print(isbreak, breaktype)

    obj = RegressPairFit(
        ts_frame[canname],
        ts_frame[refname],
        breaktime,
        candidate_freq='D',
        regress_resample=None,  # ('M', 0.3),
        bias_corr_method='linreg',
        filter=('both', 5),
        adjust_group=adjust_group,
        model_intercept=True)

    obj.plot_models()

    values_to_adjust = ts_full[canname].loc[:breaktime]
    can_adjusted = obj.adjust(
        values_to_adjust, corrections_from_core=True, resample_corrections=True,
        interpolation_method='linear', values_to_adjust_freq='D')
    obj.plot_adjustments()

    can_unchanged = obj.get_group_data(obj.other_group, obj.df_original, (obj.candidate_col_name))
    can_new = pd.concat([can_adjusted, can_unchanged], axis=0)

    ref_new = obj.get_group_data(None, obj.df_original, (obj.reference_col_name))

    nobj = RegressPairFit(can_new, ref_new, breaktime,
                          regress_resample=None, bias_corr_method=None,
                          filter=(None, 5), adjust_group=adjust_group, model_intercept=True)
    nobj.plot_models()
    testds = TsRelBreakTest(can_new, ts_frame[refname],
                            breaktime=breaktime,
                            bias_corr_method='linreg')

    isbreak, breaktype, testresult, errorcode = testds.run_tests()

    print(isbreak, breaktype)


if __name__ == '__main__':
    usecase()
