# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
from datetime import datetime
from pybreaks.base import TsRelBreakBase
from pybreaks.break_test import TsRelBreakTest
from pybreaks.adjust_linear_model_pair_fitting import RegressPairFit
from pybreaks.adjust_freq_quantile_matching import QuantileCatMatch, N_Quantile_Exception
from pybreaks.adjust_higher_order_moments import HigherOrderMoments
from pybreaks.temp_coverage import drop_months_data
from pybreaks.utils import merge_dicts
import warnings
from pybreaks.break_adjust_check import BreakAdjustInputCheck
import copy
import numpy as np
import os

'''
Combines the classes for break detection and adjustment. Detects a break
in the passed data and performs the selected adjustment method. Also calls
pre and post adjustment evaluations to test if the correction was successful.
This is the framework for adjustment of breaks in SM time series.

TODO #################################
(++) HOM with flagged values leads to keeping only flagged value after adjustment
    (separate model/prediction input and adjustment input)
(+) HOM cannot handle different values to adjust then used for predictixon 
    (which is fine)
(+) Implement and test the separate adjustment of frame data and rest with the 
    models from the frames

NOTES ################################
- add interval for worsening where its ok? In the check after after correction?
'''

class TsRelBreakAdjust(TsRelBreakBase):
    """
    Framework that combines testing, correction and checking for a single break.
    """
    def __init__(self, candidate, reference, breaktime, adjustment_method,
                 candidate_flags=None, timeframe=None, bias_corr_method='linreg',
                 adjust_tf_only=False, adjust_group=0, input_resolution='D',
                 adjust_check_pearsR_sig=(0.7, 0.01), adjust_check_fix_temp_coverage=False,
                 adjust_check_min_group_range=365, adjust_check_coverdiff_max=None,
                 adjust_check_ppcheck=(True, False), create_model_plots=False,
                 test_kwargs=None, adjmodel_kwargs=None):
        """
        Parameters
        ----------
        candidate : pd.Series
            Series containing the candidate time series (that contains breaks)
        reference : pd.Series or pd.DataFrame
            Series containing the reference time series
        breaktime : datetime.datetime
            Time to test for a break of the passed candidate
        adjustment_method : str or None
            'LMP' or 'QCM' or 'HOM'
            Name of the method used for adjusting the break, module must exist
            and must be implemented in this class.
            If None is passed, only testing is performed.
        candidate_flags : (pd.Series, list), optional (default: None)
            Tuple of series that contains flags for values that should be used for
            testing and model generation and a list of flags that are used.
        timeframe : (datetime.datetime, datetime.datetime), optional (default: None)
            If a time frame is passed, testing and model generation is done for
            data in the timeframe, and adjustment offers the possibility to
            either adjust only data before/after the break within the timeframe
            or all valus before/after the break time.
        bias_corr_method : str or None, optional (default: 'linreg')
            Method for bias correction of reference data, as defined in pytesmo
            that is used for scaling. If None is passed, no scaling is performed.
        adjust_tf_only : bool, optional (default: True)
            If this is True, only values for the selected group within the time
            frame are adjusted, else, ALL input values, before or after the break
            time (depending on the selected adjust_group) are adjusted.
        adjust_group : int, optional (default: 0)
            What part of the candidate should be adjusted
            (0 = before breaktime, 1 = after breaktime)
        input_resolution : str, optional (default: 'D')
            Whether the input data is daily or monthly
        adjust_check_pearsR_sig : (float, float), optional (default: (0.7, 0.05))
            Minimum r and maximum p for pearson correlation between candidate and
            reference OF EACH GROUP, to attempt ADJUSTMENT.
            At low correlation the linear model is not representative of the data.
        adjust_check_fix_temp_coverage : bool, optional (default: False)
            NOT IMPLEMENTED
            If set True, in cases where the temporal coverage is different,
            months are dropped.
        adjust_check_min_group_range : int, optional (default: 365)
            Minimum covered days by either adjustment group (range between first
            and last day, not individual observations). This is always in Days,
            also if the input data is e.g. monthly.
        adjust_check_ppcheck : dict or (bool, bool), optional
                (default: {'bias_diff': True, 'varratio_diff':False} = (True, False)
            Activate, deactivate implemented post-adjustment checks, that compare
            the adjusted candidate against the reference series.
            e.g. by default check if differences in bias got smaller and
            differences in variance ratio got smaller for the candidate across
            the breaktime wrt. to reference series.
        adjust_check_coverdiff_max : float, optional (default: None)
            The maximum coverage difference that a month may have (before vs. after)
            the break. e.g. 0.5 would lead to no adjustment if any month before
            the breaktime contains less than half the observations of the same
            month after tbe break (relative to all days possible)
        create_model_plots : bool, optional (default: False)
            Activate the creation of a plot collection of the models for the
            selected adjustment method (slow).
        ------------------------------------------------------------------------
        test_kwargs : dict, optional (default: None)
            Parameters that are used to create the test object.
            If None is passed the default ones are used.
            --------------------------------------------------------------------
            test_resample : (str, float), optional (default : ('M', 0.3))
                Time period and minimum data coverage in this period for resampling
                before testing, e.g. >30% values in a month to create one monthly
                average.
            mean_test : str, optional (default: 'wilkoxon')
                Name of the test for detecting breaks in the time series means,
                supported by TsRelBreakTest module
            var_test : str, optional (default: 'scipy_fligner_killeen')
                Name of the test for detecting breaks in the time series variances,
                supported by TsRelBreakTest module
            alpha : float, optional (default: 0.01)
                Minimum significance level to detect a break, valid for all tests.
            test_check_min_data : int, optional (default: 5)
                Minimum number of values before / after breaktime to attempt testing
            test_check_spearR_sig : (float, float), optional (default: (0, 0.01))
                Minimum r and maximum p for Spearman correlation between candidate
                and reference to attempt testing
        ------------------------------------------------------------------------
        adjmodel_kwargs : dict, optional (default: None)
            Parameters that are used to create the adjustment object.
            If None are passed the default ones are used.
            Parameters that should be passed here, depend on the selected Method
            that was chosen in adjust_method.
            --------------------------------------------------------------------
            LMP: Parameters are used to create the LMP object
                regress_resample : tuple, optional (default: None)
                     Time period and minimum data coverage in this period for resampling
                     candidate and reference before generating regression models from
                     the resampled data. eg ('M', 0.3)
                filter : tuple or None, optional (default: ('both', 5))
                    (parts to filter , percentile)
                        parts to filter : which part of the input data (wrt to
                        the break time) is being filtered
                            ('first', 'last', None, or 'both')
                    percent threshold : 0 = No filtering, 100 = all removed
                    (based on difference to reference)
                model_intercept : bool, optional (default: True)
                    Create a 2 parameter linear model, if this is False, only
                    the slope will be modeled (intercept is 0) and matched.
            #------------------------------------------------------------------
            QCM : Parameters are used to create the QCM object
                n_quantiles : int, optional (default: 12)
                    Number of percentiles that are fitted (equal distribution)
                    (0 and 100 are always considered) must be >=1.
                fist_last : str, optional (default: 'extrapolate')
                    'formula', 'equal' or 'extrapolate'
                    Select 'formula' to calc the boundary values after the formula
                    or 'equal' to use the same value as for the quantile category
                    they are in. By default, the extrapolation of scipy is used.
                fit : str, optional (default: 'mean')
                    Select mean or median to fit the QC means or medians.
            #------------------------------------------------------------------
            HOM : Parameters used to create the Higher Order Moments model
                regress_resample : tuple, optional (default: None)
                     Time period and minimum data coverage in this period for resampling
                     candidate and reference before generating regression models from
                     the resampled data. eg ('M', 0.3)
                filter : tuple, optional (default: None)
                    (parts to filter, percent)
                    parts to filter : which part of the input data (wrt to break time)
                    is being filtered: ('first', 'last', None, or 'both')
                    percent threshold : the amount of values that will be dropped
                    (based on difference to reference)
                force_model_order : int, optional (default: None)
        """
        if adjust_check_fix_temp_coverage:
            raise NotImplementedError('Matching the temporal coverage is not '
                                      'properly implemented')

        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method, dropna=False)

        if candidate_flags is not None:  # if flags are given, use them to filter candidate
            self.good_flags = candidate_flags[1]
            self.df_original['candidate_flags'] = candidate_flags[0]
            self.flagged_col_name = self.candidate_col_name + '_flagged'
            self.df_original[self.flagged_col_name] = \
                self._filter_with_flags(self.df_original, self.candidate_col_name)
        else:
            self.good_flags = None
            self.flagged_col_name = None

        self.input_resolution = input_resolution

        # testing
        if timeframe is None:
            # use the full, available time frame
            self.timeframe = (self.df_original.index[0].date(),
                              self.df_original.index[-1].date())
        else:
            self.timeframe = timeframe

        # cut the reference part to the time frame
        if adjust_group == 0:
            self.df_original = self.df_original[:self.timeframe[1]]
        else:
            self.df_original = self.df_original[self.timeframe[0]:]

        self.test_kwargs = test_kwargs

        # adjustment
        self.adjustment_method = adjustment_method
        self.adjmodel_kwargs = adjmodel_kwargs
        self.adjust_group = adjust_group

        # this
        self.adjust_tf_only = adjust_tf_only
        self.create_model_plots = create_model_plots
        self.original_candidate_col_name = self.candidate_col_name
        self.df_frame = self.df_original.loc[self.timeframe[0]:self.timeframe[1]].copy(True)
        self.force_supress_adjust = True if adjustment_method is None else False

        # checking
        self.adjust_check_fix_temp_coverage = adjust_check_fix_temp_coverage
        self.input_check_kwargs = {'pearsR_sig': adjust_check_pearsR_sig,
                                   'min_group_range': adjust_check_min_group_range,
                                   'coverdiff_max': adjust_check_coverdiff_max}
        self.adjust_check_ppcheck_args = adjust_check_ppcheck
        self.checkstats_adjust = {}
        self.error_code_adjust, self.error_text_adjust = None, None

        # run initial testing
        self.initial_test_obj = self.run_tests()

        self.df_original_store = self.df_original.copy(True)

        if self.isbreak and not self.force_supress_adjust:
            if self.create_model_plots:
                self.plot_collection_figure = plt.figure(figsize=(10, 4))
            else:
                self.plot_collection_figure = None

            # check the input data, if adjustment can be attempted
            adjust_check_stats = self._check_adjust_input_data()
            self.checkstats_adjust.update(adjust_check_stats)
        else:
            self.error_code_adjust = None
            self.plot_collection_figure = None

        # will be calculated later
        self.adjust_obj = None
        self.initial_adjust_obj = None
        self.adjusted_col_name = None  # not yet adjusted

    def _filter_with_flags(self, df, col_name):
        """
        Creates a column candidate_flagged with the values in the passed column
        that were filtered with the self.good_flags in 'candidate_flags'.

        Parameters
        -------
        df : pd.DataFrame
            DataFrame that contains a column 'candidate_flags' and that is filtered.
        col_name : str
            Name of the column that is filtered based in the flags.

        Returns
        -------
        ds : pd.Series
            The filtered, selected column
        """
        return df[col_name].loc[df['candidate_flags'].isin(self.good_flags + [np.nan])]

    def _check_adjust_input_data(self):
        """
        Creates the check object and runs checks depending on all the passed
        arguments on the frame data.

        Returns
        -------
        input_checkstats : dict
            Dictionary of the input check results and stats
        """

        ccn = self.candidate_col_name if self.flagged_col_name is None else self.flagged_col_name
        rcn = self.reference_col_name
        candidate = self.get_group_data(None, self.df_frame, [ccn])[ccn]
        reference = self.get_group_data(None, self.df_frame, [rcn])[rcn]

        group = None

        check = BreakAdjustInputCheck(candidate=candidate,
                                      reference=reference, group=group,
                                      breaktime=self.breaktime, timeframe=self.timeframe,
                                      input_resolution=self.input_resolution,
                                      error_code_pass=[0])

        check.check(**self.input_check_kwargs)
        self.error_code_adjust = check.error_code_adjust
        self.error_text_adjust = check.error_text_adjust

        mdrop = check.mdrop

        if (mdrop is not None) and self.adjust_check_fix_temp_coverage:
            self._fix_temp_coverage(mdrop)
            self.adjust_check_fix_temp_coverage = False  # done that, set to False for retrying
            # # re-check, no inf loop
            self._check_adjust_input_data()

        return check.checkstats

    def _fix_temp_coverage(self, drop_months):
        """
        Drop values in the frame data to fit the coverage of the 2 groups.
        This alters the values in self.df_frame as it removes values there.

        Parameters
        -------
        drop_months : np.array
            List of integers for months that will be removed eg [1,2] for Jan and Feb
        """
        warnings.warn('Trying to fix Temp coverage is not properly tested')

        df0 = self.get_group_data(0, self.df_frame, 'all')
        df1 = self.get_group_data(1, self.df_frame, 'all')

        self.checkstats_adjust['TempCoverFit_nMonthsDropped'] = drop_months.size

        df0 = drop_months_data(df0, drop_months)
        df1 = drop_months_data(df1, drop_months)

        self.df_frame = pd.concat([df0, df1], axis=0)
        self.df_frame = self.df_frame.dropna()

    def _model_plots_add_row(self):
        """
        Fetch the plots for the current adjustment method here and append them
        to the bottom of the plot collection like
        # iter1_plotA -- iter1_plotB -- iter1_plotC -- ...
        # iter2_plotA -- iter2_plotB -- iter2_plotC -- ...
            ...             ...             ...
        """

        # This is the plotting function for the adjustment method that was chosen,
        # has to be named "plot()" for all methods

        if not self.plot_collection_figure:
            return

        if self.adjustment_method == 'LMP':
            ppr = 2  # Number of columns for the specific adjustment method
        elif self.adjustment_method == 'QCM':
            ppr = 2
        elif self.adjustment_method == 'HOM':
            ppr = 2
        else:
            raise Exception('Method for break adjustment not implemented in '
                            '_model_plots_add_row()')

        plot_row_counter = int(math.ceil(len(self.plot_collection_figure.axes) /
                                         float(ppr)))

        if plot_row_counter == 0:
            axs = [self.plot_collection_figure.add_subplot(1, ppr, i) for i in range(1, ppr + 1)]
        else:
            self.plot_collection_figure.set_size_inches(10, 4 * (plot_row_counter + 1))
            numrows = 0
            for j in range(plot_row_counter):
                numrows = plot_row_counter + 1

                raise NotImplementedError('matplotlib.axes._subplots.change_geometry is non supported anymore')

                self.plot_collection_figure.axes[ppr * j]. \
                    change_geometry(numrows, ppr, ppr * j + 1)
                self.plot_collection_figure.axes[ppr * j + 1]. \
                    change_geometry(numrows, ppr, ppr * j + ppr)
            axs = []
            for k in range(1, ppr + 1):
                ax = self.plot_collection_figure.add_subplot(numrows,
                                                             ppr,
                                                             ppr * plot_row_counter + k)
                axs.append(ax)

        supress_title = False if plot_row_counter == 0 else True
        _ = self.adjust_obj.plot_models(axs=axs, supress_title=supress_title)

    def _adjust_obj(self, perform_bias_corr=True, **model_kwargs):
        """
        Initializes the objects for getting correction values

        Parameters
        -------
        perform_bias_corr : bool
            Perform bias correction on the adjustment data (again).
        model_kwargs : dict
            Keyword arguments, that are passed to create the method specific
            object. Are different for each of the implemented methods.

        Returns
        -------
        adjust_obj : RegressPairFit or QuantileCatMatch or HigherOrderMoments
            Object for performing adjustment
        """
        if perform_bias_corr:
            bias_corr_method = self.bias_corr_method
        else:
            bias_corr_method = None

        ccn = self.candidate_col_name if self.flagged_col_name is None else self.flagged_col_name
        rcn = self.reference_col_name

        if self.adjustment_method == 'LMP':
            # input data for modelling
            candidate = self.df_frame.dropna()[ccn].copy(True)
            reference = self.df_frame.dropna()[rcn].copy(True)
            candidate.name, reference.name = 'CAN', 'REF'

            self.adjust_obj = \
                RegressPairFit(candidate=candidate,
                               reference=reference,
                               breaktime=self.breaktime,
                               bias_corr_method=bias_corr_method,
                               adjust_group=self.adjust_group,
                               **model_kwargs)  # type: RegressPairFit

        elif self.adjustment_method == 'QCM':
            candidate = self.df_frame.dropna()[ccn].copy(True)
            reference = self.df_frame.dropna()[rcn].copy(True)
            candidate.name, reference.name = 'CAN', 'REF'

            # try with decreasing n_quantiles, until the object could be created
            while model_kwargs['categories'] >= 1:
                try:
                    self.adjust_obj = \
                        QuantileCatMatch(candidate=candidate,
                                         reference=reference,
                                         breaktime=self.breaktime,
                                         bias_corr_method=bias_corr_method,
                                         adjust_group=self.adjust_group,
                                         **model_kwargs)  # type: QuantileCatMatch
                    break  # escape object was created
                except N_Quantile_Exception:
                    # reduce the number of quantile categories if it fails (until 1, where it must work)
                    model_kwargs['categories'] -= 1

        elif self.adjustment_method == 'HOM':
            candidate = self.df_frame.dropna()[ccn].copy(True)
            reference = self.df_frame.dropna()[rcn].copy(True)

            candidate.name, reference.name = 'CAN', 'REF'

            self.adjust_obj = \
                HigherOrderMoments(candidate=candidate,
                                   reference=reference,
                                   breaktime=self.breaktime,
                                   bias_corr_method=bias_corr_method,
                                   adjust_group=self.adjust_group,
                                   **model_kwargs)  # type: HigherOrderMoments
        else:
            raise NotImplementedError('Method {} for break adjustment not implemented '
                                      'in _adjust_obj()'.format(self.adjustment_method))

        # add the linear models to the plot collection
        if self.plot_collection_figure:
            self._model_plots_add_row()

        return self.adjust_obj

    def _discard_adjustment(self):
        """
        Drop the adjusted values and replace them with the
        original data.
        This is done when the adjustment was performed, but not found successful
        in the output tests.
        """
        # restore initial test results
        self.isbreak = True
        self.breaktype = self.initial_test_obj.breaktype

        self.df_original = self.df_original_store.copy(True)
        self.df_frame = self.df_original.loc[self.df_frame.index]

    def get_checkstats(self):
        """
        Merge the error codes from testing and adjustment into 2 dicts.

        Returns
        -------
        checkstats_test : dict
            Stats for the last test run (test run at init)
        checkstats_adjust : dict
            Stats for the last adjustment run (empty if no adjustment attempted)
        """

        try:
            _, error_dict, checkstats_test = self.current_test_obj.get_results()
        except AttributeError:
            _, error_dict, checkstats_test = self.initial_test_obj.get_results()

        checkstats_test['error_code_test'] = error_dict['error_code_test']

        checkstats_adjust = self.checkstats_adjust
        checkstats_adjust['error_code_adjust'] = self.error_code_adjust

        return checkstats_test, checkstats_adjust

    def _check_adjust_output_data(self, bias_diff=True, varratio_diff=True):
        """
        This function calculates statistics for ALL the values that are adjusted
        So not the frame data or the flagged data, but the data that is really being adjusted"
        Returns error in cases that the adjustment should not be accepted.

        Parameters
        -------
        bias_diff : bool, optional (default: True)
            Activate to check if the relative bias in the adjusted values
            across the break decreased  relative to the reference data.
        varratio_diff : bool, optional (default: True)
            Activate to check if the difference in variance ratios between the
            candidate and reference decreased across the break time.
        """

        # check if break was removed
        if self.error_code_test != 0:
            return

        if self.isbreak:
            self.error_code_adjust = 1
            self.error_text_adjust = 'Could not remove break after all iterations'
            return

        cols = [self.adjusted_col_name, self.original_candidate_col_name, self.reference_col_name]

        # check the frame if only the frame is adjusted, else all values
        frame = self.df_frame if self.adjust_tf_only else self.df_original

        # only compare values over the same time period
        frame = frame.dropna()
        stats, vert_met, hor_err = self.get_validation_stats(
            frame, columns=cols, can_name='CAN', ref_name='REF', adj_name='ADJ')

        hor_err = hor_err['group0_group1']
        # BIAS FIT CHECK ###
        can_bias_diff = hor_err['CAN_REF_AbsDiff_bias']
        adj_bias_diff = hor_err['ADJ_REF_AbsDiff_bias']

        self.checkstats_adjust['can_bias_diff'] = can_bias_diff
        self.checkstats_adjust['adj_bias_diff'] = adj_bias_diff

        # VAR RATIO FIT CHECK ###
        can_varratio_diff = hor_err['CAN_REF_AbsDiff_var_Ratio']
        adj_varratio_diff = hor_err['ADJ_REF_AbsDiff_var_Ratio']

        self.checkstats_adjust['can_varratio_diff'] = can_varratio_diff
        self.checkstats_adjust['adj_varratio_diff'] = adj_varratio_diff

        if bias_diff:
            if adj_bias_diff > can_bias_diff:
                self.error_code_adjust = 3
                self.error_text_adjust = 'Increased bias diff in adjusted data'
                return

        if varratio_diff:
            if adj_varratio_diff > can_varratio_diff:
                self.error_code_adjust = 4
                self.error_text_adjust = 'Increased var ratio diff in adjusted data'
                return

    def _adjust(self, correct_below_0=True, **adjfct_kwargs):
        """
        Perform 1x adjustment with the selected method of the class

        Parameters
        -------
        correct_below_0 : bool, optional (default: True)
            If there are values beyond the physical plausible range, these values
            are replaced with the minumum of the original input.
        adjust_separately : bool
            Adjust HSPa and all values before HSPa separately using the same model
            as found for HSPa.

        Returns
        -------
        df_adjust : pd.DataFrame
            The input and output values
        """
        if self.adjusted_col_name is None:
            self.adjusted_col_name = self.candidate_col_name + '_adjusted'

        if self.adjusted_col_name not in self.df_original.columns:
            self.df_original[self.adjusted_col_name] = \
                self.df_original[self.candidate_col_name].copy(True)

        if self.adjust_tf_only:
            df_adjust = self.get_group_data(self.adjust_group, self.df_frame, 'all')
        else:
            df_adjust = self.get_group_data(self.adjust_group, self.df_original, 'all')

        if self.adjustment_method == 'LMP':
            values_to_adjust = df_adjust[self.candidate_col_name].copy(True)
            adjust_obj = self.adjust_obj  # type: RegressPairFit
            adjusted_values = adjust_obj.adjust(values_to_adjust=values_to_adjust,
                                                **adjfct_kwargs)

        elif self.adjustment_method == 'QCM':
            values_to_adjust = df_adjust[self.candidate_col_name].copy(True)
            adjust_obj = self.adjust_obj  # type: QuantileCatMatch
            adjusted_values = adjust_obj.adjust(values_to_adjust=values_to_adjust,
                                                **adjfct_kwargs)

        elif self.adjustment_method == 'HOM':
            values_to_adjust = df_adjust[self.candidate_col_name].copy(True)
            adjust_obj = self.adjust_obj  # type: HigherOrderMoments
            adjusted_values = adjust_obj.adjust(values_to_adjust=values_to_adjust,
                                                **adjfct_kwargs)

        else:
            raise ValueError(self.adjustment_method,
                             'Adjustment method not implemented)')

        if self.adjusted_col_name not in df_adjust.columns:
            df_adjust[self.adjusted_col_name] = df_adjust[self.candidate_col_name]

        df_adjust.loc[adjusted_values.dropna().index, self.adjusted_col_name] = \
            adjusted_values.dropna()

        if correct_below_0:  # values < 0 after adjustment are replaced by the minimum value of the time frame
            try:
                min_val = np.nanmin(values_to_adjust.values)
            except ValueError:
                min_val = np.nanmin(values_to_adjust['CAN'].values)
            df_adjust.loc[df_adjust[self.adjusted_col_name] < 0, self.adjusted_col_name] = min_val

        return df_adjust

    def adjust_checks_pass(self, error_code_pass=[0]):
        """
        Checks if the adjustment raised an error or not.

        Parameters
        -------
        error_code_pass : list, optional (default : [0])
            List of error codes that dont lead to the checks to fail.

        Returns
        -------
        adjust_checks_pass : bool
            Indicates whether all checks passed or not.
        """
        if self.error_code_adjust not in error_code_pass:
            return False
        else:
            return True

    def test_and_adjust(self, min_iter=0, max_iter=5, correct_below_0=True,
                        **adjfct_kwargs):
        """
        Tests the input data or time frame for breaks and adjusts the candidate
        iteratively.

        Parameters
        -------
        min_iter : int, optional (default: 0)
            Minimum number of iterations that are forced when there is a break.
            If this is not 0, adjustment is forced.
        max_iter : int, optional (default: 5)
            Maximum number of iterations, if the break is still there, this is
            shown in the adjustment error messages
        correct_below_0 : bool, optional (default : True)
            If this is set to true, values below 0 after adjustment will be set
            to the minimum value of the current time frame instead - so that there
            are never values below 0.
        adjfct_kwargs :
            kwargs that are specific to the adjustment method and are passed
            to the adjust() function of the object
        """
        if min_iter not in [None, 0, False]:
            warnings.warn('Forcing multiple adjustments may bias the output')
        else:
            min_iter = 0

        if self.adjust_tf_only and self.timeframe is None:
            raise ValueError(self.timeframe, 'No timeframe specfied')

        if not self.isbreak:
            # print('No break detected')
            pass
        else:
            if not self.adjust_checks_pass():
                # print('No adjustment due to: %s' % self.error_text_adjust)
                pass
            else:
                i = 0
                while (self.isbreak and i < max_iter) or (i < min_iter):
                    self.adjust_obj = self._adjust_obj(perform_bias_corr=False,
                                                       **self.adjmodel_kwargs)
                    if i == 0:
                        self.initial_adjust_obj = copy.deepcopy(self.adjust_obj)

                    # print(self.breaktype)

                    # DO THE ADJUSTMENT
                    df_adjust = self._adjust(correct_below_0, **adjfct_kwargs)

                    # Add the new values to df_original in the col for the adjusted data
                    # but only in cases where the break was removed.
                    self.df_original.loc[df_adjust.index, self.adjusted_col_name] = \
                        df_adjust[self.adjusted_col_name]

                    # if using flagged data, update the flagged candidate as the adjusted values
                    if self.flagged_col_name is not None:
                        self.df_original[self.flagged_col_name] = \
                            self._filter_with_flags(self.df_original, self.adjusted_col_name)

                    # replace also the old frame data!!!
                    self.df_frame = self.df_original.loc[self.df_frame.index]
                    # change the name of the candidate col (flagged one was updated and stays)
                    self.candidate_col_name = self.adjusted_col_name

                    # re-run the tests
                    self.last_test_obj = self.run_tests()

                    i += 1

                self._check_adjust_output_data(*self.adjust_check_ppcheck_args)

    def get_adjusted(self):
        """
        Get the adjusted values, clean the dataframe of any failed attempts by
        discarding the wrongly adjusted column.

        Returns
        -------
        df_original : pandas.DataFrame
            The main_verification.py data frame with an addition column for adjusted values
        stillbreak : bool
            Whether there is still a break after the last adjustment iteration
        error_code : int
            Error code (0=success, all other=failed)
        """
        if self.error_code_adjust is not None:
            if not self.adjust_checks_pass():
                self._discard_adjustment()

        return self.df_original, self.isbreak, self.error_code_adjust

    def run_tests(self):
        """
        Take the candidate and reference column of the frame data and create a
        test object, run the tests and return the test results.
        Settings for testing were set when creating the object.

        Returns
        -------
        current_test_obj : TsRelBreakTest
            The object which was used for testing
        """
        if self.flagged_col_name is None:
            ccn = self.candidate_col_name
        else:
            ccn = self.flagged_col_name

        rcn = self.reference_col_name
        df = self.df_frame.loc[:, [ccn, rcn]]

        candidate = df.loc[:, [ccn]].dropna()
        reference = df.loc[:, [rcn]].dropna()

        self.current_test_obj = TsRelBreakTest(candidate=candidate,
                                               reference=reference,
                                               breaktime=self.breaktime,
                                               bias_corr_method=self.bias_corr_method,
                                               **self.test_kwargs)
        self.isbreak, self.breaktype, self.testresult, self.error_code_test = \
            self.current_test_obj.run_tests()
        return self.current_test_obj

    def _ts_props(self):
        """ Specific for each child class """
        def adj_failed(isbreak, adjusted_name, error_code_adjust):
            if error_code_adjust is (None or isbreak) is None or (adjusted_name is None):
                return None
            elif (error_code_adjust in [0]) and (not isbreak) and (adjusted_name is not None):
                return False
            else:
                return True

        props = {'isbreak': self.isbreak,
                 'breaktype': self.breaktype,
                 'candidate_name': self.original_candidate_col_name,
                 'reference_name': self.reference_col_name,
                 'adjusted_name': self.adjusted_col_name,
                 'adjust_failed': adj_failed(self.isbreak, self.adjusted_col_name,
                                             self.error_code_adjust)}

        return props

    def plot_coll_fig(self, save_path=None, gpi=None):
        """
        Write the linear model plot collection to image file.

        Parameters
        ----------
        save_path : str, optional (default: None)
            path and file name of the file that should be written.
        gpi : int, optional (default: None)
            Identifies the current collection for file name only.
            If this is None it's not included in the file name.
        """
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            filename = ('{}model_{}.png'.format(str(gpi)+'_' if gpi else '',
                                               str(self.breaktime.date())))

            self.plot_collection_figure.savefig(os.path.join(save_path, filename),
                                                dpi=200)
            plt.close()
        else:
            return self.plot_collection_figure

    def get_results(self):
        """
        Get the results for testing and modelling (before and after adjustment
        if adjustment was done)

        Returns
        -------
        test_results_b4 : dict
            Test results from initialisation test object.
            These are the test results for the unchanged input data.
        mp_iter0 : dict
            Only if a break was found: Regression models parameters as dictionaries
            These are for the model of the first iteration of adjustment.
        test_results_aft : dict
            Only if adjustment was performed: Test results after adjustment
            These are the test results for the input data after all iterations of
            adjustment.
        mp_last : dict
            Only if adjustment was performed: Regression models parameters for
            the last iteration.
            These are for the model of the last iteration of adjustment.
        group_stats : dict
            Statistics (mean, median, etc.) for the (adjusted) frame per group
            Some of the parameters are also in the test dicts and the check dict,
            so they must match!
        vertical_metrics : dict
            Contains metrics for the 2 groups (before/after break). They are
            compared (eg via diff) for the horizontal errors.
        hor_errors : dict
            Comparison statistics of frame statistics before/after break time
            (mean_diff, var_ratio, etc.).
            These show the change of statistics from group stats over the break
            time. When choosing differences for comparison, the adjusted values
            should be closer to 0, than the unadjusted values. Otherwise the
            check for adjusted values should raise an error if the stat is supported
            by the check after adjustment.
        checkstats : dict
            Contains the error messages for tests of candidate and adjusted data
            and the statistics that were calculated for the checks. If an error
            is raised, the check stops, therefore errors in early checks will lead
            to later checks being ignore, and no stats for later checks are calculated.
        """

        # group (comparison) metrics
        props = self._ts_props()
        cols = [self.original_candidate_col_name, self.reference_col_name]
        adj_name = None

        if props['adjusted_name'] is not None and props['adjust_failed']:
            cols.append(self.adjusted_col_name)
            adj_name = 'ADJ'


        group_stats, vertical_metrics, hor_errors = self.get_validation_stats(
            self.df_frame, columns=cols, can_name='CAN', ref_name='REF',
            adj_name=adj_name, as_dict=True)

        # error codes and check stats
        checkstats_test, checkstats_adjust = self.get_checkstats()
        checkstats = merge_dicts(checkstats_test, checkstats_adjust)

        if self.adjusted_col_name is None:  # before adjustment
            # test stats
            testresults_ifirst = self.initial_test_obj.get_flat_results()

            if self.isbreak:
                if self.force_supress_adjust:
                    return testresults_ifirst, None, None, None, group_stats, \
                           vertical_metrics, hor_errors, checkstats
                else:
                    # models stats from init (unadjusted)
                    return testresults_ifirst, None, None, None, group_stats, \
                           vertical_metrics, hor_errors, checkstats
            else:
                # no break --> no model created
                return testresults_ifirst, None, None, None, group_stats, \
                       vertical_metrics, hor_errors, checkstats
        else:
            # models stats from init (unadjusted)
            models_ifirst = self.initial_adjust_obj.get_model_params()
            # models stats from current (last) adjust object
            models_ilast = self.adjust_obj.get_model_params()
            testresults_ifirst = self.initial_test_obj.get_flat_results()

            # also add the error code for the adjustment to the test results?
            testresults_ilast = self.current_test_obj.get_flat_results()

            return testresults_ifirst, models_ifirst, testresults_ilast, \
                models_ilast, group_stats, vertical_metrics, hor_errors, \
                checkstats

    def plot_adj_stats_ts(self, kind='line', save_path=None, title=None):
        """
        Always plot the data, that is also adjusted.
        Plot time series and stats

        Parameters
        -------
        kind : str
            box or line
        save_path : str, optional
            Path where the image is stored
        title : str
            Title for the plot
        """
        if self.adjust_tf_only:
            frame = self.df_frame
        else:
            frame = self.df_original
        exclude_cols = [self.flagged_col_name, 'candidate_flags']
        frame = frame[[col for col in frame.columns.values if col not in exclude_cols]]

        self.plot_stats_ts(frame, kind=kind, save_path=save_path, title=title)

    def plot_adj_ts(self, save_path=None, title=None, ax=None):
        """
        Always plot the data, that is also adjusted.
        Plot time series only

        Parameters
        -------
        save_path : str, optional
            Path where the image is stored
        title : str
            Title for the plot
        ax : matplotlib.axes.Axes
            Axes that is used for the plot

        Returns
        -------
        ax : matplotlib.axes.Axes
            Plot axes
        """
        if self.adjust_tf_only:
            frame = self.df_frame
        else:
            frame = self.df_original

        exclude_cols = [self.flagged_col_name, 'candidate_flags']
        frame = frame[[col for col in frame.columns.values if col not in exclude_cols]]

        return super(TsRelBreakAdjust, self).plot_ts(frame, title=title,
                                                     only_success=True,
                                                     ax=ax,
                                                     save_path=save_path)

    @staticmethod
    def get_meta_dicts():
        """
        Returns a dictionary of error messages and error codes that my arise during testing

        Returns
        -------
        adjust_meta : OrderedDict
            Dictionary containing all error codes and messages for adjustment
        test_meta : OrderedDict
            Dictionary containing all error codes and messages for testing
        """

        test_meta = TsRelBreakTest.get_test_meta()

        cont = [('0', '(AA) No error occurred'),
                ('1', '(AA) Max. N adjustment iterations reached'),
                ('2', '(BA) Group Pearson R for adjustment failed'),
                ('3', '(AA) Increased bias diff in adjusted data'),
                ('4', '(AA) Increased var ratio diff in adjusted data'),
                ('5', '(BA) Group month. temp. coverage differs too much'),
                ('6', '(BA) Group temp. coverage span too short'),
                ('7', ''),
                ('8', ''),
                ('9', 'Unknown Error')]

        adjust_meta = OrderedDict(cont)

        return adjust_meta, test_meta

def usecase_qcm():
    from io_data.otherfunctions import smart_import

    gpi = 707393  # bad: 395790,402962
    BREAKTIME = datetime(2012, 7, 1)
    TIMEFRAME = [datetime(2010, 1, 15), datetime(2018, 6, 30, 0, 0)]

    canname = 'CCI_44_COMBINED'
    refname = 'MERRA2'

    ts_full, plotpath = smart_import(gpi, canname, refname)
    ts_full['original'] = ts_full[canname].copy(True)
    ts_full = ts_full.rename(columns={canname:'can', refname:'ref'})

    test_kwargs = dict([('test_resample', ('M', 0.3)),
                        ('mean_test', 'wilkoxon'),
                        ('var_test', 'scipy_fligner_killeen'),
                        ('alpha', 0.01),
                        ('test_check_min_data', 5),
                        ('test_check_spearR_sig', [0, 0.01])])

    adjmodel_kwargs = dict([('n_quantiles', 4),
                            ('first_last', 'formula'),
                            ('fit', 'mean')])

    adjfct_kwargs = {'interpolation_method': 'cubic'}


    ts_full = ts_full.loc['2007-01-01':]
    ts_full.loc[:BREAKTIME, 'can'] += 5.

    ds = TsRelBreakAdjust(candidate=ts_full['can'],
                          reference=ts_full['ref'],
                          breaktime=BREAKTIME,
                          adjustment_method='QCM',
                          candidate_flags=(ts_full['flags'], [0]),
                          timeframe=TIMEFRAME,
                          bias_corr_method='cdf_match',
                          adjust_tf_only=False,
                          adjust_group=0,
                          input_resolution='D',
                          adjust_check_pearsR_sig=(0.5, 0.1),
                          adjust_check_fix_temp_coverage=False,
                          adjust_check_min_group_range=365,
                          adjust_check_coverdiff_max=None,
                          adjust_check_ppcheck=(True, False),
                          create_model_plots=True,
                          test_kwargs=test_kwargs,
                          adjmodel_kwargs=adjmodel_kwargs)

    print(ds.isbreak, ds.breaktype)
    print(ds._ts_props())

    # always look at data_adjusted together with ther error message and the final test!
    ds.test_and_adjust(min_iter=None, max_iter=3, correct_below_0=True,
                       **adjfct_kwargs)
    ds.plot_adj_stats_ts()
    ds.plot_adj_ts()
    ds.plot_coll_fig()

    ds.get_results()
    ds.get_checkstats()


    data_adjusted, stillbreak, error = ds.get_adjusted()

    test_results_b4, models_ifirst, test_results_aft, models_ilast, \
    group_stats, vertical_metrics, hor_errors, checkstats = \
        ds.get_results()

    print(stillbreak)
    if ds.adjusted_col_name and not stillbreak and error == 0:
        ts_full.loc[data_adjusted.index, 'candidate'] = \
            data_adjusted[ds.adjusted_col_name]
    else:
        print('Error or still a break')


def usecase_hom():
    from io_data.otherfunctions import smart_import

    gpi = 707393  # bad: 395790,402962
    BREAKTIME = datetime(2012, 7, 1)
    TIMEFRAME = [datetime(2010, 1, 15), datetime(2018, 6, 30, 0, 0)]

    canname = 'CCI_44_COMBINED'
    refname = 'MERRA2'

    ts_full, plotpath = smart_import(gpi, canname, refname)
    ts_full['original'] = ts_full[canname].copy(True)
    ts_full = ts_full.rename(columns={canname:'can', refname:'ref'})

    test_kwargs = dict([('test_resample', ('M', 0.3)),
                        ('mean_test', 'wilkoxon'),
                        ('var_test', 'scipy_fligner_killeen'),
                        ('alpha', 0.01),
                        ('test_check_min_data', 5),
                        ('test_check_spearR_sig', [0, 0.01])])

    adjmodel_kwargs = dict([('regress_resample', ('M', 0.3)),
                            ('filter', ('both', 5)),
                            ('poly_orders', [2,3]),
                            ('cdf_types', None)])

    adjfct_kwargs = {'alpha': 0.6,
                     'use_separate_cdf': False,
                     'from_bins': False}

    ts_full = ts_full.loc['2007-01-01':]
    ts_full.loc[:BREAKTIME, 'can'] += 5.

    ds = TsRelBreakAdjust(candidate=ts_full['can'],
                          reference=ts_full['ref'],
                          breaktime=BREAKTIME,
                          adjustment_method='HOM',
                          candidate_flags=(ts_full['flags'], [0]),
                          timeframe=TIMEFRAME,
                          bias_corr_method='cdf_match',
                          adjust_tf_only=False,
                          adjust_group=0,
                          input_resolution='D',
                          adjust_check_pearsR_sig=(0.5, 0.1),
                          adjust_check_fix_temp_coverage=False,
                          adjust_check_min_group_range=365,
                          adjust_check_coverdiff_max=None,
                          adjust_check_ppcheck=(True, False),
                          create_model_plots=True,
                          test_kwargs=test_kwargs,
                          adjmodel_kwargs=adjmodel_kwargs)

    print(ds.isbreak, ds.breaktype)
    print(ds._ts_props())

    # always look at data_adjusted together with ther error message and the final test!
    ds.test_and_adjust(min_iter=None, max_iter=3, correct_below_0=True,
                       **adjfct_kwargs)
    ds.plot_adj_stats_ts()
    ds.plot_adj_ts()
    ds.plot_coll_fig()

    checkstats_test, checkstats_adjust = ds.get_checkstats()

    data_adjusted, stillbreak, error = ds.get_adjusted()

    test_results_b4, models_ifirst, test_results_aft, models_ilast, \
    group_stats, vertical_metrics, hor_errors, checkstats = \
        ds.get_results()

    print(stillbreak)
    if ds.adjusted_col_name and not stillbreak and error == 0:
        ts_full.loc[data_adjusted.index, 'candidate'] = \
            data_adjusted[ds.adjusted_col_name]
    else:
        print('Error or still a break')


def usecase_lmp():
    from io_data.otherfunctions import smart_import

    gpi = 707393  # bad: 395790,402962
    BREAKTIME = datetime(2012, 7, 1)
    TIMEFRAME = [datetime(2010, 1, 15), datetime(2018, 6, 30, 0, 0)]

    canname = 'CCI_45_COMBINED'
    refname = 'MERRA2'

    ts_full, plotpath = smart_import(gpi, canname, refname)
    ts_full['original'] = ts_full[canname].copy(True)
    ts_full = ts_full.rename(columns={canname:'can', refname:'ref'})

    test_kwargs = dict([('test_resample', ('M', 0.3)),
                        ('mean_test', 'wilkoxon'),
                        ('var_test', 'scipy_fligner_killeen'),
                        ('alpha', 0.01),
                        ('test_check_min_data', 5),
                        ('test_check_spearR_sig', [0, 0.01])])

    adjmodel_kwargs = dict([('regress_resample', ('M', 0.3)),
                            ('model_intercept', True),
                            ('filter', ('both', 5))])
    adjfct_kwargs = dict([('corrections_from_core', True),
                          ('values_to_adjust_freq', 'D'),
                          ('resample_corrections', True),
                          ('interpolation_method', 'linear')])



    ts_full = ts_full.loc['2007-01-01':]
    ts_full.loc[:BREAKTIME, 'can'] += 5.

    ds = TsRelBreakAdjust(candidate=ts_full['can'],
                          reference=ts_full['ref'],
                          breaktime=BREAKTIME,
                          adjustment_method='LMP',
                          candidate_flags=(ts_full['flags'], [0]),
                          timeframe=TIMEFRAME,
                          bias_corr_method='cdf_match',
                          adjust_tf_only=False,
                          adjust_group=0,
                          input_resolution='D',
                          adjust_check_pearsR_sig=(0.5, 0.1),
                          adjust_check_fix_temp_coverage=False,
                          adjust_check_min_group_range=365,
                          adjust_check_coverdiff_max=None,
                          adjust_check_ppcheck=(True, False),
                          create_model_plots=True,
                          test_kwargs=test_kwargs,
                          adjmodel_kwargs=adjmodel_kwargs)

    print(ds.isbreak, ds.breaktype)
    print(ds._ts_props())

    # always look at data_adjusted together with ther error message and the final test!
    ds.test_and_adjust(min_iter=None, max_iter=3, correct_below_0=True,
                       **adjfct_kwargs)
    ds.plot_adj_stats_ts()
    ds.plot_adj_ts()
    ds.plot_coll_fig()

    ds.get_results()
    ds.get_checkstats()


    data_adjusted, stillbreak, error = ds.get_adjusted()

    test_results_b4, models_ifirst, test_results_aft, models_ilast, \
    group_stats, vertical_metrics, hor_errors, checkstats = \
        ds.get_results()

    print(stillbreak)
    if ds.adjusted_col_name and not stillbreak and error == 0:
        ts_full.loc[data_adjusted.index, 'candidate'] = \
            data_adjusted[ds.adjusted_col_name]
    else:
        print('Error or still a break')


if __name__ == '__main__':
    usecase_lmp()
    usecase_qcm()
    usecase_hom()
