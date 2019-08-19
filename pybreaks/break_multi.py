# -*- coding: utf-8 -*-

import numpy as np
from pybreaks.timeframes import TsTimeFrames
import pandas as pd
import matplotlib.pyplot as plt
from pybreaks.base import TsRelBreakBase
import warnings
from datetime import datetime
from pybreaks.break_test import TsRelBreakTest
from pybreaks.break_adjust import TsRelBreakAdjust
import os
from functools import partial

'''
Module for iteratively test and adjust multiple potential break times in a single
time series. Allows adjustment for frames, between breaks and the full TS. 
Allows an initial test only run, to find homogeneous transition dates and extend
adjustment frames across them.

# TODO #################################
(+) Implement an option to retroactively undo corrections in case that for a
    a break is not being removed, as correction might have increased the break.
(+) Some names for the candidate series are not valid, ie when they contain
    'adjusted' or ADJ, fetch these cases.
(+) Activate gpi dependence


# NOTES ################################
# -
'''


class TsRelMultiBreak(TsRelBreakBase):
    """
    Class that tests input data for breaks in multiple break times and adjusts
    detected breaks.
    This contains methods for extended time frames, adjustment of all values
    before break and everything else that takes multiple time frames into account.
    """

    def __init__(self, candidate, reference, breaktimes, adjustment_method,
                 candidate_flags=None,
                 full_period_bias_corr_method='cdf_match',
                 sub_period_bias_corr_method='linreg', base_breaktime=None,
                 HSP_init_breaktest=False, models_from_hsp=True,
                 adjust_within='breaks', input_resolution='D',
                 test_kwargs=None, adjmodel_kwargs=None, adjcheck_kwargs=None,
                 create_model_plots=False, frame_ts_figure=False,
                 frame_tsstats_plots=False):
        """
        Parameters
        ----------
        candidate : pd.Series
            Pandas series containing the candidate time series, if the series
            has a name, it will be used (it must not contain 'adjusted' or 'ADJ')
        reference : pd.Series or pd.DataFrame
            Pandas object containing the reference time series
        breaktimes : np.array of datetimes
            Times to test for a break
        adjustment_method : str or None
            Name of the method used for adjusting candidate (LMP, HOM, QCM)
            if None is selected, we perform testing only.

        candidate_flags : tuple, optional (default:None)
            (flags, good_flags)
            Series that contains flags for values that should be used for
            testing and model generation and a list of flags that are used.
            All values (also the flagged ones will be adjusted based on the
            corrections from the un-flagged values)
        full_period_bias_corr_method : str, optional (default: cdf_match)
            Scaling method (as implemented in pytesmo), to use for scaling the
            full reference to the full candidate series.
        sub_period_bias_corr_method : str, optional (default: linreg)
            Scaling method (as implemented in pytesmo), to use for scaling the
            reference to candidate within each sub-period that is tested/adjusted.

        base_breaktime : int, optional (default: None)
            indicates the base period (which other periods are scaled to).
            0 = last period, 1 = second last, etc.
        HSP_init_breaktest : bool, optional (default: False)
            Perform an test only run before creating homogeneous sub-periods from
            the test results. Mandatory for 'adjust-within = breaks'.
        models_from_hsp : bool, optional (default: False)
            Create the model from the HSP and not necessarily from the same data
            range that is used for adjustment. If this is False, the full data frame
            that is passed to break_adjust will be used for finding the models.
            In this case the values to adjust would be the same as the model input
            values on the side of the break that is adjusted.
        adjust_within : str, optional (default: 'breaks')
            breaks : First find the breaks, than adjust periods between breaks
            full : For each break, adjust until the beginning/end of the series
            frame : For each break, adjust until the next potential break time
        input_resolution : str, optional (default: None)
            Temporal resolution of the input data (D=daily, M=monthly)

        test_kwargs : dict, optional (default: None)
            kwargs that are passed to initialising the testing objects
        adjmodel_kwargs : dict, optional (default: None)
            kwargs that are passed to initialising break quantification objects
        adjcheck_kwargs : dict, optional (default: None)
            kwargs that are used to initialize the class for checking the adjustment

        create_model_plots : bool, optional (default: False)
            Create a collection of plots for models used during adjustment for ALL
            breaktimes that are attempted (slow!)
        frame_ts_figure : bool, optional (default: False)
            Create a collection of time series plots that can be accessed via
            the plot function for ALL breaktimes (slow!)
        frame_tsstats_plots : bool, optional (default: False)
            Create a collection of time series / stats plots that can be acces
            via the plot function for ALL breaktimes.
        """

        self.break_adjust_kwargs = \
            dict([('input_resolution', input_resolution),
                  ('bias_corr_method', sub_period_bias_corr_method),
                  ('adjust_group', None),  # set this later
                  ('adjust_tf_only', None),  # set this later
                  ('create_model_plots', create_model_plots),
                  ('test_kwargs', test_kwargs),
                  ('adjmodel_kwargs', adjmodel_kwargs)])

        if adjcheck_kwargs is not None:
            self.break_adjust_kwargs.update(adjcheck_kwargs)

        # ---------------------------------------------------------------------
        # -------BREAKTIME STORING --------------------------------------------
        # Contain breaktimes that have no break or where break was adjusted
        self.nobreak_original = []
        self.nobreak_adjusted = []
        self.break_adjusted = []
        # ---------------------------------------------------------------------
        # -------RESULTS STORING ----------------------------------------------
        # Contains results for testing and adjusting all the breaktimes and
        # statistics before/after the break time
        data_template = {breaktime: None for breaktime in breaktimes}

        self.models_ifirst, self.models_ilast = data_template.copy(), data_template.copy()
        self.testresults_init, self.testresults_ifirst, self.testresults_ilast = \
            data_template.copy(), data_template.copy(), data_template.copy()
        self.group_stats, self.vertical_metrics, self.hor_errors = \
            data_template.copy(), data_template.copy(), data_template.copy()

        self.checkstats = data_template
        # ---------------------------------------------------------------------
        # -------FIGURE STORING -----------------------------------------------
        self.frame_ts_figure = frame_ts_figure
        self.model_figures = create_model_plots
        self.frame_tsstats_plots = frame_tsstats_plots
        if self.frame_ts_figure:
            # single figure with frame ts for each break time
            self.frame_ts_figure = plt.figure(figsize=(25, 15))
        else:
            self.frame_ts_figure = None
        if self.model_figures:
            self.model_figures = {}
        else:
            self.model_figures = None
        if self.frame_tsstats_plots:
            self.tsstats_figures = {}
        else:
            self.tsstats_figures = None
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------

        self.base_breaktime = base_breaktime
        self.adjust_within = adjust_within
        self.models_from_hsp = models_from_hsp
        self.adjustment_method = adjustment_method

        TsRelBreakBase.__init__(
            self, candidate, reference, None,
            bias_corr_method=full_period_bias_corr_method,  # scales the FULL series
            dropna=False)

        if candidate_flags is not None:  # if flags are given, use them to filter candidate
            self.flagged_col_name = self.candidate_col_name + '_flagged'
            self.candidate_flags = candidate_flags[0]
            self.good_flags = candidate_flags[1]
            self.df_original[self.flagged_col_name] = \
                candidate.loc[self.candidate_flags.loc[
                    self.candidate_flags.isin(self.good_flags)].index]
        else:
            self.flagged_col_name = None

        # must not contain _adjusted, or other classes will be confused!!
        self.adjusted_col_name = self.candidate_col_name + '_ADJ'
        breaktimes = sorted(breaktimes, reverse=True)

        # Turn break times into strings if necessary
        if not all(isinstance(breaktime, str) for breaktime in breaktimes):
            breaktimes = [str(time.date()) for time in breaktimes]

        # to skip break times, this is done before, and the corrected ones are passed to the class!!
        self.otimes = TsTimeFrames(start=candidate.index[0], end=candidate.index[-1],
                                   breaktimes=breaktimes, min_set_days=None,
                                   skip_breaktimes=None, ignore_position=True,
                                   grid=None, base_breaktime=base_breaktime)

        self.adjust_meta, self.test_meta = None, None

        # this is the same in the default case and overridden by the build_adjustframes method, if chosen
        self.HSP_init_breaktest = HSP_init_breaktest

        if HSP_init_breaktest:
            self.init_test_results = self.build_frames_from_init_testing()
        else:
            self.init_test_results = None
            self.otestable = self.otimes

        if self.adjust_within == 'breaks':
            if HSP_init_breaktest is False:
                raise ValueError('Need to select init_testing=True when adjusting withing breaks')

    def _init_check_testing_possible(self):
        """
        Run tests for all break times and find break times, where testing works
        without an error.

        Returns
        -------
        init_test_not_testable : list
            Break times where testing was initially not possible
        init_test_testable : list
            Break times where testing was initially possible
        init_test_breaks : list
            Break times where a break was initially detected
        init_test_nobreaks : list
            Break times that initially where found to be homogeneous
        init_testresults : dict
            Initial testresults combined in a dictionary
        """
        # deactivate the plot generation for the initial run, activate afterwards
        if self.tsstats_figures is not None:
            tsstats_store = self.tsstats_figures
            self.tsstats_figures = None
        else:
            tsstats_store = None

        if self.frame_ts_figure is not None:
            frame_ts_figure_store = self.frame_ts_figure
            self.frame_ts_figure = None
        else:
            frame_ts_figure_store = None

        times = self.otimes.get_times(as_datetime=True)

        init_test_testable = []
        init_test_not_testable = []
        init_test_breaks = []
        init_test_nobreaks = []

        init_testresults = {breaktime: None for breaktime in times['breaktimes']}

        for i, (breaktime, timeframe) in enumerate(zip(times['breaktimes'],
                                                       times['timeframes'])):

            isbreak, breaktype, testresults = self.test_timeframe(breaktime)

            if testresults['error_code_test'] != 0:
                init_test_not_testable.append(breaktime)
            else:
                init_test_testable.append(breaktime)

            if isbreak == False: # None would be not testable
                init_test_nobreaks.append(breaktime)
            elif isbreak:
                init_test_breaks.append(breaktime)

            init_testresults[breaktime] = testresults

        # re-set what was removed before
        self.tsstats_figures = tsstats_store
        self.frame_ts_figure = frame_ts_figure_store

        return init_test_not_testable, init_test_testable, init_test_breaks, \
               init_test_nobreaks, init_testresults

    def _timeframe_data(self, time, dataframe):
        """
        Take the input candidate and reference and returns only data for the
        time frame.

        Parameters
        -------
        time : (datetime.datetime, datetime.datetime) or datetime.datetime
            Start and end date of the subset that is extracted from the candidate
            and reference, or a single break time, then the according time frame
            is searched and used.
        dataframe : pd.DataFrame
            The dataframe from which the subset is taken

        Returns
        -------
        df_frame : pandas.DataFrame
            Data for the selected time frame, cut from self.df_original
        """
        if isinstance(time, datetime):
            timeframe = self.otimes.timeframes_from_breaktimes(time)
        else:
            timeframe = time  # type: tuple

        return dataframe.loc[timeframe[0]:timeframe[1]].copy(True)

    def test_timeframe(self, breaktime):
        """
        Tests a single time frame for a break, no adjustment is performed.
        The time frame is searched according to the passed break time.

        Parameters
        ----------
        breaktime : datetime.datetime
            Time that should be tested for an inhomogeneity

        Returns
        -------
        isbreak : bool
            True if any break is found, False otherwise
        breaktype : str
            'mean' or 'var' or 'both', which kind of break was found
        flatresults : dict
            The flat test statistics for the selected tests
        """
        timeframe = self.otimes.timeframe_for_breaktime(None, breaktime)

        df_frame = self._timeframe_data(timeframe, self.df_original)

        if 'test_kwargs' not in self.break_adjust_kwargs:
            test_kwargs = {}
        else:
            test_kwargs = self.break_adjust_kwargs['test_kwargs']

        if self.flagged_col_name is None:
            candidate = df_frame[self.candidate_col_name]
        else:
            candidate = df_frame[self.flagged_col_name]

        reference = df_frame[self.reference_col_name]
        test_obj = TsRelBreakTest(candidate=candidate,
                                  reference=reference,
                                  bias_corr_method=self.break_adjust_kwargs['bias_corr_method'],
                                  breaktime=breaktime,
                                  **test_kwargs)

        if not self.test_meta: # this is not changing and can be re-used
            self.test_meta = test_obj.get_test_meta()

        isbreak, breaktype, _, error_code_test = test_obj.run_tests()
        flatresults = test_obj.get_flat_results()

        if self.tsstats_figures is not None:
            title = str(breaktime.date()) + '_ts_stats'
            self.tsstats_figures[breaktime] = \
                test_obj.plot_stats_ts(frame=df_frame, save_path=None,
                                       kind='line', title=title)

        return isbreak, breaktype, flatresults

    def _extend_unadjusted_part_adjustframe(self, timeframe):
        """
        Extend the passed time frame across untestable parts in the initial test
        object. So that data at untestable break times is adjusted together with
        later break times.

        Parameters
        ------
        timeframe : np.array
            the current time frame that is being extended

        Returns
        -------
        adjustframe : tuple
            The new, extended time frame that is used for adjustment, not for
            modelling
        """

        # todo: check, is this working?
        adjust_group = self.break_adjust_kwargs['adjust_group']

        adjustframe = list(timeframe)

        if self.init_test_testable is None:
            raise ValueError('Must run build_adjustment_frames before using this method')

        times = self.otestable.get_times(as_datetime=True)
        testable_times = times['breaktimes'].tolist()
        ranges = times['ranges'].tolist()

        if adjust_group == 0:
            # expand data that will be adjusted over break times that are not
            # testable
            while adjustframe[0] not in (testable_times + ranges):
                adjustframe[0] = self.otimes.get_adjacent(adjustframe[0], 1, None)
        elif adjust_group == 1:
            while adjustframe[1] not in (testable_times + ranges):
                adjustframe[1] = self.otestable.get_adjacent(adjustframe[1], -1, None)
        else:
            raise ValueError(adjust_group,
                             'Adjustment Group for extending the time frame unknown')

        return adjustframe

    def _extend_adjusted_part_timeframe(self, timeframe):
        """
        Extend the passed time frame across homogeneous break times of this object
        that have already been adjusted

        Parameters
        ------
        timeframe : np.array
            the current time frame that is being extended

        Returns
        -------
        timeframe : tuple
            The new, extended time frame for adjustment
        """

        adjust_group = self.break_adjust_kwargs['adjust_group']

        timeframe = list(timeframe)

        if adjust_group == 0:
            while timeframe[1] in (self.nobreak_adjusted + self.nobreak_original):
                timeframe[1] = self.otimes.get_adjacent(timeframe[1], -1, None)
                # print('expand tf to %s' % str(timeframe[1].date()))
        elif adjust_group == 1:
            while timeframe[0] in (self.nobreak_adjusted + self.nobreak_original):
                timeframe[0] = self.otimes.get_adjacent(timeframe[0], 1, None)
                # print('expand tf to %s' % str(timeframe[0].date()))
        else:
            raise ValueError(adjust_group,
                             'Adjustment Group for extending the time frame unknown')

        return timeframe

    def plot_adjustment_ts_full(self, save_path=None, prefix=None, resample='D',
                                ax=None, legend=True):
        """
        Create a plot of candidate, adjusted candidate and reference time series
        for the whole gpi time series.
        Calculate time series stats for comparison and save the plot

        Parameters
        -------
        save_path : str, optional (default: None)
            Path where the image is stored
        prefix: str, optional (default: None)
            Prefix for the name of the plot file to create.
        resample : str, optional (default: 'D')
            Resamples the time series to the given temporal resolution, does
            not influence the statistics, only visual
        ax : plt.axes, optional (default: None)
            Axes object, if this is passed the plot is a new subplot for
            the passed ax.
        legend : bool, optional (default: True)
            Set True to plot the legend for the current plot
        """

        if save_path and not os.path.isdir(save_path):
            os.mkdir(save_path)

        times = self.otimes.get_times(as_datetime=True)
        breaktimes = times['breaktimes']

        if ax is None:
            fig_full_ts, ax = plt.subplots(figsize=(15, 5))
        else:
            fig_full_ts = None

        plot_df = pd.concat([self.df_original[self.candidate_col_name].rename('candidate'),
                             self.df_original[self.adjusted_col_name].rename('adjusted'),
                             self.df_original[self.reference_col_name].rename('reference')], axis=1)

        if resample:
            plot_df = plot_df.resample(resample).mean()

        not_adjusted_values = plot_df.loc[plot_df['adjusted'] ==
                                          plot_df['candidate']].index
        plot_df.loc[not_adjusted_values, 'adjusted'] = np.nan

        styles = ['r-', 'b-', 'k--']

        for col, style in zip(plot_df.columns, styles):
            d = plot_df[col].resample(resample).fillna(np.nan)
            if not d.empty:
                d.plot(ax=ax, style=style)

        if legend:
            ax.legend(ncol=3, shadow=True, title=None, fancybox=True)
        ax.set_title('Detected/Adjusted Breaks and Observations', fontsize=15)
        ax.set_xlabel('Time [year]', fontdict={'size': 12})
        ax.set_ylabel('SM [%] ', fontdict={'size': 12})

        for breaktime in breaktimes:
            if breaktime in self.nobreak_adjusted:
                color = 'green'
                linestyle = 'solid'
            elif breaktime in self.nobreak_original:
                color = 'green'
                linestyle = 'dashed'
            elif breaktime in self.break_adjusted:
                color = 'red'
                linestyle = 'solid'
            else:
                color = 'grey'
                linestyle = 'dashed'

            ax.axvline(breaktime, linestyle=linestyle, color=color, lw=2)

        plt.tight_layout()

        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            gpi_str = str(prefix) + '_' if prefix else ''
            filename = '%sTS_adjustment_full_%s.png' % (gpi_str, resample)

            fig_full_ts.savefig(os.path.join(save_path, filename))
            plt.close()
        else:
            if fig_full_ts is None:
                return ax
            else:
                return fig_full_ts

    def plot_frame_ts_figure(self, save_path=None, prefix=None):
        """
        Show the frame figure plot if it was created

        Parameters
        -------
        save_path : str, optional (default: None)
            Path where the image is created.
        prefix : str, optional (default: None)
            Prefix for the name of the plot file to create.
        """
        if save_path and not os.path.isdir(save_path):
            os.mkdir(save_path)

        if self.frame_ts_figure is None:
            warnings.warn('Must enable time series plots to create the figure')
            return

        self.frame_ts_figure.tight_layout()
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            gpi_str = str(prefix) + '_' if prefix else ''
            filename = '%sTS_adjustment_frames.png' % gpi_str
            self.frame_ts_figure.savefig(os.path.join(save_path, filename))
            plt.close(self.frame_ts_figure)
        else:
            self.frame_ts_figure.show()

    def plot_models_figures(self, save_path=None, prefix=None):
        """
        Saves the linear model plots collections to file or shows them

        Parameters
        ----------
        save_path : str
            Path where the image is created.
        prefix : str, optional (default: None)
            Prefix for the name of the plot file to create.

        Returns
        ----------
        model_figures : None or dict
            If no path is given, the dictionary with the figures is returned.
        """

        if save_path and not os.path.isdir(save_path):
            os.mkdir(save_path)

        if self.model_figures is None:
            warnings.warn('Must enable linear model plots generation to create the figures')
            return
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for breaktime, figure in self.model_figures.items():
                gpi_st = str(prefix) + '_' if prefix else ''
                filename = '%s%s_models.png' % (gpi_st, str(breaktime.date()))
                plt.tight_layout()
                figure.savefig(os.path.join(save_path, filename))
        else:
            return self.model_figures

    def plot_tsstats_figures(self, save_path=None, prefix=None):
        """
        Saves the time series stats plots to file or shows them

        Parameters
        ----------
        save_path : str, optional (default: None)
            Path where the image is created.
        prefix : str, optional (default: None)
            Prefix for the name of the plot file to create.
        """
        if save_path and not os.path.isdir(save_path):
            os.mkdir(save_path)

        if self.tsstats_figures is None:
            warnings.warn('Must enable linear model plots generation to create the figures')
            return
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for breaktime, figure in self.tsstats_figures.items():
                gpi_str = str(prefix) + '_' if prefix else ''
                filename = '%s%s_tsstats_frames.png' % (gpi_str, str(breaktime.date()))
                figure.savefig(os.path.join(save_path, filename))
                plt.close(figure)
        else:
            for breaktime, figure in self.tsstats_figures.items():
                figure.show()

    def _gen_adjust_frame(self, breaktime, extended_reference, adjust_group):
        """
        Create the input dataframe for the current break time and time frame
        for the adjustment object. The input df may be different to the model frame
        data in cases where more than the frame data is used for adjustment.
        i.e where the model is used to adjust more values then were originally
        used to create the model.
        The original time frame (that indicates the range for modelling and testing
        is unchanged!)

        Parameters
        ----------
        breaktime : datetime
            Current break time that is being tested and adjusted
        extended_reference : bool
            If True, the time frame from the break time is extended across
            homogeneous break times
        adjust_group : int
            Group that is being adjusted (0 is before break, 1 is after break)

        Returns
        -------
        input_df : pd.DataFrame
            A subset of self.df_original that is cut to fit the range of
            values that should be used for adjustment and testing.
        """
        self.break_adjust_kwargs['adjust_group'] = adjust_group
        adjust_within = self.adjust_within

        timeframe = self.otimes.timeframe_for_breaktime(gpi=None,
                                                        breaktime=breaktime)

        if extended_reference:
            timeframe = self._extend_adjusted_part_timeframe(timeframe)

        if adjust_within == 'frames':
            adjust_tf_only = True
            adjustframe = timeframe
        elif adjust_within == 'full':  # todo: full take the FULL series not to next break
            adjust_tf_only = False

            if adjust_group == 0:
                adjustframe = (None, timeframe[1])
            else:
                adjustframe = (timeframe[0], None)
        elif adjust_within == 'breaks':
            # extend the adjustment frame only
            adjust_tf_only = False
            adjustframe = self._extend_unadjusted_part_adjustframe(timeframe)
        else:
            raise ValueError(adjust_within, 'Unknown input for period to adjust (enter: full, breaks or frame)')

        self.break_adjust_kwargs['adjust_tf_only'] = adjust_tf_only
        # cut the full frame to the subset that should be adjusted
        adjust_df = self._timeframe_data(adjustframe, self.df_original)

        return adjust_df, timeframe

    def _adjust_break(self, input_df, timeframe, breaktime,
                      **adjfct_kwargs):
        """
        Test for a break in the time series time frame (from the passed break time).
        Attempt to adjust a detected break and return the adjusted time series.

        Parameters
        ----------
        input_df : pd.DataFrame
            Subset of self.df_original, dataframe for testing and adjustment
        timeframe : list or None
            Start and end of the frame data used for testing and modelling
        breaktime : datetime
            Time of the possible break, for testing and time frame creation
        max_retries : int, optional (default: 5)
            Retries the adjustment multiple times, in case that the break was not
            removed.

        Returns
        -------
        adjust_obj : TsRelBreakAdjust
            Current adjustment object

        """
        candidate = input_df[self.adjusted_col_name].copy(True)
        candidate.name = None
        reference = input_df[self.reference_col_name].copy(True)
        reference.name = None

        if self.flagged_col_name:
            flags = (self.candidate_flags, self.good_flags)
        else:
            flags = None

        adjust_obj = TsRelBreakAdjust(candidate=candidate,
                                      reference=reference,
                                      breaktime=breaktime,
                                      timeframe=timeframe,
                                      candidate_flags=flags,
                                      adjustment_method=self.adjustment_method,
                                      **self.break_adjust_kwargs)

        adjust_obj.test_and_adjust(min_iter=None, correct_below_0=True,
                                   **adjfct_kwargs)

        if not self.adjust_meta or not self.test_meta:
            self.adjust_meta, self.test_meta = adjust_obj.get_meta_dicts()

        return adjust_obj

    def _add_breakadjust_results(self, adjust_obj, breaktime):
        """
        Add the results from the passed adjust object to the store of this class
        for the passed break time.

        Parameters
        ----------
        adjust_obj : TsRelBreakAdjust
            The adjustment object
        breaktime : datetime
            Break time for which the data is stored
        """

        self.testresults_ifirst[breaktime], self.models_ifirst[breaktime], \
            self.testresults_ilast[breaktime], self.models_ilast[breaktime], \
            self.group_stats[breaktime], self.vertical_metrics[breaktime], \
            self.hor_errors[breaktime], self.checkstats[breaktime] = adjust_obj.get_results()

    def _add_plots(self, adjust_obj, i):
        """
        Add plots to the collections, if there are collections

        Parameters
        ----------
        adjust_obj : TsRelBreakAdjust
            The adjustment object for the current iteration
        i : int
            Counter for the break times that were already processed
        """

        breaktime = adjust_obj.breaktime

        if self.frame_ts_figure is not None:
            # plot the time series that is adjusted
            ax = self.frame_ts_figure.add_subplot(3, 3, i + 1)
            adjust_obj.plot_adj_ts(ax=ax, title=str(breaktime.date()))

        if self.model_figures is not None:
            if adjust_obj.plot_collection_figure is not None:
                self.model_figures[breaktime] = adjust_obj.plot_coll_fig()

    def _replace_adjusted(self, new_values):
        """ Replace adjusted data in main data frame """
        self.df_original.loc[new_values.index, self.adjusted_col_name] = new_values

    def adjust_all(self, extended_reference=True, **adjfct_kwargs):
        """
        Iterate over all break times and test and adjust the candidate using
        the passed parameters.

        Parameters
        ----------
        extended_reference : bool, optional (default: True)
            If True, the time frame from the break time is extended across
            homogeneous break times

        Returns
        -------
        candidate_adjusted : pd.Series
            The multi break adjusted candidate series

        """
        # copy to the adjusted col, from now on the adjusted is changed
        if self.adjustment_method is None:
            raise ValueError('A method for adjustment must be selected.'
                             'test_all() can be used to test only.')

        self.df_original[self.adjusted_col_name] = self.df_original[self.candidate_col_name].copy()

        times = self.otimes.get_times(as_datetime=True)

        testable_breaktimes = self.otestable.get_times(as_datetime=True)['breaktimes']

        for i, (breaktime, timeframe) in enumerate(zip(times['breaktimes'],
                                                       times['timeframes'])):
            # print breaktime
            if breaktime not in testable_breaktimes:
                continue

            # If break time is AFTER base break time, adjust second part of the
            # time series, else the first part
            if self.base_breaktime:
                if breaktime < datetime.strptime(self.otimes.breaktimes[self.base_breaktime],
                                                 '%Y-%m-%d'):
                    adjust_group = 0
                else:
                    adjust_group = 1
            else:
                # by default, base break time is last break time,
                # so always adjust first part of the sub time series.
                adjust_group = 0

            adjust_df, timeframe = self._gen_adjust_frame(breaktime=breaktime,
                                                          extended_reference=extended_reference,
                                                          adjust_group=adjust_group)
            if not self.models_from_hsp:
                timeframe = None

            adjust_obj = self._adjust_break(input_df=adjust_df, timeframe=timeframe,
                                            breaktime=breaktime, **adjfct_kwargs)

            # write the results for the breaktime to the respective collections
            self._add_breakadjust_results(adjust_obj, breaktime)

            if self.tsstats_figures is not None:
                # plot the time series that is used for models
                self.tsstats_figures[breaktime] = \
                    adjust_obj.plot_adj_stats_ts(title=str(breaktime.date()), kind='line')

            # check the output, discard changes if necessary
            df_adjusted, isbreak, error = adjust_obj.get_adjusted()

            self._add_plots(adjust_obj, i)

            if adjust_obj.adjusted_col_name is None:  # no adjustment was done
                df_adjusted = None

            if isbreak:  # Break was not removed or break type changed
                self.break_adjusted.append(breaktime)
                continue
            elif isbreak is None:  # Testing or adjustment was not possible
                continue
            elif isbreak == False:  # No break or break was removed
                if df_adjusted is None:  # no break was detected
                    self.nobreak_original.append(breaktime)
                else:
                    acl = adjust_obj.adjusted_col_name
                    if acl in df_adjusted:
                        # Break was adjusted, tests succeeded, store data
                        self.nobreak_adjusted.append(breaktime)
                        self._replace_adjusted(df_adjusted[acl])
                    else:
                        # output check failed #todo: this is never reached?
                        # self.break_adjusted.append(breaktime)
                        continue

        return self.df_original[self.adjusted_col_name]

    def test_all(self):
        """
        Only run tests for detection of breaks, no adjustment, on all break times
        """

        times = self.otimes.get_times(as_datetime=True)

        for i, (breaktime, timeframe) in enumerate(zip(times['breaktimes'],
                                                       times['timeframes'])):

            # this adjustment object is only for testing
            candidate = self.df_original[self.candidate_col_name].copy(True)
            candidate.name = None
            reference = self.df_original[self.reference_col_name].copy(True)
            reference.name = None

            adjust_obj = TsRelBreakAdjust(candidate=candidate,
                                          reference=reference,
                                          breaktime=breaktime,
                                          timeframe=timeframe,
                                          adjustment_method=None,
                                          **self.break_adjust_kwargs)

            self.testresults_ifirst[breaktime], _, _, _, self.group_stats[breaktime], \
            self.vertical_metrics[breaktime], self.hor_errors[breaktime], \
            self.checkstats[breaktime] = adjust_obj.get_results()

            if not adjust_obj.isbreak:
                self.nobreak_original.append(breaktime)

            self._add_plots(adjust_obj, i)

    def build_frames_from_init_testing(self):
        """
        Creates time frames over homogeneous parts, if this module did not run
        before adjustment, the default time frames are used.

        Runs the break testing for the default break times and then uses the
        test results to re-build time frames from break time that are untestable
         or contain breaks and ignores such break times where no break was found.

        Returns
        -------
        init_testresults : dict
            Initial testresults combined in a dictionary
        """

        init_test_not_testable, init_test_testable, init_test_breaks, \
        init_test_nobreaks, init_testresults = self._init_check_testing_possible()

        self.init_test_testable = [str(testable.date()) for testable in sorted(
            init_test_testable, reverse=True)]

        # contains times with breaks and times that could not be tested initially
        breaktimes = init_test_breaks + init_test_not_testable
        breaktimes = [str(bt.date()) for bt in sorted(breaktimes, reverse=True)]

        times = self.otimes.get_times(as_datetime=False)
        start = times['ranges'][0]
        end = times['ranges'][1]

        if self.otimes.base_breaktime is not None:
            raise Exception('Building adjustment frames with different base breaktime does not work')

        # this time frame object is only over positively tested break times
        self.otimes = TsTimeFrames(start=start, end=end,
                                   breaktimes=breaktimes, min_set_days=None,
                                   skip_breaktimes=None, ignore_position=True,
                                   grid=None, base_breaktime=None)

        self.otestable = TsTimeFrames(start=self.otimes.ranges[0],
                                      end=self.otimes.ranges[1],
                                      breaktimes=self.init_test_testable, min_set_days=None,
                                      skip_breaktimes=None, ignore_position=True,
                                      grid=None, base_breaktime=None)

        return init_testresults

    def get_validation_results(self):
        """
        Return results on the cci_validation of the adjusted/tested data sets.
        """
        self.get_validation_stats(self.df_original)

    def candidate_has_changed(self):
        """
        Check if the candidate has changed during adjustment or if adjustment
        did not have any impact.

        Returns
        -------
        has_changed : bool
            True if the adjustment changed the time series, False if not
        """
        if self.adjusted_col_name not in self.df_original.columns:
            return False

        orig = self.df_original.dropna()[self.candidate_col_name].values
        adj = self.df_original.dropna()[self.adjusted_col_name].values

        return not all(orig == adj)

    def get_results(self, breaktime=None):
        """
        Get the results for all or a specific break time

        Parameters
        ----------
        breaktime : datetime, optional (default: None)
            The break time for which results are returned. If None is passed,
            all are returned.

        Returns
        -------
        testresults_ifirst : dict
            The test results before adjustment was attempted (for the breaktime)
        testresults_ilast : dict
            The test results after adjustment was attempted the last time
            (for the breaktime)
        models_ifirst : dict
            Regression model parameters before adjustment was attempted
            (for the breaktime)
        models_ilast : dict
            Regression model parameters after adjustment was attempted the last
            time (for the breaktime)
        group_stats : dict
            Statistics (median, variance etc.) for the data before and after the
            break time (from self.df_original)
        vertical_metrics : dict
            Comparison statistics (mean_diff etc) for data before and after the
             break time (from self.df_original)
        hor_errors : dict
            Comparison statistics (mean_diff etc) between candidate or adjusted
            and reference data (from self.df_original)
        checkstats : dict
            Contains values from checking the input and output data
        """

        if not breaktime:
            return self.testresults_ifirst, self.testresults_ilast, \
                   self.models_ifirst, self.models_ilast, \
                   self.group_stats, self.vertical_metrics, self.hor_errors, \
                   self.checkstats

        else:
            return self.testresults_ifirst[breaktime], self.testresults_ilast[breaktime], \
                   self.models_ifirst[breaktime], self.models_ilast[breaktime], \
                   self.group_stats[breaktime], self.vertical_metrics[breaktime], \
                   self.hor_errors[breaktime], self.checkstats[breaktime]


if __name__ == '__main__':
    pass
