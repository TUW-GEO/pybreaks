# -*- coding: utf-8 -*-


from pytesmo.scaling import linreg_stored_params, linreg_params, mean_std, min_max
from pytesmo.cdf_matching import CDFMatching
import pandas as pd
from datetime import datetime
import numpy as np
from pybreaks.horizontal_errors import HorizontalVal
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

"""
This is the base class for break detection and adjustment. It contains methods 
that come in handy when creating data frames, splitting data for groups depending 
on the break time and for bias correction between candidate and reference.
"""

# TODO:
#   (+) Make the plot_stats_ts function return nicer results
#   (o) Add properties, getters, setters
#----------
# NOTES:
#   - Reference CDF scaling uses the same percentiles always, give option to select?

class TsRelBreakBase(object):
    def __init__(self, candidate, reference, breaktime,
                 bias_corr_method='linreg', dropna=False):
        """
        Initialize the break base object.
        All other classes for break detection and adjustment build on this base
        class, which contains helper functions, plotting and data organization
        for 2 time series (candidate and reference) around a single break time.

        Parameters
        ----------
        candidate : pd.Series or pd.DataFrame
            Pandas object containing the candidate time series
        reference : pd.Series or pd.DataFrame
            Pandas object containing the reference time series
        breaktime : datetime or None
            Time that separates the dataframe into 2 groups, if None is passed
            there is only one large group.
        bias_corr_method : str, optional (default: 'linreg')
            Name of the method that is used to correct the bias between
            candidate and reference. Currently supported methods are
            'linreg' (for scaling the reference to the candidate based on
            fitting slope and intercept of the 2 series) and 'cdf_match'
            (for scaling the reference to the candidate based on fitting
            percentiles 0, 5, 10, 30, 50, 70, 90, 95 and 100 of the reference
            to the candidate with linear interpolation.)
        dropna : bool, optional (default: False)
            Keep only common values between the candidate and the reference series.
            As the implemented testing and adjustment methods build on difference
            values between candidate and reference, it is necessary to use only
            common values of the 2. In cases where the reference contains less
            values than the candidate, this option will lead to a loss in
            candidate data and can therefore be deactivated.
        """

        candidate = candidate.copy(True)  # type: pd.Series
        reference = reference.copy(True)  # type: pd.Series

        if isinstance(breaktime, str):
            self.breaktime = datetime.strptime(breaktime, '%Y-%m-%d')
        else:
            self.breaktime = breaktime
        self.bias_corr_method = bias_corr_method

        self.df_original = self._make_dataframe(candidate, reference, dropna)

        empty = self.df_original[self.candidate_col_name].dropna().empty or \
            self.df_original[self.reference_col_name].dropna().empty

        if self.bias_corr_method and not empty:
            self.df_original[self.reference_col_name] = \
                self._reference_bias_correction(frame=self.df_original,
                                                method=self.bias_corr_method)

    def _ts_props(self):
        """ Specific for each child class, meta-info for plotting"""
        props = dict([('isbreak', None),
                      ('breaktype', None),
                      ('candidate_name', self.candidate_col_name),
                      ('reference_name', self.reference_col_name),
                      ('adjusted_name', None),
                      ('adjust_failed', None)])

        return props

    def _plotstyle(self, only_success=False):
        """ Creates plot styles based on the success of an adjustment step """
        props = self._ts_props()
        adjust_failed = props['adjust_failed']  # True: Failed, None: not tried, False: Success

        if only_success:
            not_adjusted = (props['adjusted_name'] is None) or (adjust_failed in [True, None])
        else:
            not_adjusted = props['adjusted_name'] is None

        adjusted = not not_adjusted

        if adjusted:
            style = ['r-', 'b--', 'k-']
        else:
            style = ['r-', 'k-']

        return adjusted, style

    def _make_dataframe(self, candidate, reference, dropna=True):
        """
        Take the input candidate and reference and build a dataframe
        with columns 'candidate', 'reference' and

        Parameters
        -------
        candidate : pd.Series
            The candidate data set
        reference : pd.Series
            The reference data set
        dropna : bool, optional (default: True)
            Drop all lines, where either candidate or reference is nan, so the
            joined df contains no nans at all.

        Returns
        -------
        joined_df : pd.DataFrame
            The joined input data and difference time series
        """

        if isinstance(candidate, pd.DataFrame):
            original_name = candidate.columns.values[0]
            self.candidate_col_name = original_name
        else:
            try:
                self.candidate_col_name = candidate.name
                if self.candidate_col_name is None:
                    raise ValueError('No name')
            except:
                self.candidate_col_name = 'candidate'
            candidate = candidate.to_frame(name=self.candidate_col_name)

        if isinstance(reference, pd.DataFrame):
            original_name = reference.columns.values[0]
            self.reference_col_name = original_name
        else:
            try:
                self.reference_col_name = reference.name
                if self.reference_col_name is None:
                    raise ValueError('No name')
            except:
                self.reference_col_name = 'reference'
            reference = reference.to_frame(name=self.reference_col_name)

        joined_df = candidate.join(reference, how='outer')
        if dropna:
            joined_df = joined_df.dropna()

        return joined_df

    @staticmethod
    def _check_group_no(group_no):
        """
        Checks if the passed group is either 0 or 1, otherwise raises an exception.
        Used to set a main group (e.g. that is adjusted)

        Parameters
        -------
        group_no : int
            Group (0=before, 1=after break) of the group to make the main group

        Returns
        --------
        group_no : int
            Index (0 or 1) of the main group
        other_no : int
            Index (0 or 1) of the second group
        """

        if group_no not in [None, 0, 1]:
            raise Exception('Select None, 0 (before breaktime) or 1 (after breaktime)')
        if group_no is None:
            other_no = None
        else:
            other_no = 0 if group_no == 1 else 1

        return group_no, other_no

    def _reference_bias_correction(self, frame, method='linreg', group=None):
        """
        Scales the 'reference' column to the 'candidate' column via fitting
        of regression parameters.

        Parameters
        -------
        frame : DataFrame
            The DataFrame with the candidate and reference data
        method : str, optional (default: 'linreg')
            Method for bias correction as described in pytesmo
        group : int or None, optional (default: None)
            0 or 1, if a group is selected, bias is calculated only for values
            of the group and applied to the whole frame, if None is selected,
            bias is calculated from and applied to the full frame.

        Returns
        -------
        df_reference : pd.DataFrame
            The bias corrected input data frame reference column
        """

        dframe = self.get_group_data(None, frame, columns=[self.candidate_col_name,
                                                           self.reference_col_name])
        if dframe.index.size > 1:
            df = dframe.copy(True)
            if group:
                # reference data is changed...to fit the candidate!!
                src = self.get_group_data(group, df.dropna(),
                                          columns=[self.reference_col_name])
                src = src[self.reference_col_name].values

                can = self.get_group_data(group, df.dropna(),
                                          columns=[self.candidate_col_name])
                can = can[self.candidate_col_name].values
            else:
                src = df.dropna()[self.reference_col_name].values  # reference data is changed
                can = df.dropna()[self.candidate_col_name].values  # ...to fit the candidate

            if method == 'linreg':
                slope, inter = linreg_params(src, can)
                df[self.reference_col_name] = \
                    linreg_stored_params(df[self.reference_col_name], slope, inter)
            elif method == 'cdf_match':
                percentiles = [0, 5, 10, 30, 50, 70, 90, 95, 100]
                matcher = CDFMatching(percentiles=percentiles, )
                matcher.fit(src, can)
                scaled = matcher.predict(df[self.reference_col_name])
                scaled[scaled < 0] = 0

                df[self.reference_col_name] = scaled

            elif method == 'mean_std':
                df[self.reference_col_name] = mean_std(src, can)
            elif method == 'min_max':
                df[self.reference_col_name] = min_max(src, can)
            else:
                raise ValueError(method, 'Method for bias correction is not supported')

            return df[[self.reference_col_name]]

    def _rename_main_cols(self, df, can_name=None, ref_name=None, adj_name=None):
        """
        Rename the candidate, reference and adjusted data column if the exist
        in the passed data frame

        Parameters
        ----------
        df : pd.DataFrame
            The data frame in which the candidate, reference, adjusted are searched
        can_name : str, optional (default: None)
            If a name is passed, the candidate column is renamed
        ref_name : str, optional (default: None)
            If a name is passed, the reference column is renamed
        adj_name : str, optional (default: None)
            If a name is passed, the adjusted column is renamed if it exists.

        Returns
        -------
        df_renamed : pd.DataFrame
            The renamed input df data frame with the 2 (rep. 3) columns
        """
        if can_name:
            try:
                # todo: this is the actual candidate name in some child classes, not nice...
                df = df.rename(columns={self.original_candidate_col_name: can_name})
            except AttributeError:
                df = df.rename(columns={self.candidate_col_name: can_name})
        if ref_name:
            df = df.rename(columns={self.reference_col_name: ref_name})
        if adj_name:
            props = self._ts_props()
            if not props['adjust_failed'] and props['adjusted_name'] is not None:
                df = df.rename(columns={props['adjusted_name']: adj_name})
            else:
                pass

        return df

    def get_group_data(self, group_no, frame, columns=None, extended=False):
        """
        Get the selected columns of the selected group (0 or 1) from the
        main_verification.py dataframe.

        Parameters
        ----------
        group_no : int or None
            Group index (0 = before break time, 1 = after break time) or None
            for all values.
        frame : pd.DataFrame
            The data frame where the group data is taken from
        columns : list or 'all', optional (default: None)
            Name of the columns in the main_verification.py data frame. If this None, only the
            indices are returned. When 'all' is passed, all columns are returned.
        extended : bool, optional (default: False)
            Force the inclusion of the index before/after the break time.

        Returns
        -------
        df_sub : DataFrame
            Subset of the passed data frame
        """

        self._check_group_no(group_no)

        df = frame.copy(True)

        # TODO: BREAKTIME SHOULD BELONG TO THE SECOND GROUP
            # the date when the break is introduced is already biased

        if group_no == 0:
            index = df.loc[:self.breaktime].index
        elif group_no == 1:
            index = df.loc[self.breaktime + pd.DateOffset(1):].index
        else:
            index = df.loc[:].index

        if extended:
            if group_no == 0:
                # include also next one
                index = df.iloc[: df.index.get_loc(index[-1]) + 2].index
            elif group_no == 1:
                # include also the previous one
                index = df.iloc[df.index.get_loc(index[0]) - 1:].index
            else:
                raise Exception('Breaktime always included if no group selected.')

        if isinstance(df, pd.Series):
            return df.loc[index]
        elif not columns:
            return index
        elif columns == 'all':
            return df.loc[index, :]
        else:
            return df.loc[index, columns]

    def calc_diff(self, input_df):
        """
        Adds a new column to the passed dataframe for the difference values
        between candidate and reference in the main data frame

        Parameters
        -------
        input_df : pd.DataFrame
            Dataframe that contains a candidate and reference column.

        Returns
        -------
        df_Q : pd.DataFrame
            Difference Series between candidate and reference

        """
        df = input_df.copy(True)
        df['Q'] = df[self.candidate_col_name] - df[self.reference_col_name]
        if df['Q'].empty:
            return []
        else:
            return df[['Q']]  # todo: try Series instead of frame

    def get_validation_stats(self, frame, columns=None, comp_meth='AbsDiff',
                             can_name=None, ref_name=None, adj_name=None,
                             as_dict=False, digits=None):
        """
        Calculate statistics between 2 or three columns of the data frame.
        Comparison is performed against the column specified as the reference
        column.

        Parameters
        ----------
        frame : pd.DataFrame
            The input data frame with columns for which the stats are calculated
        columns : list, optional (default: None)
            Subset of column names in frame, which are used.
        comp_meth : str, optional (default: 'AbsDiff')
            Supported comparison method for calculating the horizontal metrics
            eg AbsDiff, Diff, Ratio,
        can_name : str, optional (default: None)
            Rename the candidate column to this name in the output df
        ref_name : str, optional (default: None)
            Rename the reference column to this name in the output df
        adj_name : str, optional (default: None)
            Rename the adjusted column to this name in the output df
        as_dict : bool, optional (default: False)
            Return the results in dictionary format
        digits : int, optional (default: None)
            Round the returned results to this number of decimals.

        Returns
        -------
         df_group_stats : pd.DataFrame or dict
            Statistics for the columns for the 2 groups
         df_group_metrics : pd.DataFrame or dict
            Error metrics between the candidate and reference for the 2 groups
         df_metric_change : pd.Series or dict
            Comparison across break time between error metrics of the 2 groups
        """

        frame = frame.dropna()

        if not columns:  # all columns
            columns = frame.columns.values.tolist()

        stats_data = self.get_group_data(None, frame, columns)

        stats_data = self._rename_main_cols(stats_data,
            can_name=can_name, ref_name=ref_name, adj_name=adj_name)

        if ref_name is None:
            ref_name = self.reference_col_name

        other_names = [c for c in stats_data.columns if c != ref_name]

        df_group_stats, df_group_metrics, df_metric_change = [], [], []

        for i, other_name in enumerate(other_names):
            can = stats_data[other_name]
            ref = stats_data[ref_name]

            prefix_str = '%s_%s_' % (other_name, ref_name)

            val = HorizontalVal(can, ref, self.breaktime)
            val.run(comparison_method=comp_meth)
            # group stats
            df_group_stats.append(val._get_group_stats(
                can=True, ref=True if i == 0 else False))
            # vertical metrics and rename
            metrics = val.df_group_metrics
            metrics.index = (prefix_str + val.df_group_metrics.index.astype(str))
            df_group_metrics.append(metrics)
            # horizontal errors and rename
            delta_metrics = val.df_metric_change
            delta_metrics.index = (prefix_str + val.df_metric_change.index.astype(str))
            df_metric_change.append(delta_metrics)

        df_group_stats = pd.concat(df_group_stats, axis=0)
        df_group_metrics = pd.concat(df_group_metrics, axis=0)
        df_metric_change = pd.concat(df_metric_change, axis=0)

        if digits:
            df_group_stats = df_group_stats.round(digits)
            df_group_metrics = df_group_metrics.round(digits)
            df_metric_change = df_metric_change.round(digits)

        if as_dict:  # Merge the index and col name to a single variable name
            df_group_stats_dict = {}
            df_group_metrics_dict = {}
            df_metric_change_dict = df_metric_change.to_dict()

            for col in df_group_stats.columns:
                for index in df_group_stats.index:
                    val = df_group_stats.loc[index, col]
                    df_group_stats_dict['%s_%s' % (index, col)] = val

            for col in df_group_metrics.columns:
                for index in df_group_metrics.index:
                    val = df_group_metrics.loc[index, col]
                    df_group_metrics_dict['%s_%s' % (index, col)] = val

            return df_group_stats_dict, df_group_metrics_dict, df_metric_change_dict
        else:
            return df_group_stats, df_group_metrics, df_metric_change

    def plot_ts(self, frame, title=None, only_success=False, ax=None,
                save_path=None):
        """
        Plot the passed frame only.

        Parameters
        ----------
        frame : pd.DataFrame
            Frame to read the time series from
        title : str, optional (default: None)
            Title that is used for the plot, if None is passed, it gets no title.
        only_success : bool, optional (default: False)
            Only plot the adjusted candidate if adjustment removed the break
            and all checks passed.
        ax : matplotlib.axes.Axes, optional (default: None)
            Axes that is used for the plot
        save_path : str, optional (default: None)
            Path to file where the image is stored.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Plot axes
        """

        if ax and save_path:
            raise Exception('Either create new figure to save or add to existing one.')

        props = self._ts_props()
        adjust_failed = props['adjust_failed']  # True: Failed, None: not tried, False: Success

        plot_df = frame.copy()  # type: pd.DataFrame

        adjusted, styles = self._plotstyle(only_success)

        if adjusted:
            plot_df = plot_df.loc[:, plot_df.columns[:3]]
            plot_df = plot_df[[props['candidate_name'], props['adjusted_name'],
                               props['reference_name']]]
        else:
            plot_df = plot_df.loc[:, plot_df.columns[:2]]
            plot_df = plot_df[[props['candidate_name'], props['reference_name']]]

        if not ax:
            fig_frame_ts = plt.figure(figsize=(10,4))
            ax = fig_frame_ts.add_subplot(1, 1, 1)
        else:
            fig_frame_ts = None

        if plot_df.empty:
            return

        for col, style in zip(plot_df.columns, styles):
            plot_df[col].dropna().plot(ax=ax, style=style)

        ax.set_title(title)
        plt.legend()

        ax.set_xlabel('Years')
        ax.set_ylabel('SM')

        if props['adjusted_name'] is not None:
            linestyle = 'solid'
        else:
            linestyle = 'dashed'

        if not props['isbreak']:
            if props['isbreak'] is None:
                color = 'grey'
            else:
                color = 'green'
        elif props['isbreak']:
            color = 'red'
        else:
            raise Exception('unexpected test result or status')

        if self.breaktime and adjust_failed:
            ax.axvline(self.breaktime, linestyle='solid', color='red', lw=2)
        elif self.breaktime:
            ax.axvline(self.breaktime, linestyle=linestyle, color=color, lw=2)
        else:
            pass

        if save_path:
            fig_frame_ts.savefig(save_path)
            plt.close(fig_frame_ts)

        return ax

    def plot_stats_ts(self, frame, kind='box', save_path=None, title=None, stats=False):
        """
        Create a plot of candidate and reference time series for the tested period.
        Indicates whether a break was found or not.
        Calculate time series stats for comparison and save the plot

        Parameters
        -------
        frame: pd.DataFrame
            The dataframe that is plotted
            eg self.df_original, or self.df_frame etc.
        kind : str, optional (default: 'box')
            Select 'box' or 'line 'to plots stats as line plots or box plots
        save_path : str, optional (default: None)
            Path to file where the image is stored
        title : str, optional (default: None)
            Title for the plot, if None is passed, it gets no title.
        stats: bool, optional (default: False)
            If True, adds stats in a text box to the empty panels of the figure.

        Returns
        -------
        fig_frame_ts : plt.figure
            Figure of the time series and stats plots
        """

        props = self._ts_props()

        adjust_failed = props['adjust_failed']  # True: Failed, None: not tried, False: Success
        breaktype = props['breaktype']
        isbreak = props['isbreak']
        if breaktype is None and isbreak is None:
            breaktype = 'Not Tested'

        rows, cols = 6, 3

        plot_df = frame.copy()  # type: pd.DataFrame

        fig_frame_ts = plt.figure(figsize=(15, 11))
        main_ax = fig_frame_ts.add_subplot(rows, 1, 1)

        _ = self.plot_ts(plot_df, title, only_success=False, ax=main_ax)

        adjusted, style = self._plotstyle(only_success=False)

        if adjusted:
            columns = {props['candidate_name']: 'CAN',
                       props['adjusted_name']: 'ADJ',
                       props['reference_name']: 'REF'}
        else:
            columns = {props['candidate_name']: 'CAN', props['reference_name']: 'REF'}

        if not all([col in plot_df.columns for col in columns.keys()]):
            print(self.breaktime)
            raise ValueError(columns, 'Cannot find column in data frame')

        plot_df = plot_df.rename(columns=columns)
        # plot_df = plot_df.loc[:, columns]

        if plot_df.empty:
            # there is no data to plot. return None
            return fig_frame_ts

        plot_df = plot_df.dropna()

        group_stats, vertical_metrics, horizontal_errors = \
            self.get_validation_stats(frame=plot_df,
                                      can_name='CAN',
                                      ref_name='REF',
                                      adj_name='ADJ' if 'ADJ' in plot_df.columns else None,
                                      comp_meth='AbsDiff', digits=5)
        horizontal_errors = horizontal_errors['group0_group1']
        stats_df = pd.DataFrame(index=plot_df.index)
        g0_index = self.get_group_data(0, plot_df, None)
        g1_index = self.get_group_data(1, plot_df)

        for statname in group_stats.index.values:
            stats_df.loc[g0_index, statname] = group_stats.loc[statname, 'group0']
            stats_df.loc[g1_index, statname] = group_stats.loc[statname, 'group1']

        if len(style) == 3:
            style = {'CAN': style[0], 'ADJ': style[1], 'REF': style[2]}
        else:
            style = {'CAN': style[0], 'REF': style[1]}

        # histograms in 1st and 2nd col
        for i, col in enumerate(plot_df.columns):
            ax = plt.subplot(rows, cols, cols * i + (cols + 1))
            sns.distplot(self.get_group_data(0, plot_df, [col]),
                         label=col, color=style[col][0])
            ax.axvline(group_stats.loc['mean_%s' % col, 'group%i' % 0],
                       color=style[col][0])
            if i == 0: ax.set_title('Before Breaktime')

        for i, col in enumerate(plot_df.columns):
            ax = plt.subplot(rows, cols, cols * i + (cols + 2))
            sns.distplot(self.get_group_data(1, plot_df, [col]),
                         label=col, color=style[col][0])
            ax.axvline(group_stats.loc['mean_%s' % col, 'group%i' % 1],
                       color=style[col][0])
            if i == 0:
                ax.set_title('After Breaktime')

        if kind == 'line':
            # mean var plots on the right
            ax = plt.subplot(rows, cols, 6)
            df_mean = stats_df[[name for name in group_stats.index.values if 'mean' in name]]
            df_mean.plot(title='Group Means', style=[style[n[-3:]] for n in df_mean.columns], ax=ax, legend=False)

            ax = plt.subplot(rows, cols, 9)
            df_med = stats_df[[name for name in group_stats.index.values if 'median' in name]]
            df_med.plot(title='Group Medians', style=[style[n[-3:]] for n in df_med.columns], ax=ax, legend=False)

            ax = plt.subplot(rows, cols, 12)
            df_var = stats_df[[name for name in group_stats.index.values if 'var' in name]]
            df_var.plot(title='Group Variances', style=[style[n[-3:]] for n in df_var.columns], ax=ax, legend=False)

            ax = plt.subplot(rows, cols, 15)
            df_iqr = stats_df[[name for name in group_stats.index.values if 'iqr' in name]]
            df_iqr.plot(title='Group IQRs', style=[style[n[-3:]] for n in df_iqr.columns], ax=ax, legend=False)

            ax = plt.subplot(rows, cols, 18)
            df_min = stats_df[[name for name in group_stats.index.values if 'min' in name]]
            df_min.plot(title='Group Mins', style=[style[n[-3:]] for n in df_min.columns], ax=ax, legend=False)
        elif kind == 'box':
            ax = plt.subplot(rows, cols, 6)
            data0 = self.get_group_data(0, plot_df, 'all')
            stats = cbook.boxplot_stats(data0.values, labels=columns, bootstrap=10000)
            ax.bxp(stats, showmeans=True, meanline=True)
            ax.set_title('Before Break', fontsize=12)

            ax = plt.subplot(rows, cols, 9)
            data1 = self.get_group_data(1, plot_df, 'all')
            stats = cbook.boxplot_stats(data1.values, labels=columns, bootstrap=10000)
            ax.bxp(stats, showmeans=True, meanline=True)
            ax.set_title('After Break', fontsize=12)
        else:
            raise ValueError(kind, 'Unknown style, choose box or line')

        if stats:
            # text stats in last panel
            n = horizontal_errors.index.size
            for i, r in enumerate([(0, n//2), (n//2, n)]):
                ax = plt.subplot(rows, cols, 13 + i)
                ax.text(0.5, 1, horizontal_errors.iloc[r[0]:r[1]].to_string(), fontsize=8,
                        va='top', ha='center')
                ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig_frame_ts.savefig(save_path)
            plt.close(fig_frame_ts)

        return fig_frame_ts

if __name__ == '__main__':
    pass
