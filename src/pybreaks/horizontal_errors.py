# -*- coding: utf-8 -*-
"""
Module to calculate error metrics for a CAN and a REF series grouped by a date
(break time) to compare the performance of different parts of the full series.
"""

import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats as scistats
from pytesmo import metrics
import os
import warnings

os.environ["PYDEVD_USE_FRAME_EVAL"] = "NO"


def compare(v0, v1, how='AbsDiff'):
    """
    Takes the horizontal comparison metrics for a candidate set and compares it to the metric
    for a reference data set via the selected function:

    Parameters
    ----------
    v0 : float
        Value 0 (first in operation)
    v1 : float
        Value 1 (second in operation)
    how : str
        Implemented comparison operation.

    Returns
    -------
    error : float
        The error from comparing the 2 inputs via the comparison method
    """
    v0, v1 = float(v0), float(v1)

    if np.isnan(v0) or np.isnan(v1):
        return np.nan

    if how == 'Ratio':
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                rat = np.true_divide(v0, v1)
            if np.isinf(rat):
                raise ZeroDivisionError
        except ZeroDivisionError:
            rat = np.nan
        return rat
    elif how == 'Diff':
        return v0 - v1
    elif how == 'AbsDiff':
        return np.abs(v0 - v1)
    else:
        raise ValueError(how, "Unknown definition to combine the metrics")


class HorizontalVal(object):
    """
    Calculate statistics for a candidate and reference in each group,
    and across the 2 groups
    """
    def __init__(self, candidate, reference, breaktime):
        """
        Initialze the object for horizontal cci_validation.

        Parameters
        ----------
        candidate : pandas.Series
            The candidate series (no bias!)
        reference : pandas.Series
            The reference series (no bias!)
        breaktime : datetime
            Time to split the candidate and reference into 2 sets
            The breaktime is part of group 1 TODO: It should belong to group 2
        """
        self.candidate_name = candidate.name
        self.reference_name = reference.name

        self.df = pd.DataFrame(data={self.candidate_name: candidate.copy(True),
                                     self.reference_name: reference.copy(True)})

        self.breaktime = breaktime

        self.set0 = self.df.loc[:breaktime]
        self.set1 = self.df.loc[breaktime + pd.DateOffset(1):]
        self.setfull = self.df.copy(True)

        # this method is the INITIAL comparison (within the group)
        self.stats = np.array([('mean', 'Diff'),
                               ('median', 'Diff'),
                               ('min', 'Diff'),
                               ('max', 'Diff'),
                               ('var', 'Ratio'),
                               ('iqr', 'Ratio')])

        # for these metrics, the comparison will be always Diff
        self.metrics = np.array(['bias', 'mad', 'rmsd', 'nrmsd',
                                 'PearsonR', 'SpearmanR'])

    def _get_group_stats(self, can=True, ref=True):
        if can and ref:
            return self.df_group_stats
        elif can:
            return self.df_group_stats[self.df_group_stats.index.str.contains(self.candidate_name)]
        elif ref:
            return self.df_group_stats[self.df_group_stats.index.str.contains(self.reference_name)]
        else:
            return None

    def _calc_stats(self):
        """
        Calculate the statistical values needed for comparison within and
        across groups.

        Returns
        -------
        group_stats : pd.DataFrame
            Frame that contains the group stats
        """
        df_group_stats = pd.DataFrame()
        basic_measures = self.stats[:, 0]

        for group_no, subset_data in enumerate([self.set0, self.set1, self.setfull]):
            if group_no in [0, 1]:
                group = 'group%i' % group_no
            else:
                group = 'FRAME'
            for name in [self.candidate_name, self.reference_name]:
                # Basic measures
                if 'min' in basic_measures:
                    min = np.nan if subset_data[name].empty else \
                        np.min(subset_data[name].values)
                    df_group_stats.at['min_%s' % name, '%s' % group] = min
                if 'max' in basic_measures:
                    max = np.nan if subset_data[name].empty else \
                        np.max(subset_data[name].values)
                    df_group_stats.at['max_%s' % name, '%s' % group] = max
                if 'var' in basic_measures:
                    var = np.nan if subset_data[name].empty else \
                        np.var(subset_data[name].values)
                    df_group_stats.at['var_%s' % name, '%s' % group] = var
                if 'mean' in basic_measures:
                    mean = np.nan if subset_data[name].empty else \
                        np.mean(subset_data[name].values)
                    df_group_stats.at['mean_%s' % name, '%s' % group] = mean
                if 'median' in basic_measures:
                    median = np.nan if subset_data[name].empty else \
                        np.median(subset_data[name].values)
                    df_group_stats.at['median_%s' % name, '%s' % group] = median
                if 'iqr' in basic_measures:
                    iqr = np.nan if subset_data[name].empty else \
                        scistats.iqr(subset_data[name].values, nan_policy='omit')
                    df_group_stats.at['iqr_%s' % name, '%s' % group] = iqr

        return df_group_stats

    def _calc_stats_compare(self):
        """
        Compare stats within a group.
        We find e.g. the difference in min, max, mean between the candidate and
        reference of each group.

        Returns
        -------
        df_groupstats_compare: pd.DataFrame
            Metric comparison of the groups
        """
        df_groupstats_compare = pd.DataFrame()

        # from basic stats
        for var, meth in self.stats:
            for group in ['group0', 'group1', 'FRAME']:
                index = '{}_{}'.format(var, self.candidate_name)
                candidate_metric = self.df_group_stats.loc[index, group]
                index = '{}_{}'.format(var, self.reference_name)
                reference_metric = self.df_group_stats.loc[index, group]

                vmetric = compare(candidate_metric, reference_metric, meth)

                index = '{}_{}'.format(var, meth)
                df_groupstats_compare.loc[index, group] = vmetric

        return df_groupstats_compare


    def _calc_metric_change(self, df_groupstats_compare, df_validation_metrics,
                            how='Diff'):
        '''
        Compare changes in  metrics across the the 2 groups
        via the passed comparison function (usually difference).

        Parameters
        ----------
        df_groupstats_compare : pd.DataFrame
            DataFrame that contains the group stats
        df_validation_metrics : pd.DataFrame
            DataFrame that contains the group metrics
        how : str, optional (default: 'Diff')
            Method how the groups are compared, by default use the difference

        Returns
        -------
        df_metric_change : pd.DataFrame
            Change of metrics between the groups across the break time.
        '''

        df_metric_change = pd.Series(dtype='float64')

        for var, meth in self.stats:
            s0 = df_groupstats_compare.loc['%s_%s' % (var, meth), 'group0']
            s1 = df_groupstats_compare.loc['%s_%s' % (var, meth), 'group1']
            hmetric = compare(s0, s1, how)
            df_metric_change.at['%s_%s_%s' % (how, var, meth)] = hmetric

        for var in self.metrics:
            s0 = df_validation_metrics.loc[var, 'group0']
            s1 = df_validation_metrics.loc[var, 'group1']
            hmetric = compare(s0, s1, how)
            df_metric_change.at['%s_%s' % (how, var)] = hmetric

        return df_metric_change.to_frame('group0_group1')

    def _calc_validation_metrics(self):
        """
        Calculate vertical metrics between candidate and reference using pytesmo.

        Currently implemented:
            bias, mad, rmsd, nrmsd,
        Returns
        -------
        df_validation_metrics: pd.DataFrame
            Data Frame that contains the metrics between the candidate and reference
            for the 2 groups
        """
        df_validation_metrics = pd.DataFrame()

        for group_no, subset_data in enumerate([self.set0, self.set1, self.setfull]):
            if group_no in [0,1]:
                group = 'group%i' % group_no
            else:
                group = 'FRAME'
            if 'bias' in self.metrics:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    bias = np.nan
                else:
                    bias =metrics.bias(subset_data[self.reference_name].values,
                                       subset_data[self.candidate_name].values)
                df_validation_metrics.at['bias', '%s' % group] = bias

            if 'mad' in self.metrics:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    mad = np.nan
                else:
                    mad =metrics.mad(subset_data[self.reference_name].values,
                                     subset_data[self.candidate_name].values)
                df_validation_metrics.at['mad', '%s' % group] = mad

            if 'rmsd' in self.metrics:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    rmsd = np.nan
                else:
                    rmsd =metrics.rmsd(subset_data[self.reference_name].values,
                                       subset_data[self.candidate_name].values)
                df_validation_metrics.at['rmsd', '%s' % group] = rmsd

            if 'nrmsd' in self.metrics:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    nrmsd = np.nan
                else:
                    nrmsd =metrics.nrmsd(subset_data[self.reference_name].values,
                                         subset_data[self.candidate_name].values)
                df_validation_metrics.at['nrmsd', '%s' % group] = nrmsd

            if 'PearsonR' in self.metrics:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    pr, pp = np.nan, np.nan
                else:
                    with warnings.catch_warnings():  # supress scipy warnings
                        warnings.filterwarnings('ignore')
                        pr, pp =metrics.pearsonr(subset_data[self.reference_name].values,
                                                 subset_data[self.candidate_name].values)

                df_validation_metrics.at['PearsonR', '%s' % group] = pr
                df_validation_metrics.at['Pp', '%s' % group] = pp

            if 'SpearmanR' in self.metrics:
                if any([subset_data[col].empty for col in [self.candidate_name, self.reference_name]]):
                    sr, sp = np.nan, np.nan
                else:
                    with warnings.catch_warnings():  # supress scipy warnings
                        warnings.filterwarnings('ignore')
                        sr, sp = metrics.spearmanr(subset_data[self.reference_name].values,
                                                   subset_data[self.candidate_name].values)

                df_validation_metrics.at['SpearmanR', '%s' % group] = sr
                df_validation_metrics.at['Sp', '%s' % group] = sp

        return df_validation_metrics


    def run(self, comparison_method='AbsDiff'):
        '''
        Calculate the measures and compare them over the break time

        Parameters
        ----------
        comparison_method : str, optional (default: AbsDiff)
            How the metrics across the groups are compared. Default
            is the difference between group 1 and group 2.

        Returns
        -------
        df_metric_change : pd.DataFrame
            How the metrics and group stats have changed across the break time.

        '''
        # First calculate the stats for can and ref in both groups
        self.df_group_stats = self._calc_stats() # basic stats
        # Then use the stats to compare them (between can and ref)
        df_groupstats_compare = self._calc_stats_compare()
        # Then calculate some validation metrics
        df_validation_metrics = self._calc_validation_metrics()

        # Merge the group and validation metrics in 1 big data frame
        # ("vertical metrics")
        self.df_group_metrics = pd.concat([df_groupstats_compare,
                                           df_validation_metrics], axis=0)

        # compare changes in stats metrics and validation metrics "hor. errors")
        self.df_metric_change = self._calc_metric_change(df_groupstats_compare,
                                                         df_validation_metrics,
                                                         how=comparison_method)

        return self.df_metric_change


def usecase():
    can = pd.Series(index=pd.date_range(start='2000-01-01', end='2000-12-31', freq='D'),
                    data=np.random.rand(366), name='thecan')

    ref = pd.Series(index=pd.date_range(start='2000-01-01', end='2000-12-31', freq='D'),
                    data=np.random.rand(366), name='theref')

    ds = HorizontalVal(can, ref, datetime(2000, 1, 7))

    errors = ds.run('Diff')


if __name__ == '__main__':
    usecase()

