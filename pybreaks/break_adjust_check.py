# -*- coding: utf-8 -*-

import scipy.stats as stats
import pandas as pd
import numpy as np
import calendar
from datetime import datetime

from pybreaks.base import TsRelBreakBase
from pybreaks.temp_coverage import compare_temp_cov

class BreakAdjustInputCheck(TsRelBreakBase):
    """
    Class that implements a set of methods for checking data before adjusting it
    """

    def __init__(self, candidate, reference, breaktime, timeframe, input_resolution,
                 group=None, error_code_pass=[0]):
        '''
        Parameters
        ----------
        candidate : pd.Series
            THe candidate series that is checked
        reference :  pd.Series
            The reference series that is used for checking
        breaktime : datetime
            The date of the potential break
        timeframe : tuple
            The timeframe of the 2 groups that are checked
        input_resolution : str
            The input resolution of the data D or M
        group : int or None
            Only check one group (0 or 1), if None, check both
        error_code_pass : list
            Values that do not lead to an error
        '''
        # TODO: bias correction again, so we dont detect the adjustment as a bias?

        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method=None, dropna=True)

        self.group = group

        self.df_check = self.df_original.copy(True)

        self.timeframe = timeframe
        self.input_resolution = input_resolution
        self.error_code_pass = error_code_pass

        self.checkstats = {}
        self.mdrop = None

    def check(self, min_group_range=None, pearsR_sig=None, coverdiff_max=None):
        '''
        Calls functions for performing adjustment specific checks.

        Parameters
        -------
        min_group_range : int, optional (default:None)
            Minimum covered days by either adjustment group (range between first
            and last day, not individual observations).
        pearsR_sig : tuple or None, optional (dafault: None)
            Minimum r and maximum p for pearson correlation between candidate and
            reference OF EACH GROUP, to attempt adjustment.
            At low correlation the linear model is not representative of the data.
        coverdiff_max : float, optional (default: None)
            Between 0 and 1, how large the difference in relative temporal
            coverage for any month may be.

        Returns
        -------
        error_code_adjust : int
            The error code for the check that failed
        error_text_adjust : str
            The text for the check that failed
        '''

        self.error_code_adjust = 0
        self.error_text_adjust = 'No error occurred'

        self.checkstats.update({'error_code_adjust': self.error_code_adjust})


        if min_group_range is not None:
            self._check_group_length(min_days=min_group_range)
            self.checkstats.update({'error_code_adjust': self.error_code_adjust})

            if not self.checks_pass(): return

        if pearsR_sig is not None:
            self._check_pearson_corr(min_corr = pearsR_sig[0], max_p=pearsR_sig[1])
            self.checkstats.update({'error_code_adjust': self.error_code_adjust})

            if not self.checks_pass(): return

        if coverdiff_max is not None:
            self._check_temp_coverage(coverdiff_max)
            self.checkstats.update({'error_code_adjust': self.error_code_adjust})

            if not self.checks_pass(): return

        '''
        #todo: not implemented
        self._check_frame_errors()
        if not self.checks_pass(): return
        '''

    def _frame_times_to_months(self):
        '''
        For monthly data, the time frame and break time has to be adapted
        '''
        start = self.timeframe[0]
        start_days = calendar.monthrange(start.year, start.month)
        start = datetime(start.year, start.month, start_days)

        end = self.timeframe[1]
        end_days = calendar.monthrange(end.year, end.month)
        end = datetime(end.year, end.month, end_days)

        breaktime = self.breaktime
        breaktime_days = calendar.monthrange(breaktime.year, breaktime.moth)
        breaktime = datetime(breaktime.year, breaktime.month, breaktime_days)

        return start, breaktime, end

    def _check_temp_coverage(self, coverdiff_max=0.5):
        '''
        Check the coverage of the candidate data of the 2 groups.
        Create an error, if the coverage comparison fails or is below the
        selected threshold

        Parameters
        ----------
        coverdiff_max : float, between 0 and 1
            The maximum difference for day/month that the coverage of the 2 groups
            may differ.
        try_fix : bool
            Fix the coverage differences by dropping unequal months in both
            groups and the re-test.
        '''

        ccn = self.candidate_col_name

        ds0 = self.get_group_data(0, self.df_check, [ccn])[ccn].dropna()
        ds1 = self.get_group_data(1, self.df_check, [ccn])[ccn].dropna()

        if self.input_resolution == 'M':
            resolution = 'M'
            start1, end1, end2 = self._frame_times_to_months()
            start2 = (end1 + pd.DateOffset(months=1)).to_pydatetime()
        else:
            resolution = 'D'
            start1, end1 = self.timeframe[0], self.breaktime
            start2 = (self.breaktime + pd.DateOffset(days=1)).to_pydatetime()
            end2 = self.timeframe[1]

        succ, cover1, cover2, dcover, mdrop = \
            compare_temp_cov(ds0, ds1, start1, end1, start2, end2, resolution,
                             coverdiff_max)

        self.checkstats['THRES_max_coverdiff'] = coverdiff_max

        for month in dcover.index:
            self.checkstats['dTempCover_M%i' % month] = dcover.at[month, 'diff']

        if 'TempCoverFit_nMonthsToDrop' not in self.checkstats.keys():
            self.checkstats['TempCoverFit_nMonthsToDrop'] = np.nan

        if not succ:
            self.mdrop = mdrop
            self.checkstats['TempCoverFit_nMonthsToDrop'] = mdrop.size
            self.error_code_adjust = 5
            self.error_text_adjust = 'Group temp. coverage diff above %i for at least 1 month' \
                                     % coverdiff_max

    def _check_frame_errors(self):
        '''
        Check if the SM error of the 2 groups is compareable.
        High error might lead to unfitting linear models.

        Returns
        -------

        '''
        # TODO: implement, use the noise column of the CCI SM data

        pass

    def _check_group_length(self, min_days):
        """
        Check if each group spans over a sufficiently long time period.

        Parameters
        ----------
        min_days : int
            Number of days that have to be covered by each group.
            Rhe range between first and last observation, not the number
            of actual observations counts!

        Returns
        -------
        """
        self.checkstats['THRES_min_days'] = min_days
        ccn = self.candidate_col_name
        rcn = self.reference_col_name

        groups = [0, 1] if self.group is None else [self.group]

        for group_no in groups:
            df = self.get_group_data(group_no, self.df_check.dropna(), [ccn, rcn])

            can_span = (df.iloc[-1].name - df.iloc[0].name).days

            if can_span < min_days:
                self.error_code_adjust = 6
                self.error_text_adjust = 'Group temp. coverage span too short'

    def _check_pearson_corr(self, min_corr, max_p):
        """
        Check if the correlation of the 2 groups is above the passed threshold

        Parameters
        ----------
        min_corr : float
            minimum correlation that is necessary
        max_p : float
            significance that is necessary

        Returns
        -------
        """
        self.checkstats['THRES_R_pearson'] = min_corr
        self.checkstats['THRES_p_pearson'] = max_p

        ccn, rcn = self.candidate_col_name, self.reference_col_name

        groups = [0, 1] if self.group is None else [self.group]

        for group_no in groups:
            df = self.get_group_data(group_no, self.df_check.dropna(),[ccn, rcn])

            corr, pval = stats.pearsonr(df[ccn], df[rcn])

            self.checkstats['R_pearson_group%i' % group_no] = corr
            self.checkstats['p_pearson_group%i' % group_no] = pval

            if not (corr > min_corr and pval < max_p):
                self.error_code_adjust = 2
                self.error_text_adjust = 'Group Pearson R for adjustment failed'

    def checks_pass(self):
        """
        Checks if the adjustment raised an error or not.

        Parameters
        -------
        error_codes_pass : list
            List of error codes that dont lead to the checks to fail.

        Returns
        -------
        checks_pass : bool
            Indicates whether all checks passed or not.
        """
        if self.error_code_adjust not in self.error_code_pass:
            return False
        else:
            return True



if __name__ == '__main__':
    from cci_timeframes import CCITimes
    from io_data.otherfunctions import smart_import

    qdegdata = False
    gpi = 402962  # bad: 395790,402962
    canname = 'CCI'
    refname = 'REF'
    adjname = 'ADJ'

    times = CCITimes('CCI_41_COMBINED', min_set_days=None, skip_breaktimes=[1,3]). \
        get_times(gpi=gpi)

    ts_full, plotpath = smart_import(gpi, canname, refname, adjname)

    ts_full = ts_full[[canname, refname]]

    ts_full['original'] = ts_full[canname].copy(True)

    adjustmeth = 'PairRegress'

    breaktime = times['breaktimes'][1]
    timeframe = times['timeframes'][1]

    ts_frame = ts_full[timeframe[0]: timeframe[1]].copy(True)

    check_kwargs =  {'pearsR_sig' : (0, 0.5),
                     'min_group_range': 365,
                     'coverdiff_max' : 0.5}

    check = BreakAdjustInputCheck(ts_frame[canname], ts_frame[refname], breaktime, timeframe,
                          'D', 1)

    check.check(**check_kwargs)