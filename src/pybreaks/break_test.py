# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import fligner, mannwhitneyu, ranksums
from scipy import stats
import numpy as np
from pybreaks.fligner import fk_test
from pybreaks.base import TsRelBreakBase
from collections import OrderedDict
from pybreaks.utils import df_conditional_temp_resample
import copy
import warnings

'''
Class that contains statistical methods for homogeneity testing.
Allows testing of a single candidate series and an accrding reference, both are
separated into 2 parts by a potential break time. Testing compares the 2 parts
of the candidate relative to the reference.

TODO #################################
(+)

NOTES ################################
- 
'''


class TsRelBreakTest(TsRelBreakBase):
    """
    Module for detecting structural breaks in time series
    Using relative statistical methods
    """

    def __init__(self, candidate, reference, breaktime, test_resample=('M', 0.3),
                 bias_corr_method='linreg', alpha=0.01, mean_test='wilkoxon',
                 var_test='scipy_fligner_killeen', test_check_min_data=10,
                 test_check_spearR_sig=(0, 0.01)):
        """
        Initilize the break detection object

        Parameters
        ----------
        candidate : pd.Series or pd.DataFrame
            Pandas object containing the candidate time series
        reference : pd.Series or pd.DataFrame
            Pandas object containing the reference time series
        breaktime : datetime.datetime
            Time to test for a break
        test_resample : tuple, optional (default: ('M', 0.3))
            time period and minimum data coverage in this period for resampling
            before testing
        bias_corr_method : str or None, (default : 'linreg')
            Method for bias correction of reference data, as defined in pytesmo
        alpha : float, optional (default: 0.01)
            Minimum significance level to detect a break
        mean_test : str, optional (default: 'wilkoxon')
            Name of the test for breaks in the data means supported by this package
        var_test : str, optional (default: 'scipy_fligner_killeen')
            Name of the test for breaks in the data variances supported by this package
        test_check_min_data : int, optional (default: 5)
            Minimum number of values before / after breaktime to attempt testing
        test_check_spearR_sig : tuple, optional (default: (0, 0.01))
            Minimum r and maximum p for correlation between candidate and
            reference to attempt testing

        alpha : float
        """

        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method, dropna=True)

        self.alpha = alpha
        self.mean_test = mean_test
        self.var_test = var_test

        if test_resample:
            self.resample_to = test_resample[0]
            self.resample_threshold = test_resample[1]
            self.df_test_resampled = \
                df_conditional_temp_resample(self.df_original,
                                             self.resample_to,
                                             self.resample_threshold)
        else:
            self.df_test_resampled = self.df_original.copy(True)

        if self._check_notnull() and bias_corr_method:
            self.df_test_resampled[self.reference_col_name] = \
                self._reference_bias_correction(frame=self.df_test_resampled,
                                                method=self.bias_corr_method,
                                                group=None)

        self.test_min_data = test_check_min_data
        self.min_corr = test_check_spearR_sig[0]
        self.max_p = test_check_spearR_sig[1]

        self.error_code_test, self.error_text_test = self._check_input_data()

        self.df_test_resampled['Q'] = self.calc_diff(self.df_test_resampled)

    def _check_error(self):
        """
        Checks if the pre processing raised an error or not

        Returns
        -------
        status : bool
            True if everything is ok, False if an Error occurred
        """

        if self.error_code_test != 0:
            return False
        else:
            return True

    def _check_notnull(self):
        """
        Checks if this object contains any data for further processing, otherwise
        raises Exception

        Returns
        -------
        error_code: int
            Code for the error message, see meta dict for message to code
        error_message: str
            Error message for the code
        """
        candnull = self.df_test_resampled[self.candidate_col_name].isnull().all()
        refnull = self.df_test_resampled[self.reference_col_name].isnull().all()
        if candnull or refnull:
            return 1, 'No data for selected time frame'
        else:
            return 0, 'No error occurred'

    def _check_group_obs(self, min_obs_n):
        """
        Checks if in either group are more than the given number of minimum observations

        Parameters
        ----------
        min_obs_n : int
            Number of observations that must be at least in each of the 2 groups

        Returns
        -------
        error_code : int
            Code for the error message thrown here
        error_message : str
            Message for the error in this function
        n0 : int
            Number of observations for the group before the break time
        n1 : int
            Number of observations for the group after the break time


        """
        n0 = self.get_group_data(group_no=0, frame=self.df_test_resampled).size
        n1 = self.get_group_data(group_no=1, frame=self.df_test_resampled).size

        if n0 < min_obs_n or n1 < min_obs_n:
            return 3, 'min. obs N not reached. G1: %i, G2: %i (!> %i)' % (n0, n1, min_obs_n), n0, n1
        else:
            return 0, 'No error occurred', n0, n1

    def _check_spearman_corr(self, min_corr, max_p, group_no=None):
        """
        Check if the correlation of the selected data is above the passed
        thresholds

        Parameters
        ----------
        group_no: int or None
            Number of the group to check (0=before break, 1=after break)
        min_corr : float
            minimum correlation that is necessary
        max_p : float
            significance that is necessary

        Returns
        -------
        error_code : int
            Code for the error message in this function
        error_message : int
            Message for the error in this function
        corr : float
            Spearman correlation for the passed time series
        pval : float
            Significance value for the correlation
        """
        df = self.get_group_data(group_no, self.df_test_resampled, [self.candidate_col_name,
                                                                    self.reference_col_name])
        with warnings.catch_warnings(): # supress scipy warnings
            warnings.filterwarnings('ignore')
            corr, pval = stats.spearmanr(df[self.candidate_col_name], df[self.reference_col_name])

        if not (corr > min_corr and pval < max_p):
            msg = 'Spearman correlation failed with correlation %f and pval %f ' % (corr, pval)
            return 2, msg, corr, pval
        else:
            return 0, 'No error occurred', corr, pval

    def _check_input_data(self):
        """
        Calls functions for data pre-processing and checking

        Returns
        -------
        error_code_test : int
            Code that identifies which test failed
        error_text_test : str
            Text that describes the problem
        """

        n0, n1, corr, pval = np.nan, np.nan, np.nan, np.nan

        error_code_test = 0
        error_text_test = 'No error occurred'
        try:
            error_code_test, error_msg = self._check_notnull()
            if error_code_test == 0:
                error_code_test, error_msg, n0, n1 = self._check_group_obs(self.test_min_data)
                if error_code_test == 0:
                    error_code_test, error_msg, corr, pval = \
                        self._check_spearman_corr(self.min_corr, self.max_p)
            if error_code_test != 0:
                error_text_test = str(error_msg)
        except:
            error_code_test = 9
            error_text_test = 'Unknown Error'

        self.checkstats = {'n0': n0, 'n1': n1, 'frame_spearmanR': corr, 'frame_corrPval': pval}

        self.error_code_test = error_code_test
        self.error_text_test = error_text_test

        return self.error_code_test, self.error_text_test

    def _wk_test(self, alternative='two-sided', alpha=0.01):
        """
        Perform wilkoxon rank sums test for relative shifts in dataset mean ranks

        Parameters
        ----------
        alternative : str
            refer to documentation of stats.mannwitneyu
        alpha : float
            significance level for detection of a break

        Returns
        -------
        h : int
            1 if a break was found, 0 if not
        stats_wk : dict
            Tests statistics
        """

        q0 = self.get_group_data(0, self.df_test_resampled, ['Q'])
        q1 = self.get_group_data(1, self.df_test_resampled, ['Q'])

        u_wk, p_wk = mannwhitneyu(q0, q1, alternative=alternative)
        stats_wk = ranksums(q0, q1)[0]

        if p_wk <= alpha:
            h = 1
        else:
            h = 0

        stats_wk = {'zval': stats_wk, 'pval': p_wk}

        return h, stats_wk

    def _fk_test(self, mode='median', alpha=0.01):
        """
        Implementation of the original fk test


        Parameters
        ----------
        mode : str
            Use 'mean' or 'median' for fk statistics calculation
        alpha : float
            significance level for detection of a break

        Returns
        -------
        h : int
            1 if a break was found, 0 if not
        stats: dict
            Test statistics
        """

        q0 = self.get_group_data(0, self.df_test_resampled, ['Q'])
        q0['group'] = 0
        q1 = self.get_group_data(1, self.df_test_resampled, ['Q'])
        q1['goup'] = 1

        df = pd.concat([q0, q1], axis=0)

        h, stats_fk = fk_test(df, mode, 'X2', alpha)

        return h, stats_fk

    def _scipy_fk_test(self, mode='median', alpha=0.01):
        """
        Fligner Killeen Test for differences in data variances
        Scipy implementation uses the CHI2 approximation for calculation of the FK
        statistics.

        Parameters
        ----------
        mode
        alpha

        Returns
        -------
        h : int
            0 if no break found, 1 if break was found
        stats_fk : dict
            Fligner test statistics
        """
        q0 = self.get_group_data(0, self.df_test_resampled, ['Q'])
        q1 = self.get_group_data(1, self.df_test_resampled, ['Q'])

        with warnings.catch_warnings(): # supress scipy warnings
            warnings.filterwarnings('ignore')
            fstats, pval = fligner(q0, q1, center=mode)

        stats_fk = {'z': fstats, 'pval': pval}

        if stats_fk['pval'] <= alpha:  # With CHI2 approximation
            h = 1
        else:
            h = 0

        return h, stats_fk

    def _lv_test(self):
        """
        Implementation of Levene Test

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError('Levene Test is not implemented')

    def check_test_results(self):
        """
        Checks if any / which test found a break

        Returns
        -------
        status : bool
            True if any break was found, False if both are negative
        identifier : str
            Identifies the test(s) that found a break
        """

        mean_test_results = self.testresults['mean']['h']
        var_test_results = self.testresults['var']['h']

        if mean_test_results == 1 and var_test_results == 0:
            return True, 'mean'
        elif mean_test_results == 0 and var_test_results == 1:
            return True, 'var'
        elif mean_test_results == 1 and var_test_results == 1:
            return True, 'both'
        elif mean_test_results == 0 and var_test_results == 0:
            return False, None
        elif np.isnan(mean_test_results) and np.isnan(var_test_results):
            return None, None
        else:
            raise Exception('Unexpected test result')

    def run_tests(self):
        """
        Runs the selected tests and homogenizes the output to match a common format for all
        supported tests by this object.

        Returns
        -------
        isbreak : bool
            True if any break is found, False otherwise
        breaktype : str
            'mean' or 'var' or 'both', which kind of break was found
        testresult : dict
            The test statistics for the selected tests
        error_code_test : int
            Status information, whether testing could be performed or why not
            (see meta data dictionary)
        """

        tests = {'mean': self.mean_test, 'var': self.var_test}
        testresults = {}

        if not self._check_error():
            if tests['mean']:
                testresults['mean'] = {'h': np.nan, 'stats': np.nan}
            if tests['var']:
                testresults['var'] = {'h': np.nan, 'stats': np.nan}
            # print(self.error_text_test)

        else:
            if tests['mean']:
                if tests['mean'] == 'wilkoxon':  # Run the tests for mean breaks
                    h_wk, stats_wk = self._wk_test(alternative='two-sided', alpha=self.alpha)
                    wilkoxon = {'h': h_wk, 'stats': stats_wk}
                    testresults['mean'] = wilkoxon
                else:
                    raise Exception("'%s' is not a supported test for detection of breaks in data means")
            else:
                testresults['mean'] = {'h': np.nan, 'stats': np.nan}

            if tests['var']:
                if tests['var'] == 'scipy_fligner_killeen':  # Run the tests for var breaks
                    h_fk, stats_fk = self._scipy_fk_test(mode='median', alpha=self.alpha)
                    fligner_killeen = {'h': h_fk, 'stats': stats_fk}
                    testresults['var'] = fligner_killeen
                elif tests['var'] == 'fligner_killeen':
                    h_fk, stats_fk = self._fk_test(mode='median', alpha=self.alpha)
                    fligner_killeen = {'h': h_fk, 'stats': stats_fk}
                    testresults['var'] = fligner_killeen
                else:
                    raise Exception("'%s' is not a supported test for detection of breaks in data variances")
            else:
                testresults['var'] = {'h': np.nan, 'stats': np.nan}

        self.testresults = testresults

        self.isbreak, self.breaktype = self.check_test_results()

        return self.isbreak, self.breaktype, self.testresults, self.error_code_test

    def _ts_props(self):
        """ Specific for each child class """
        props = {'adjusted': False,
                 'isbreak': self.isbreak,
                 'breaktype': self.breaktype,
                 'candidate_name': self.candidate_col_name,
                 'reference_name': self.reference_col_name,
                 'adjusted_name': None,
                 'adjust_failed': None}

        return props

    def get_results(self):
        """
        Get the results of the current test run as a dictionary

        Returns
        -------
        testresults : dict
            Dict containing test results and statistics
        teststatus : dict
            Dict containing the test status / error code
        checkstats : dict
        """
        error_dict = {'error_code_test': self.error_code_test,
                      'error_text_test': self.error_text_test}

        return self.testresults, error_dict, self.checkstats

    @staticmethod
    def _merge_test_results(test_results_dict, test_error_dict):
        """
        Creates a simplified version of the test results dictionary and the
        test_error dictionary

        Parameters
        -------
        test_results_dict : dict
            Dictionary of test results
        test_error_dict : dict
            Dictionary of test results meta info

        Returns
        -------
        merged_test_results : dict
            The combined input dicts
        """
        merged_test_results = {}
        for testtype, testresults in test_results_dict.items():
            for name, val in testresults.items():
                if name == 'stats':
                    if isinstance(val, dict):
                        for n, v in val.items():
                            merged_test_results['%s_%s' % (n, testtype.upper())] = v
                    else:
                        continue
                else:
                    merged_test_results['%s_%s' % (name, testtype.upper())] = val

        merged_test_results['error_code_test'] = test_error_dict['error_code_test']

        return merged_test_results

    def get_flat_results(self):
        """
        Get test results as a single dictionary

        Returns
        -------
        results : dict
            1-Level dictionary of the results
        """
        test_results, error_dict, framestats = self.get_results()
        test_results = self._merge_test_results(test_results, error_dict)

        results = copy.deepcopy(test_results)
        results.update(framestats)

        return results

    @staticmethod
    def get_test_meta():
        """
        Returns a dictionary of error messages and error codes that my arise during testing

        Returns
        -------
        status_meta : OrderedDict
            Dictionary containing all error codes and messages
        """
        cont = [('0', 'No error occurred'),
                ('1', 'No data for selected time frame'),
                ('2', 'Spearman correlation failed'),
                ('3', 'Min. observations N not reached'),
                ('4', ''),
                ('5', ''),
                ('6', ''),
                ('7', ''),
                ('8', ''),
                ('9', 'Unknown Error')]

        test_meta = OrderedDict(cont)

        return test_meta


if __name__ == '__main__':
    pass

