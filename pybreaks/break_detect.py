# -*- coding: utf-8 -*-

from pybreaks.base import TsRelBreakBase
from pybreaks.utils import df_conditional_temp_resample
from pybreaks.model_lin_regress import LinearRegression


import pandas as pd

class TsRelBreakDetect(TsRelBreakBase):
    def __init__(self, candidate, reference, breaktime, resample=('M', 0.3),
                 spearR_sig=(0, 0.01), bias_corr_method='linreg'):
        """
        Initilize the break detection object

        Parameters
        ----------
        candidate : pd.Series or pd.DataFrame
            Pandas object containing the candidate time series
        reference : pd.Series or pd.DataFrame
            Pandas object containing the reference time series
        breaktime : datetime
            We dont know this, this is only for comparison.
        breaktime : datetime.datetime
            Time to test for a break
        resample : tuple or None
            time period and minimum data coverage in this period for resampling
            before testing
        spearR_sig : tuple
            Minimum r and maximum p for correlation between candidate and
            reference to attempt testing
        bias_corr_method : str or None
            Method for bias correction of reference data, as defined in pytesmo
        """

        TsRelBreakBase.__init__(self, candidate, reference, breaktime,
                                bias_corr_method, dropna=True)

        if resample:
            self.resample_to = resample[0]
            self.resample_threshold = resample[1]
            self.df_resampled = \
                df_conditional_temp_resample(self.df_original,
                                             self.resample_to,
                                             self.resample_threshold)

            self.df_resampled[self.reference_col_name] = \
                self._reference_bias_correction(frame=self.df_resampled,
                                                method=self.bias_corr_method,
                                                group=None)
        else:
            self.df_resampled = self.df_original.copy(True)




        self.min_corr = spearR_sig[0]
        self.max_p = spearR_sig[1]


        self.model_full = self._fit_single_regress_model()

    def _fit_single_regress_model(self, use_adjusted_col=False):
        """
        Take the input data (self.df_original) and fit all values with a single
        regression model.

        Returns
        -------
        model : LinearModel
            The linear model for all values


        """
        if use_adjusted_col:
            other_col_name = self.adjusted_col_name
        else:
            other_col_name = self.candidate_col_name

        data_group = self.get_group_data(None, self.df_resampled, [other_col_name,
                                                                  self.reference_col_name])

        data_group = data_group.dropna()

        subset_candidate = data_group[other_col_name]
        subset_reference = data_group[self.reference_col_name]

        model = LinearRegression(subset_candidate, subset_reference, None)

        return model



    def sse_around_breaktime(self, input_df, n_margin=12, use_adjusted_values=False,
                             filter_p=None):
        '''
        Calculates the sse (sum of squared residuals) function around the break time.
        2 linear models are calculated for multiple potential break times
        (margin around self.breaktime).
        SSEs for the 2 models are summed, to get an overall SSE for the function

        A break is most likely where the SSE function has a minimum (the 2 models fit best).
        This function can be used to confirm a detected break at a certain date,
        or search for a break in a prior known time period.

        Parameters
        -------
        input_df : pandas.DataFrame
            Data for which the residuals are calculated, must contain a reference data col
        n_margin : int
            Margin describes the number of dates (BEFORE AND AFTER self.breaktime)
            for which 2 regression models are calculated.
        use_adjusted_values : bool
            If this is true, force to use the 'adjusted' column in df
        filter_p : float
            For both models drop the passed percent (0-100) of worst candidate
            (high absolute difference) values compared to reference.

        Returns
        -------
        sse : pd.Series
            Series of sums of SSE of 2 linear models, for before and after the
            tested break time.
        min_date : datetime
            Date with the minimum SSE (where the break is)
        '''

        df = input_df.copy(True) # type: pd.DataFrame

        if use_adjusted_values:
            try:
                can_col_name = self.adjusted_col_name
            except AttributeError:
                print('no adjusted values in dataframe for the current class')
                return None
        else:
            can_col_name = self.candidate_col_name

        margintimes_before_breaktime = self.get_group_data(0, df, None)[-1 * n_margin:]
        margintimes_after_breaktime = self.get_group_data(1, df, None)[:n_margin]

        margintimes = [time.to_pydatetime() for time in margintimes_before_breaktime] +\
                      [time.to_pydatetime() for time in margintimes_after_breaktime]

        sse = pd.Series(index=margintimes)

        for time in margintimes:
            index0 = df.loc[:time].index # before the tested time
            index1 = df.loc[time + pd.DateOffset(1):].index # after the tested time

            # TODO raise exception if there is only 1 value in the index

            model0 = LinearRegression(candidate=df.loc[index0, can_col_name],
                                      reference=df.loc[index0, self.reference_col_name],
                                      filter_p=filter_p)

            model1 = LinearRegression(candidate=df.loc[index1, can_col_name],
                                      reference=df.loc[index1, self.reference_col_name],
                                      filter_p=filter_p)

            ssesum = model0.sse() + model1.sse()

            sse[time] = ssesum

        return sse, sse.idxmin().to_pydatetime()


    def calc_residuals_auto_corr(self, lags=range(30), use_adj_col=False):
        '''
        Calculates the autocorrelations of the residuals for a single linear model.
        If this varies for low lags it indicates the existence of a break.

        Parameters
        -------
        lags : list
            List of lags that are used for the autocorrelation function
            (unit same as self.df_original)
        use_adj_col : bool
            Force to use the self.adjusted_col_name column in
        Returns
        -------
        autocorr : pandas.Series
        Series of autocorrelation values for the selected lags
        '''

        model = self._fit_single_regress_model(use_adj_col)
        return model.residuals_autocorr(lags)
