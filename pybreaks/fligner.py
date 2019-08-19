# -*- coding: utf-8 -*-


import numpy as np
from scipy import stats


def fk_test(dataframe_in, mode='median', approx='X2', alpha=0.01):
    '''
    Fligner-Killeen test for homogeneity of variances.

    Trujillo-Ortiz, A., R. Hernandez-Walls and N. Castro-Castro. (2009).FKtest:
        Fligner-Killeen test for homogeneity of variances. A MATLAB file. [WWW document].
        URL http://www.mathworks.com/matlabcentral/fileexchange/25040

    Parameters
    ----------
    dataframe_in : pd.DataFrame
        Input Data Frame with 2 columns
            column1 (data): difference data Q
            column2 (group): group number (1 or 2 for reference data or testdata)
    mode : str
        Use 'mean' or 'median' for fk statistics calculation
    approx : str
        'X2' or 'F'
        enter X2 to use the ChiSquared Approximation for the statistics calculation or
        F to use the Fisher Approximation
    alpha : float
        significance level for detection of a break

    Returns
    -------
    h : int
        1 if a break was found, 0 if not
    stats: dict
        Test statistics

    '''


    # number of measurements and datagroups
    dataframe = dataframe_in.copy()
    df = dataframe.rename(columns={'Q': 'data'})
    df = df.dropna()
    N = df.index.size
    K = df['group'].nunique()

    df['A'] = np.nan

    if mode == 'median':
        for i in range(K):
            subset = df.ix[df['group'] == i]
            groupmed = subset.data.median()
            df.ix[df['group'] == i, 'groupmed'] = groupmed  # group medians
            df.ix[df['group'] == i, 'groupme_diff'] = np.abs(
                subset['data'] - groupmed)  # difference data-groupmedians

    if mode == 'mean':
        for i in range(K):
            subset = df.ix[df['group'] == i]
            groupmean = subset.data.mean()
            df.ix[df['group'] == i, 'groupmean'] = groupmean  # groupmeans
            df.ix[df['group'] == i, 'groupme_diff'] = np.abs(
                subset['data'] - groupmean)  # difference data-groupmeans

    Z = stats.rankdata(df['groupme_diff'])  # score ranking ALL
    sta_norm_dist = stats.norm.ppf(0.5 + (Z / (2. * (N + 1.))))  # score standard normal distribution ALL
    df['A'] = sta_norm_dist
    M = df['A'].mean()  # overall mean

    nn = []
    mm = []
    bb = []
    for i in range(K):
        subset = df.ix[df['group'] == i]

        nn.append(subset.index.size)
        mm.append(np.mean(subset['A']))
        df.ix[df['group'] == i, 'groupAmean'] = mm[i]
        bb.append((nn[i] * (mm[i] - M) ** 2))
        df.ix[df['group'] == i, 'groupB'] = bb[i]

    B = np.array(df['groupB'].unique())
    V = df['A'].var()  # Overall Variance Score

    X2 = np.sum(B) / V  # Fligner-Killeen statistic by the Chi-squared approximation
    v = K - 1  # statistic degree of freedom

    if approx == 'X2':
        P1 = 1 - stats.chi2.cdf(X2, v)
        stats_fk = {'z': X2, 'df': v, 'pval': P1}
    elif approx == 'F':
        F = (X2 / v) / ((N - 1. - X2) / (N - K))  # Fligner-Killeen statistic by the Fisher approximation
        P2 = 1 - stats.f.cdf(F, v, N - K)
        stats_fk = {'z': F, 'df': [v, N - K], 'pval': P2}
    else:
        raise ValueError(approx, 'Unknown approximation')
    # TODO: Laut Chun-Hsu statt F X2??


    if stats_fk['pval'] < alpha:
        h = 1
    else:
        h = 0

    return h, stats_fk