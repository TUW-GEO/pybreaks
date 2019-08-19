# -*- coding: utf-8 -*-
"""
Python 2 implementation of Lmoments CDF fitting. Needs lmoments (not lmoments3)
package installed.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from functools import partial
import lmoments as lm
import warnings
warnings.warn('lmoments2 is deprecated, use lmoments3 instead', DeprecationWarning)

class FitCDF2(object):
    '''
    Find the best fitting CDF to a set of passed values using a L-moments approach
    together with a KS test.
    '''
    def __init__(self, src, types=None):
        '''
        Fit the most likely distribution to the passed data.
        Parameters
        -------
        src : np.array
            Values that the CDF if fit to
        types: list or None, optional (default : None)
            Types of PDFs that are fit, the best fitting on from the passed list
            is then selected.
            Supported types (that can be in the list):
                - empirical (empirically fit the CDF)
                - 'nor' (fit a normal distribution)
                - 'pe3' (fit a Pearson Type 3 distribution)
                - 'gno' (fit a Generalized Normal distribution)
            If list contains just 1 item, this one is always used.
            If None are passed, all available are used.
        '''
        self.src = src

        self.name, self._cdf, self._pdf, self.para = self._find_cdf(types)


    def calc_quantiles(self, data):
        '''
        Find the quantiles for the passed values using the CDF.
        CDF(x) = q, eg. for a norm-dist CDF(data_mean) = ~50
        Returns
        -------
        '''
        quant = [self._cdf(d) for d in data]
        return quant

    def _nor(self, dat, l):
        '''Fir a Normal Distribution'''
        para = lm.pelnor(lm.samlmu(dat, l))
        _cdf = partial(lm.cdfnor, para=para)
        _pdf = partial(lm.pdfnor, para=para)
        return para, _cdf, _pdf


    def _wak(self, dat, l):
        ''' Fit a wakeby distribution'''
        para = lm.pelwak(lm.samlmu(dat, l))
        _cdf = partial(lm.cdfwak, para=para)
        _pdf = partial(lm.pdfwak, para=para)
        return para, _cdf, _pdf

    def _gno(self, dat, l):
        '''Fit a Generalized Normal Dist'''
        para = lm.pelgno(lm.samlmu(dat, l))
        _cdf = partial(lm.cdfgno, para=para)
        _pdf = partial(lm.pdfgno, para=para)
        return para, _cdf, _pdf

    def _pe3(self, dat, l):
        ''' Fit a Pearson type 3 distribution'''
        para = lm.pelpe3(lm.samlmu(dat, l))
        _cdf = partial(lm.cdfpe3, para=para)
        _pdf = partial(lm.pdfpe3, para=para)
        return para, _cdf, _pdf

    def _gpa(self, dat, l):
        ''' Fit a Generalised Pareto distribution'''
        para = lm.pelgpa(lm.samlmu(dat, l))
        _cdf = partial(lm.cdfgpa, para=para)
        _pdf = partial(lm.pdfgpa, para=para)
        return para, _cdf, _pdf


    def _gev(self, dat, l):
        ''' Fit a Generalised Extreme Value distribution'''
        para = lm.pelgev(lm.samlmu(dat, l))
        _cdf = partial(lm.cdfgev, para=para)
        _pdf = partial(lm.pdfgev, para=para)
        return para, _cdf, _pdf


    def _find_cdf(self, types):
        '''
        Kolmogorov-Smirnoff test that finds the best fitting CDF of the selected
        types.
        TODO: an alternative could be a max likelihood fit.
        Parameters
        -------
        types : list or None
            List of possible pdf types as in the LMoments package.
        Returns
        -------
        name : str
            name of the best fitting cdf
        cdf : partial
            CDF as from LMoments
        pdf : partial
            PDF as from LMoments
        para : list
            Lmoments cdf/pdf parameters
        '''

        # Moments for the dists
        L = {'nor' : 2, 'pe3' : 3, 'gno' : 3, 'gev':3, 'wak':5} # 'gpa': 3
        if types is None:
            types = L.keys()

        dat = self.src
        dists, done = {}, []

        if 'nor' in types: # normal dist
            name = 'nor'
            para, _cdf, _pdf = self._nor(dat, L[name])
            dists.update({name: {'cdf': _cdf, 'pdf': _pdf, 'para': para}})
            done.append(name)

        if 'pe3' in types: # pearson typ3
            name = 'pe3'
            para, _cdf, _pdf = self._pe3(dat, L[name])
            dists.update({name: {'cdf': _cdf, 'pdf': _pdf, 'para': para}})
            done.append(name)


        if 'gno' in types: # generalized normal
            name = 'gno'
            para, _cdf, _pdf = self._gno(dat, L[name])
            dists.update({name: {'cdf': _cdf, 'pdf': _pdf, 'para': para}})
            done.append(name)

        ''' # this raises Invalid Parameter error
        if 'gpa' in types: # generalized normal
            name = 'gpa'
            para, _cdf, _pdf = self._gpa(dat, L[name])
            dists.update({name: {'cdf': _cdf, 'pdf': _pdf, 'para': para}})
            done.append(name)
        '''

        if 'gev' in types:  # generalized normal
            name = 'gev'
            para, _cdf, _pdf = self._gev(dat, L[name])
            dists.update({name: {'cdf': _cdf, 'pdf': _pdf, 'para': para}})
            done.append(name)

        if 'wak' in types: # wakeby
            name = 'wak'
            para, _cdf, _pdf = self._wak(dat, L[name])
            dists.update({name: {'cdf': _cdf, 'pdf': _pdf, 'para': para}})
            done.append(name)

        if not all([d in types for d in done]):
            raise ValueError([n for n in types if n not in done], 'PDF type is not supported')


        best_fit = {'name': None, 'D': 1, 'p': 0}
        # Better fit = smaller D and larger p
        # HOM only uses the p values for detecting the best distribution (ignoring D)?
        for name, dist in dists.iteritems():
            # todo: ckeck, maybe us stats.kstest?, maybe compare the pdf so that it works without ECDF
            ecdf = ECDF(dat)
            emp = sorted(ecdf.y)
            x = sorted(dat)
            try:
                thr = dist['cdf'](x) # should
            except NameError: continue

            D, p = stats.ks_2samp(emp, thr)
            #D, p = stats.kstest(x, dist['cdf'], N=100)
            # todo: Use also the p values?
            if D < best_fit['D']:
                # D is the maximum difference in the empirical and theor. cdf
                # http://www.compbio.dundee.ac.uk/user/mgierlinski/talks/p-values1/p-values6.pdf
                best_fit['name'], best_fit['D'], best_fit['p'] = name, D, p


        name = best_fit['name']
        cdf = dists[name]['cdf']
        pdf = dists[name]['pdf']
        para = dists[name]['para']

        return name, cdf, pdf, para



    def cdf_deciles(self, i):
        '''
        Get CDF values at deciles and the category edges
        Parameters
        -------
        i : int
            Decile number (1 to 10)
        Returns
        -------
        y : float
            CDF value at the decile (quantile)
        bin : tuple
            src value for the beginning and end of the decile bin.
            Start of d1 and end of d10 will be nan
        '''
        if i not in range(1, 11):
            raise ValueError(i, 'Must be integer between 1 and 10')
        i = float(i)

        s = i - 1. if i != 1. else np.nan
        e = i + 1. if i != 10. else np.nan
        bin_start = np.nan if np.isnan(s) else self._cdf(s/10.)
        bin_end = np.nan if np.isnan(e) else self._cdf(e/10.)

        return self._cdf(i/10.), (bin_start, bin_end)

    def plot_quatiles(self):
        # plot quantiles like here https://stats.stackexchange.com/questions/132652/how-to-determine-which-distribution-fits-my-data-best
        pass


    def plot_pdf(self, plot_emp=True, name=None, xlabel='SM', style=None, ax=None):
        '''
        Plot the CDF
        '''
        if not ax:
            fig, ax = plt.subplots()

        if not style:
            color = 'blue'
            line='--'
        else:
            color = style[0]
            line = style[1:]

        if not name:
            name = ''

        # empirical CDF
        if plot_emp:
            ax.hist(self.src, color=color, density=True, alpha=0.3,
                     bins=25, label='%s empirical' % name)
        # fitted cdf (selected type)
        x = sorted(self.src)
        y = self._pdf(x)
        ax.plot(x, y, color='black', label='%s theoretical (%s)' % (name, self.name),
                linestyle=line, linewidth=2)

        ax.set_ylabel('density')
        ax.set_xlabel(xlabel if xlabel else 'x')
        ax.legend()

        return ax


    def plot_cdf(self, plot_empirical=True, name=None, xlabel='SM', style=None, ax=None):
        '''
        Plot the CDF
        '''
        if not ax:
            fig, ax = plt.subplots()

        if not style:
            color = 'blue'
            line='--'
        else:
            if isinstance(style, tuple):
                color = style[0]
                line = style[1]
            else:
                color = style[0]
                line = style[1:]

        if not name:
            name = ''

        # empirical CDF
        if plot_empirical:
            ecdf = ECDF(self.src)
            ax.scatter(ecdf.x, ecdf.y, label='%s empirical' % name, color=color,
                       alpha=0.3, marker='o', facecolors='none')

        # fitted cdf (selected type)
        x = sorted(self.src)
        y = self._cdf(x)
        if not plot_empirical:
            c = style[0]
        else:
            c = 'black'

        ax.plot(x, y, label='%s (fit %s)' % (name, self.name.upper()), color=c,
                linestyle=line, linewidth=2)



        ax.set_ylabel('cumulative probability')
        ax.set_xlabel(xlabel if xlabel else 'x')
        ax.legend(loc=4) #lower right

        return ax

def usecase():
    '''General Usecase that fits some dist to some random test data.'''
    import pandas as pd
    # Gaussian distributed
    mu, sigma = 0.5, 0.25  # mean and standard deviation
    norm_rand_data = np.random.normal(mu, sigma, 366)

    src = pd.Series(index=pd.date_range('2000-01-01', '2000-12-31', freq='D'),
                        data=norm_rand_data)

    fit = FitCDF2(src=src, types=None)

    fit.plot_cdf(plot_emp=True, name='TESTCDF', xlabel='testdata', style=('red', '--'),
                 ax=None)
    fit.plot_pdf(plot_emp=True, name='TESTPDF', xlabel='testdata', style=('green', ':'),
                 ax=None)

    fit.cdf_deciles(5)

if __name__ == '__main__':
    usecase()