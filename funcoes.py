import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import yfinance as yf
from pandas_datareader import data as pdr

warnings.filterwarnings("ignore")
yf.pdr_override() 
plt.style.use("fivethirtyeight")

# yesterday = datetime.now() - timedelta(1)
yesterday = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
today = datetime.strftime(datetime.now(), '%Y-%m-%d')
now = datetime.strftime(datetime.now(), '%Y-%m-%d %A %H:%M:%S')

## FUNCTIONS
def funcMomentum(data, momentum='simple'):
    ## create shifted columns
    data = pd.DataFrame(data['Adj Close'])
    data['shift1'] = data['Adj Close'].shift(1)
    data['shift3'] = data['Adj Close'].shift(3)
    data['shift6'] = data['Adj Close'].shift(6)
    data['returns'] = data['Adj Close'].pct_change()
    data.dropna(axis=0, inplace=True)
    
    ## create returns formula
    og = data['Adj Close'].pct_change() + data['Adj Close'].pct_change(3) + data['Adj Close'].pct_change(6)
    weighted = (12*data['Adj Close'].pct_change()) + (4*data['Adj Close'].pct_change(3)) + (2*data['Adj Close'].pct_change(6))
    simple = data['Adj Close'].pct_change(6)
    
    
    ## select returns formula
    if momentum == 'weighted':
        data['momentum'] = weighted # weighted 1,3,6 periods return formula
    elif momentum == 'og':
        data['momentum'] = og # original dual momentum score
    elif momentum == 'simple':
        data['momentum'] = simple # simple 6 periods return formula
    return data

def get60_40(interval='1MO', stocks=['vti','bnd'], momentum='simple'):
    ## Getting Data
    namespace = globals()

    prices = yf.download(stocks, interval=interval, start="2000-01-01", end=today)
    prices = prices['Adj Close']
    prices.dropna(axis=0, inplace=True)

    ## Creating shift columns and calculating momentum
    for i in prices.columns:
        namespace['%s_data' % str(i.lower())] = pd.DataFrame(prices[i], index=prices.index)
        namespace['%s_data' % str(i.lower())].columns = ['Adj Close']
        namespace['%s_data' % str(i.lower())].reset_index(inplace=True)
        namespace['%s_data' % str(i.lower())] = funcMomentum(namespace['%s_data' % str(i.lower())], momentum)
        
    ## Portfolio Construction
    ret = [1]
    
    for i in range(0, len(vti_data)): 
        ret.append(
        ((vti_data['Adj Close'].iloc[i] / vti_data['shift1'].iloc[i] - 1) * .60) + 
        ((bnd_data['Adj Close'].iloc[i] / bnd_data['shift1'].iloc[i] - 1) * .40)
        )
     
    ## Returns   
    cumulative_rets = list(np.cumsum(ret))
    ret.pop(0)
    cumulative_rets.pop(0)
    dates = prices.index[-len(ret):]
    ret_data = {'date':dates, 'ret':ret, 'Cumulative Returns':cumulative_rets}
    returns = pd.DataFrame(data=ret_data)
    return returns


def acceleratingDualMomentum(interval='1MO', stocks=["spy","scz","tip"], momentum='simple'):
    namespace = globals()

    prices = yf.download(stocks, interval=interval, start="2000-01-01", end=today)
    prices = prices['Adj Close']
    prices.dropna(axis=0, inplace=True)
    
    ## Creating shift columns and calculating momentum
    for i in prices.columns:
        namespace['%s_data' % str(i.lower())] = pd.DataFrame(prices[i], index=prices.index)
        namespace['%s_data' % str(i.lower())].columns = ['Adj Close']
        namespace['%s_data' % str(i.lower())].reset_index(inplace=True)
        namespace['%s_data' % str(i.lower())] = funcMomentum(namespace['%s_data' % str(i.lower())], momentum)
        
    ## ETF Picking
    etf = []
    ret = [1]
    for i in range(0, len(spy_data)):
        if ((spy_data['momentum'].iloc[i] > scz_data['momentum'].iloc[i]) & (spy_data['momentum'].iloc[i] > 0)):
            etf.append('SPY')
            ret.append((spy_data['Adj Close'].iloc[i] / spy_data['shift1'].iloc[i] - 1))
        elif ((scz_data['momentum'].iloc[i] > spy_data['momentum'].iloc[i]) & (scz_data['momentum'].iloc[i] > 0)):
            etf.append('SCZ')
            ret.append((scz_data['Adj Close'].iloc[i] / scz_data['shift1'].iloc[i] - 1))
        else:
            etf.append('TIP')
            ret.append((tip_data['Adj Close'].iloc[i] / tip_data['shift1'].iloc[i] - 1))
   
    ## Returns   
    cumulative_rets = list(np.cumsum(ret))
    ret.pop(0)
    cumulative_rets.pop(0)
    dates = prices.index[-len(ret):]
    ret_data = {'date':dates, 'etf':etf, 'ret':ret, 'Cumulative Returns':cumulative_rets}
    returns = pd.DataFrame(data=ret_data)
    return returns


def riskFreeRate():
    interest_rate_source = 'https://fred.stlouisfed.org/data/TB3MS.txt'
    interest_rate = float(pd.read_csv(interest_rate_source, sep=' ', skiprows=11).iloc[-1:]['Unnamed: 3'] / 100 / 3)
    return interest_rate


### functions inspired by https://github.com/enexqnt/RBAA/blob/main/RBAA.ipynb
def drawdown_DF(x):
    dd_df = pd.DataFrame()
    dd_df['Date'] = x['date']
    dd_df['wealth_index'] = 1000*(1+x['ret']).cumprod()
    dd_df['previous_peaks'] = dd_df['wealth_index'].cummax()
    dd_df['drawdowns'] = (dd_df['wealth_index'] - dd_df['previous_peaks'])/dd_df['previous_peaks']
    dd_df.set_index('Date', inplace=True)
    return dd_df


def sortino(hist,per='monthly'):
    if per == 'monthly':
        m = 12
    elif per == 'weekly':
        m = 52
    hist=hist.iloc[:,0]
    expected_return = (hist.iloc[-1]/hist.iloc[0])**(m/len(hist))-1
    hist=hist.pct_change().dropna()
    downside_returns = hist.loc[hist < 0]
    down_stdev = downside_returns.std()*(m**0.5)
    sortino_ratio = (expected_return-riskFreeRate())/down_stdev
    return sortino_ratio


def stats(hist,per='monthly'):
    if per == 'monthly':
        m = 12
    elif per == 'weekly':
        m = 52
    so=sortino(hist)
    std=hist.pct_change().std()[0]*(m**0.5)
    cagr=(((hist.iloc[-1]/hist.iloc[0])**(m/len(hist))-1)).values[0]
    sh=(cagr-riskFreeRate())/std
    dd=((hist-hist.expanding().max())/hist.expanding().max()).min()[0]
    return [cagr*100,std,sh,dd*100,so]

### EDHEC functions
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    POSITIVE SKEWNESS IS GOOD
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    HIGHER THAN 3 IS CONSIDERED A FAT TAIL DISTRIBUTION
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

                         
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


### NEW FUNCTIONS
def stats2(r, returns_col, per='monthly'):
    if per == 'monthly':
        m = 12
    elif per == 'weekly':
        m = 52
    k = kurtosis(r[returns_col])
    s = skewness(r[returns_col])
    dd_max = drawdown_DF(r)['drawdowns'].min()*100
    dd_date = datetime.strftime(drawdown_DF(r)['drawdowns'].idxmin(), '%Y-%m')
    anual_r = annualize_rets(r[returns_col], m)*100
    anual_v = annualize_vol(r[returns_col], m)*100
    sh_ratio = sharpe_ratio(r[returns_col], riskFreeRate(), m)
    pos_per = len(r[returns_col].loc[r[returns_col]>0]) / len(r[returns_col])*100
    semidev = semideviation(r[returns_col])*100
    var = var_gaussian(r[returns_col])*100
    cvar = cvar_historic(r[returns_col])*100
    
    return [k,s,dd_max,dd_date,anual_r,anual_v,sh_ratio,pos_per,semidev,var,cvar]
# ADD VaR measures

