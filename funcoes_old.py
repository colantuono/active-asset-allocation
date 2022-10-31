
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
datetime.strftime(datetime.now(), '%Y-%m-%d %A %H:%M:%S')

## FUNCTIONS
def getData(ticker="SPY", start="1994-01-01", end=datetime.strftime(datetime.now(), '%Y-%m-%d'), interval='1MO'):
    data = yf.download(tickers=ticker, interval=interval, start=start, end=end)
    data.dropna(axis=0, inplace=True)
    return data.reset_index()

def funcMomentum(data, momentum='simple'):
    data['shift1'] = data['Adj Close'].shift(1)
    data['shift3'] = data['Adj Close'].shift(3)
    data['shift6'] = data['Adj Close'].shift(6)
    data.drop(['Open','High','Low', 'Close', 'Volume'], axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)
    og = ((data['Adj Close']/data['shift1']-1)) + ((data['Adj Close']/data['shift3']-1)) + ((data['Adj Close']/data['shift6']-1).mean())
    weighted = (12*(data['Adj Close']/data['shift1']-1)) + (4*(data['Adj Close']/data['shift3']-1)) + (2*(data['Adj Close']/data['shift6']-1))
    simple = ((data['Adj Close']/data['shift6']-1))
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
    for i in stocks:
        namespace['%s_data' % str(i)] = getData(ticker=i, end=datetime.strftime(datetime.now(), '%Y-%m-%d'), interval=interval)

    ## Creating shift columns and calculating momentum
    for i in stocks:
        namespace['%s_data' % str(i)] = funcMomentum(namespace['%s_data' % str(i)], momentum)
        
    ## series lenght ## hardcoded because ETF release date won't change.
    date = vti_data[vti_data['Date'] >= '2008-01-01']
    vti_data2 = vti_data.copy()
    vti_data2 = vti_data2.iloc[-len(date):]
    bnd_data2 = bnd_data.copy()
    bnd_data2 = bnd_data2.iloc[-len(date):]

    ## Portfolio Construction
    ret = [1]
    
    for i in range(0, len(vti_data2['Date'])):
        ret.append(
        ((vti_data2['Adj Close'].iloc[i] / vti_data2['shift1'].iloc[i] - 1) * .60) + 
        ((bnd_data2['Adj Close'].iloc[i] / bnd_data2['shift1'].iloc[i] - 1) * .40)
        )
    
    ## Returns   
    cumulative_rets = list(np.cumsum(ret))
    ret.pop(0)
    cumulative_rets.pop(0)
    ret_data = {'date':vti_data2['Date'], 'ret':ret, 'Cumulative Returns':cumulative_rets}
    returns = pd.DataFrame(data=ret_data)
    return returns


def acceleratingDualMomentum(interval='1MO', stocks=["spy","scz","tip"], momentum='simple'):
    ## Getting Data
    namespace = globals()
    for i in stocks:
        namespace['%s_data' % str(i)] = getData(ticker=i, end=datetime.strftime(datetime.now(), '%Y-%m-%d'), interval=interval)

    ## Creating shift columns and calculating momentum
    for i in stocks:
        namespace['%s_data' % str(i)] = funcMomentum(namespace['%s_data' % str(i)], momentum)
        
    ## series lenght ## hardcoded because ETF release date won't change.
    date = scz_data['Date'] 
#     date = scz_data[scz_data['Date'] >= '2008-01-01']
    spy_data2 = spy_data.copy()
    spy_data2 = spy_data2.iloc[-len(date):]
    tip_data2 = tip_data.copy()
    tip_data2 = tip_data.iloc[-len(date):]
    
    ## ETF Picking
    etf = []
    ret = [1]
    close = []
    for i in range(0, len(date)):
        if ((spy_data2['momentum'].iloc[i] > scz_data['momentum'].iloc[i]) & (spy_data2['momentum'].iloc[i] > 0)):
            etf.append('SPY')
            ret.append((spy_data2['Adj Close'].iloc[i] / spy_data2['shift1'].iloc[i] - 1))
            close.append(spy_data2['Adj Close'].iloc[i])
        elif ((scz_data['momentum'].iloc[i] > spy_data2['momentum'].iloc[i]) & (scz_data['momentum'].iloc[i] > 0)):
            etf.append('SCZ')
            ret.append((scz_data['Adj Close'].iloc[i] / scz_data['shift1'].iloc[i] - 1))
            close.append(scz_data['Adj Close'].iloc[i])
        else:
            etf.append('TIP')
            ret.append((tip_data2['Adj Close'].iloc[i] / tip_data2['shift1'].iloc[i] - 1))
            close.append(tip_data2['Adj Close'].iloc[i])
   
    ## Returns   
    cumulative_rets = list(np.cumsum(ret))
    ret.pop(0)
    cumulative_rets.pop(0)
    ret_data = {'date':date, 'etf':etf, 'close':close, 'ret':ret, 'Cumulative Returns':cumulative_rets}
    returns = pd.DataFrame(data=ret_data)
    return returns

def riskFreeRate():
    interest_rate_source = 'https://fred.stlouisfed.org/data/TB3MS.txt'
    interest_rate = float(pd.read_csv(interest_rate_source, sep=' ', skiprows=11).iloc[-1:]['Unnamed: 3'] / 100 / 3)
    return interest_rate

### functions inspired by https://github.com/enexqnt/RBAA/blob/main/RBAA.ipynb
def drawdown_DF(x):
    dd_df = pd.DataFrame()
    dd_df['date'] = x['date']
    dd_df['drawdown'] = (x['Cumulative Returns']-x['Cumulative Returns'].expanding().max())/x['Cumulative Returns']
    return dd_df

def drawdown(x):
    return (x-x.expanding().max())/x

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