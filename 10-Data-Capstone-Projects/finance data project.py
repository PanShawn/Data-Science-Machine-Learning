# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:02:24 2019

@author: wyxx6
"""

from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime

# load stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks:

df = pd.read_pickle('all_banks')

df.head()

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

# what is the max Close price for each bank's stock?

df.xs(key = 'Close', axis = 1, level = 'Stock Info').max()

# Create a new empty DataFrame called returns

returns = pd.DataFrame()

for tick in tickers:
    returns[tick + 'Return'] = df[tick]['Close'].pct_change()

returns.head()

# create a pairplot using seaborn of the returns dataframe

import seaborn as sns
sns.pairplot(returns[1:])

# what dates each bank stock had the best/worst single day returns?
returns.idxmin()

# which stock is classfied as the riskiest over the entire time period? which would you classfiy as the riskiest for the year 2015?

returns.std()
returns.ix['2015-01-01':'2015-12-31'].std()

# create a distplot using seaborn of the 2015 returns for Morgan Stanley

sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MSReturn'], color = 'green', bins=100)

# create a distplot using seaborn of the 2008 returns for CitiGroup

sns.distplot(returns.ix['2008-01-01': '2008-12-31']['CReturn'], color = 'red', bins = 100)


# More Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# optional plotly method imports
import plotly
import cufflinks as cf
cf.go_offline()

# create a line plot showing Close price for each bank for the entire index of time

for tick in tickers:
    df[tick]['Close'].plot(figsize = (12, 4), label = tick)
plt.legend()

df.xs(key = 'Close', axis = 1, level = 'Stock Info').plot()

df.xs(key = 'Close', axis = 1, level = 'Stock Info').iplot()


# Moving Average
# Plot the rolling 30 day average against the Close price for BOA in 2008
BAC = df['BAC']
MS = df['MS']



plt.figure(figsize=(12,6))
BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()

# create a heatmap of the correlation between the stock Close Price

sns.heatmap(df.xs(key = 'Close', axis = 1, level = 'Stock Info').corr(), annot = True)

# use seaborn's clustermap to cluster the correlations together

sns.clustermap(df.xs(key = 'Close', axis = 1, level = 'Stock Info').corr(),annot = True)

close_corr = df.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')

# use .iplot(kind = 'candle') to create a candle plot of BOA in 2015

BAC[['Open', 'High', 'Low', 'Close']].ix['2015-01-01':'2016-01-01'].iplot(kind = 'candle')

# use. .ta_plot(study = 'sma') to create a Simple Moving Average plot for Morgan Stanley in 2015

MS['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')

# use .ta_plot(study = 'boll') to create a Bollinger Band Plot for BOA in 2015

BAC['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='boll')































