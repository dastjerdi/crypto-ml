"""
Functions for
1. Preprocessing the data
2. Normalization
"""
import pandas as pd
import seaborn as sns
import calendar
import numpy as np
import pandas as pd
from scipy import stats
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from eda_utils import *

def prepare_data(fn):
    """
    Prepares data for normalization by aggregating, removing correlated features (manually done) and returning a data frame
    :param fn: Files to be aggregated
    :return: DataFrame
    """
    p = aggregate_data(fn)
    p.head()
    # Extracting date
    p['day'] = p['date'].dt.weekday_name
    # Remove Saturday and Sunday Observations
    p = p[~ p.day.isin(['Saturday', 'Sunday'])]
    p.drop(['day'], axis=1, inplace=True)
    #Imputation
    df = impute_val(p)
    #Remove correlated predictors (EDA)
    x = ['DCOILBRENTEU', 'DHOILNYH', 'NASDAQCOM', 'RU2000PR', 'NASDAQ100', 'VXOCLS', 'VXDCLS']
    df.drop(x, axis=1, inplace=True)
    # Checking consecutive values to determine if BTC has gone up (1) or not (0)
    x = df['BTC'].tolist()
    x1 = [j - x[i - 1] for i, j in enumerate(x) if i > 0]
    y = []
    for i in x1:
        if i > 0:
            y.append(1)
        else:
            y.append(0)
    # Remove first row
    df1 = df[:-1]
    df1['y'] = y
    df1.drop(columns=['date'], axis=1, inplace=True)
    df1.index = range(len(df1.index))
    return df1


def data_normalization(df,method,wd=None):
    """
    Normalizes data and returns a dataframe with normalized data
    :param df: Input data frame
    :param method: Type of normalization -
    values = min-max or rolling_window (standardization), min-max_rw (min-max within rolling window)
    :param wd: The size of rolling window if rolling_window is used
    :return:
    """
    if method == 'rolling_window':
        ndf = pd.DataFrame()
        c = df.columns.tolist()
        c.remove('y')
        #print(c)
        step = 1
        window = wd

        for j in range(0, len(c)):
            #print(c[j])
            k = []
            x = df[c[j]].tolist()
            for i in range(0, len(x), step):
                t = x[i:i + window-1]
                k.append((t[-1] - np.mean(t)) / np.std(t))
                if i + window >= len(x):
                    ndf[c[j]] = k
                    break
        ndf['y'] = df.y[window-1:].tolist()
        ndf.replace(np.nan, 0, inplace=True)

    if method == 'min-max':
        scaler = MinMaxScaler()
        c = df.columns.tolist()
        c.remove('y')
        df[c] = scaler.fit_transform(df[c])
        ndf = df.copy()
    if method == 'min-max_rw':
        c = df.columns.tolist()
        c.remove('y')
        step = 1
        window = wd
        k = []
        scaler = MinMaxScaler()
        for i in range(0, df.shape[0], step):
            t = df.loc[i:i + window-1, c]
            x = scaler.fit_transform(t)
            k.append(x[-1])
            if i + window >= df.shape[0]:
                break
        ndf = pd.DataFrame(k)
        ndf.columns = c
        ndf['y'] = df.y[window - 1:].tolist()
        ndf.replace(np.nan, 0, inplace=True)
    return ndf

def split_data(df,prop):
    """
    Splits data in the proportion specified into training and test sets
    :param df: DataFrame to be split
    :param prop: propotion of data in training set
    :return: dataframe for training and test sets
    """
    lm = int(np.floor(df.shape[0] * prop))
    tr = df[:lm]
    ts = df[lm:]
    c = tr.columns.tolist()
    c.remove('y')
    X_train = tr[c]
    X_test = ts[c]
    y_train = tr['y']
    y_test = ts['y']
    return X_train,X_test,y_train,y_test

