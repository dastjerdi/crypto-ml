"""
Script to aggregate and preprocess data for EDA visualizations
"""
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import calendar
import numpy as np
import pandas as pd
from scipy import stats
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def aggregate_data(fn):
    """
    Function to aggregate data from files listed in data folder.
    Uses BTC as the crypt currency. Ignoring others for now
    :param fn: Filename consisting of various data files
    :return: Data frame with extracted data
    """
    btc = pd.read_csv("./data/btc.csv")
    #Using only date and price columns
    btc = btc[['date', 'close']]
    btc.rename(columns={'close': 'BTC'}, inplace=True)
    #Litecoin
    ltc = pd.read_csv("./data/ltc.csv")
    # Using only date and price columns
    ltc = ltc[['date', 'close']]
    ltc.rename(columns={'close': 'LTC'}, inplace=True)
    p = pd.merge(btc,ltc,how='inner')
    x = pd.read_csv(fn, encoding='ISO-8859-1')
    f = x['id']
    f = "./data/" + f + ".csv"
    for i in f:
        x = pd.read_csv(i)
        p = pd.merge(p, x, how='left')
    return p


def missing_val_plot(p, ax1, ax2):
    """
    Plots missing values - observations and predictors
    :param p: DataFrame
    :param ax1: Axes
    :param ax2: Axes
    :return:
    """
    k = 1 - p.apply(lambda x: x.count(), axis=1) / p.shape[1]
    ax1.hist(k)
    ax1.set_xlabel('Missing Values among % of Predictors')
    ax1.set_ylabel('Number of observations')
    ax1.set_title('Missing Values')
    k = 1 - p.apply(lambda x: x.count(), axis=0) / p.shape[0]
    plt.hist(k)
    ax2.set_xlabel('Missing Values among % of Observations')
    ax2.set_ylabel('Number of Predictors')
    ax2.set_title('Missing Values among observations')


def impute_val(p):
    """
    Sorts DataFrame by date and imputes missing values using forward and then backward fill
    :param p:
    :return:
    """
    # Reversing the data frame (Sorting by date)
    p = p.iloc[::-1]
    p = p.reset_index(drop=True)
    # Forward Imputation
    p = p.ffill()
    # backward imputation in case the first value is missing
    p = p.bfill()
    return p

def norm_minmax(p):
    """
    Normalize the data matrix using Min-Max
    :param p: DataFrame
    :return: Normalized DataFrame
    """
    scaler = MinMaxScaler()
    c = p.columns.tolist()
    c.remove('date')
    p[c] = scaler.fit_transform(p[c])
    return p