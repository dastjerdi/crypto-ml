"""
Cryptocurrency data utils.

Earliest date data available for top 10 cryptocurrencies in terms of
market-cap on coinmarketcap.com as of April 1, 2018.
	eos: 07/01/2017
	xrp: 08/04/2013
	bch: 07/23/2017
	eth: 08/07/2015
	xlm: 08/05/2014
	neo: 09/09/2016
	miota: 06/13/2017
	btc: 04/28/2013
	ada: 10/01/2017
	ltc: 04/28/2013
"""
import datetime
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
warnings.simplefilter(action='ignore', category=FutureWarning)

_MAX_UPDATE_LENGTH = 0  # used in method `print_update()`

# Top 10 cryptocurrencies in terms of market capitalization as-of April 1,
# 2018 and corresponding shorthand names from coinmarketcap.com.
# The key values should be the strings used when scraping the data from the
# web-site.
CRYPTO_NAMES = {'ripple':'xrp', 'bitcoin':'btc', 'ethereum':'eth',
                'litecoin':'ltc', 'bitcoin-cash':'bch', 'eos':'eos',
                'cardano':'ada', 'stellar':'xlm', 'neo':'neo', 'iota':'miota'}
# Reverse of `CRYPTO_NAMES` => Convert shorthand code to full name.
CRYPTO_SHORTHAND = {'xrp':'ripple', 'btc':'bitcoin', 'eth':'ethereum',
                    'ltc':'litecoin', 'bch':'bitcoin-cash', 'eos':'eos',
                    'ada':'cardano', 'xlm':'stellar', 'neo':'neo',
                    'miota':'iota'}

# Store earliest date for which data exists for cryptocurrencies passed to
# `_load_crypto_df()` function.
crypto_min_dates = {}


def _scrape_crypto (crypto, start_date=None, last_date=None):
    """Load and clean cryptocurrency data as time series from
    coinmarketcap.com.
    """
    # Load data from web-site.
    if start_date is None:
        start_date = pd.to_datetime('1/1/2011')
    # Convert to full name used by URL if given short code.
    if crypto in CRYPTO_SHORTHAND:
        crypto = CRYPTO_SHORTHAND[crypto]
    url = "https://coinmarketcap.com/currencies/{0}/historical-data/?start=" \
          "{1}&end={2}".format(crypto, start_date.strftime("%Y%m%d"),
                               time.strftime("%Y%m%d"))
    df = pd.read_html(url)[0]
    # Clean data / ensure proper dtypes.
    df['Date'] = pd.to_datetime(df['Date'])
    # Store minimum date in `crypto_min_dates` by short code.
    if crypto in CRYPTO_NAMES:
        crypto_min_dates[CRYPTO_NAMES[crypto]] = df['Date'].min()
    else:
        crypto_min_dates[crypto] = df['Date'].min()
    try:
        # Will throw "invalid type comparison" if the volume data is
        # already properly cast.
        df.loc[df['Volume']=='-', 'Volume'] = 0
    except TypeError:
        pass
    df['Volume'] = df['Volume'].astype('int64')
    # Rename columns.
    col_conversion = {'Date':'date', 'Open':'open', 'High':'high',
                      'Low':'low', 'Close':'close', 'Volume':'volume',
                      'Market Cap':'mkt_cap'}
    df.rename(columns=col_conversion, inplace=True)
    if last_date is not None:
        df = df[df['date']<=last_date]
    df.set_index('date', drop=True, inplace=True)
    return df


def scrape_crypto (crypto, start_date=None, end_date=None):
    """Returns cryptocurrency data time series."""
    if isinstance(crypto, str):
        # Single currency.
        return _scrape_crypto(crypto, start_date, end_date)

    # Multiple currencies.
    results = []
    for c in crypto:
        df = _scrape_crypto(c, start_date, end_date)
        results.append(df)
    return tuple(results)


def load_asset (x):
    """Reads DataFrame from csv file from data folder as Time Series."""
    filepath = 'data/{}.csv'.format(x)
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', drop=True, inplace=True)
    return df


def load_returns_matrix (assets, tdelta=None, start_date=None,
                         end_date=None, center=True, scale=True,
                         use_shortnames=True):
    """Returns DataFrame with rolling returns for time period in which a price
    level is available for all the assets. Loads data from data folder.

    Args:
        assets (list): Assets to include in returns matrix.
        tdelta (pd.Timedelta): Optional, defaults to daily. Rolling return
        frequency.
    """
    if tdelta is None:
        tdelta = pd.Timedelta(days=1)
    # Create time series for each asset's closing price.
    dfs = []
    for asset in assets:
        df = load_asset(asset)
        df = df[['close']]
        df.rename(columns={'close':asset}, inplace=True)
        dfs.append(df)
    # Join all the time series.
    dfout = pd.concat(dfs, axis=1, join='inner')
    dfout = dfout.pct_change(periods=1, freq=tdelta)

    # Filter to desired date range (if date restrictions provided).
    if start_date and end_date:
        dfout = dfout[(dfout.index>=start_date) & (dfout.index<=end_date)]
    elif start_date:
        dfout = dfout[dfout.index>=start_date]
    elif end_date:
        dfout = dfout[dfout.index<=end_date]

    # Drop rows with any N/As and print warning if more than 10 rows
    # dropped.
    rows_before = len(dfout.index)
    dfout.dropna(axis=0, how='any', inplace=True)
    rows_dropped = rows_before - len(dfout.index)
    if rows_dropped>10:
        sys.stderr.write('Warning: More than 10 N/A rows dropped in '
                         'load_returns_matrix.')
    # Standardize data (if desired).
    if center or scale:
        xout = preprocessing.scale(dfout, axis=0, with_mean=center,
                                   with_std=scale)
        dfout = pd.DataFrame(xout, columns=dfout.columns, index=dfout.index)
    if use_shortnames:
        dfout.rename(columns=CRYPTO_NAMES, inplace=True)
    return dfout


def print_update (msg):
    """Handles writing updates to stdout when we want to clear previous
    outputs to the console (or other stdout).
    """
    global _MAX_UPDATE_LENGTH
    _MAX_UPDATE_LENGTH = max(_MAX_UPDATE_LENGTH, len(msg))
    empty_chars = _MAX_UPDATE_LENGTH - len(msg)
    msg = '{0}{1}'.format(msg, ' '*empty_chars)
    sys.stdout.write('{}\r'.format(msg))
    sys.stdout.flush()


def save_cryptos_to_file ():
    """Script used to save cryptocurrency data to CSVs."""
    for crypto in CRYPTO_SHORTHAND.keys():
        print_update('Saving {}'.format(crypto))
        df = scrape_crypto(crypto)
        save_path = 'data/{}.csv'.format(crypto)
        df.to_csv(save_path)


def fmt_date (dt):
    """Applies same date formatting for both Python's datetime and
    numpy.datetime64 objects.
    """
    if isinstance(dt, datetime.datetime):
        return dt.strftime("%m/%d/%Y")
    elif isinstance(dt, np.datetime64):
        return pd.to_datetime(str(dt)).strftime("%m/%d/%Y")
    else:
        raise ValueError('Unhandled type: {}'.format(type(dt)))


def demo_loading_return_matrix ():
    """Example of loading return matrix for two cryptocurrencies."""
    sd = pd.to_datetime('1/1/2016')
    ed = pd.to_datetime('1/1/2018')
    df_trial = load_returns_matrix(['btc', 'eth'], start_date=sd, end_date=ed)
    print('Example return matrix:')
    print(df_trial.head())

if __name__=='__main__':
    demo_loading_return_matrix()
