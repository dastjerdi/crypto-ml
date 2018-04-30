"""
Unit tests for crypto_utils.py.

Ensures features are created successfully by comparing to hand-calculated
figures.
"""

import unittest

import pandas as pd

import crypto_utils as cryp

DEC_ACCY = 5  # decimals to check for accuracy


class TestDesignMatrix(unittest.TestCase):

    def setUp (self):
        """Load design matrix we'll be testing.

        We purposely use different rolling windows for price and volume to
        ensure they are calculated separately in the event we do want
        different windows.
        """
        x_cryptos = ['ltc', 'xrp', 'xlm', 'eth']
        y_crypto = 'btc'
        kwargs = {'n_rolling_price':1, 'n_rolling_volume':2,
                  'x_assets':['SP500'], 'n_std_window':20}

        self.dm = cryp.DesignMatrix(x_cryptos=x_cryptos, y_crypto=y_crypto,
                                    **kwargs)

    def test_sanity_check (self):
        """Ensure the outputs from latest set of features corresponds in
        shape and form to what we expect.
        """
        X, Y = self.dm.get_data(std=True, lag_indicator=True)

        # Ensure number of rows between what we expect.
        row_bound = (800, 1000)
        actual_rows = X.shape[0]
        msg = 'Number of rows not within expected bounds.'
        self.assertTrue(row_bound[0] < actual_rows < row_bound[1], msg)

        msg = 'X and Y have different number of rows.'
        self.assertEqual(X.shape[0], Y.shape[0], msg)

        # Ensure X columns match.
        expected_x_cols = ['SP500', 'ltc_px_std', 'xrp_px_std', 'xlm_px_std',
                           'eth_px_std', 'btc_px_std', 'ltc_volume_std',
                           'xrp_volume_std', 'xlm_volume_std', 'eth_volume_std',
                           'btc_volume_std', 'lagged_others']
        actual_x_cols = X.columns.tolist()
        msg = 'Number of X columns different than expected.'
        self.assertEqual(len(actual_x_cols), len(expected_x_cols), msg)

        for col in expected_x_cols:
            msg = 'Expected column not found: {}'.format(col)
            self.assertTrue(col in actual_x_cols, msg)


    def test_load_time_series (self):
        """Ensure initial time series is created successfully by comparing
        price- and volume-change data.
        """
        self.dm._load_time_series()
        df = self.dm.df_final
        # Define expected results.
        dt_1 = pd.to_datetime('3/1/2018')
        dt_2 = pd.to_datetime('2/10/2018')

        expected = [('SP500', dt_1, -0.013324),
                    ('btc', dt_1, 0.053193),
                    ('btc_volume', dt_1, 0.05040065),
                    ('eth', dt_2, -0.026542),
                    ('eth_volume', dt_2, -0.209714)]
        for (col, idx, value) in expected:
            msg = '{0} value not what expected on {1}'.format(
                  col, cryp.fmt_date(idx))
            actual_value = df.loc[idx, col]
            self.assertAlmostEqual(value, actual_value, DEC_ACCY, msg)

        ##############################
        # Test non-crypto-asset (SPX) to ensure returns properly rolled forward.
        ##############################
        # Return on 1/13/2018 (Saturday), 1/14 (Sunday), and 1/15 (MLK day)
        # should all be equal to the return on 1/12 (Friday).
        # Next "new" return should be on 1/16 (price change from 1/12 to 1/16).
        expected_return_1 = 0.006750  # index change from 1/11 -> 1/12.
        expected_return_2 = -0.003524  # index change from 1/12 -> 1/16.
        expected_spx = [(pd.to_datetime('1/12/2018'), expected_return_1),
                        (pd.to_datetime('1/13/2018'), expected_return_1),
                        (pd.to_datetime('1/14/2018'), expected_return_1),
                        (pd.to_datetime('1/15/2018'), expected_return_1),
                        (pd.to_datetime('1/16/2018'), expected_return_2)]

        for (idx, value) in expected_spx:
            msg = 'SP500 return on {} not what expected.'.format(
                  cryp.fmt_date(idx))
            actual_value = df.loc[idx, 'SP500']
            self.assertAlmostEqual(value, actual_value, DEC_ACCY, msg)

    def test_standardize_crypto_figures (self):
        self.dm._load_time_series()
        self.dm._standardize_crypto_figures()
        df = self.dm.df_final

        # Define expected results.
        # Rolling figures for 7/1 were standardized by data from 20-day
        # window between 6/11 - 6/30 (inclusive).
        test_date = pd.to_datetime('7/1/2016')

        expected = [('btc_px_std', test_date, -0.087190233),
                    ('btc_volume_std', test_date, -0.402742414)]
        for (col, idx, value) in expected:
            msg = '{0} value not what expected on {1}'.format(
                  col, cryp.fmt_date(idx))
            actual_value = df.loc[idx, col]
            self.assertAlmostEqual(value, actual_value, DEC_ACCY, msg)
        print('\n\nX-feature names: {}'.format(self.dm.x_feature_names))

    def test_add_relative_lag_indicator (self):
        self.dm._load_time_series()
        self.dm._standardize_crypto_figures()
        self.dm._add_relative_lag_indicator()
        df = self.dm.df_final

        # Define dates where we know btc's standardized return was lower than
        # the four other cryptos.
        expected_true = ['1/9/2016', '1/12/2016', '1/25/2016']
        # Define dates where we know it was higher than the other four.
        expected_false = ['1/10/2016', '1/11/2016', '1/13/2016', '1/14/2016',
                          '1/15/2016', '1/16/2016', '1/17/2016', '1/18/2016',
                          '1/19/2016', '1/20/2016', '1/21/2016', '1/22/2016',
                          '1/23/2016', '1/24/2016']
        for dt in expected_true:
            idx = pd.to_datetime(dt)
            msg = 'Expected indicator `lagged_others` to be True but was ' \
                  'False on {}.'.format(cryp.fmt_date(idx))
            actual = df.loc[idx, 'lagged_others']
            self.assertTrue(actual, msg)

        for dt in expected_false:
            idx = pd.to_datetime(dt)
            msg = 'Expected indicator `lagged_others` to be False but was ' \
                  'True on {}.'.format(cryp.fmt_date(idx))
            actual = df.loc[idx, 'lagged_others']
            self.assertFalse(actual, msg)

    def test_Y_no_std (self):
        """Ensure y-value (bitcoin price return) at future date is what we
        expect.

        For example, the Y value for 1/12/2018 should be equal to the bitcoin
        return on 1/13.
        """
        X, Y = self.dm.get_data(std=True, y_std=False)
        expected = [(pd.to_datetime('1/12/2018'), 0.027151911),
                    (pd.to_datetime('1/13/2018'), -0.040960432),
                    (pd.to_datetime('1/14/2018'), 0.00347081),
                    (pd.to_datetime('1/15/2018'), -0.168548025)]

        for (idx, e) in expected:
            msg = 'Y value not what expected on {}'.format(cryp.fmt_date(idx))
            actual = Y[idx]
            self.assertAlmostEqual(e, actual, DEC_ACCY, msg)

    def test_Y_std (self):
        """Ensure y-value (bitcoin price return) at future date is what we
        expect.

        For example, the Y value for 1/12/2018 should be equal to the bitcoin
        return on 1/13 (standardized).
        """
        X, Y = self.dm.get_data(std=True)
        expected = [(pd.to_datetime('1/12/2018'), 0.377138),
                    (pd.to_datetime('1/13/2018'), -0.633414),
                    (pd.to_datetime('1/14/2018'), 0.025708),
                    (pd.to_datetime('1/15/2018'), -2.569766)]

        for (idx, e) in expected:
            msg = 'Y value not what expected on {}'.format(cryp.fmt_date(idx))
            actual = Y[idx]
            self.assertAlmostEqual(e, actual, DEC_ACCY, msg)


if __name__=='__main__':
    unittest.main()
