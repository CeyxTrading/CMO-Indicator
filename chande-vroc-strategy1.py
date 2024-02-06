###################################################################
#
#  Name: Chande VROC Strategy 1
#  Description: Strategy that uses the Chande Momentum indicator, MAs, VROC and candle stick patterns
#  Author: B/O Trading Blog
#
###################################################################
import pandas as pd
import numpy as np
import pandas_ta as ta
import talib
from datetime import datetime, timedelta
import os
import requests
from backtesting import Backtest, Strategy
pd.options.mode.chained_assignment = None


#  Output directories
CACHE_DIR = "cache"
RESULTS_DIR = "results"

#  todo Get API key
TIINGO_API_KEY = os.environ['TIINGO_API_KEY']


def fetch_nasdaq100_symbols():
    try:
        file_name = f"nasdaq100_symbols.csv"
        path = os.path.join(CACHE_DIR, file_name)
        if os.path.exists(path):
            symbols_df = pd.read_csv(path)
            return symbols_df
        else:
            table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#:~:text=It%20created%20two%20indices%3A%20the,firms%2C%20and%20Mortgage%20loan%20companies.')
            symbols_df = table[4]
            symbols_df.to_csv(path)
            return symbols_df
    except Exception as e:
        print(f"Failed to fetch NASDAQ 100 symbols, error: {str(e)}")
        return None


def fetch_prices(symbol, interval, start_date_str, end_date_str):
    try:
        file_name = f"{symbol}-{interval}-{start_date_str}-{end_date_str}-prices.csv"
        path = os.path.join(CACHE_DIR, file_name)
        if os.path.exists(path):
            prices_df = pd.read_csv(path)
            prices_df['Date'] = pd.to_datetime(prices_df['Date'])
            prices_df.set_index('Date', inplace=True)
            prices_df = prices_df.loc[:, ~prices_df.columns.str.contains('^Unnamed')]
            return prices_df
        else:
            fetch_url = f"https://api.tiingo.com/iex/{symbol}/prices?startDate={start_date_str}&endDate={end_date_str}&resampleFreq={interval}&columns=date,open,high,low,close,volume&token={TIINGO_API_KEY}"
            headers = {
                'Accept': 'application/json'
            }

            #  Send GET request
            response = requests.get(fetch_url, headers=headers)
            data = response.json()

            prices_df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            prices_df.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            prices_df['Date'] = pd.to_datetime(prices_df['Date'])
            prices_df.set_index('Date', inplace=True)

            #  Cache data
            prices_df.to_csv(path)
            return prices_df
    except Exception as e:
        print(f"Failed to fetch stock data for {symbol}, error: {str(e)}")
        return None


def calculate_sma(close_series, window):
    sma_series = ta.sma(close=close_series, length=window)
    return sma_series


def compute_slope(y_values, x_values=None):
    if x_values is None:
        x_values = np.arange(len(y_values))
    m, _ = np.polyfit(x_values, y_values, 1)
    return m


def calculate_cmo(close_series, period=3):
    cmo = talib.CMO(close_series, timeperiod=period)
    return cmo


def calculate_vroc(volume_data, period=3):
    vroc = talib.ROC(volume_data, timeperiod=period)
    return vroc


def return_series(df, column_name):
    return df[column_name]


class ChandeVrocStrategy(Strategy):
    lt_sma_window = 110
    chande_period = 10
    lng_entry_chande_threshold = -10
    lng_exit_chande_threshold = 60
    vroc_period = 12
    lng_max_price = None
    last_lng_entry_price = None
    lng_sl_pct = 0.6
    trailing_sl_pct = 0.7
    period = 0

    def init(self):
        #  Add market indicators
        self.add_market_indicators()

        #  Select market indicators for strategy and plotting
        self.mkt_lt_sma = self.I(return_series, self.data.df, "Mkt_Lt_SMA", plot=False)
        self.mkt_lt_sma_slope = self.I(return_series, self.data.df, "Mkt_Lt_SMA_Slope", plot=True)
        self.mkt_lng_lt_sma_signal = self.I(return_series, self.data.df, "Mkt_Lng_Lt_SMA_Signal", plot=False)

        #  Add asset indicators
        self.add_stock_indicators()

        #  Select asset indicators for strategy and plotting
        self.lt_sma = self.I(return_series, self.data.df, "Lt_SMA", plot=True)
        self.lt_sma_slope = self.I(return_series, self.data.df, "Lt_SMA_Slope", plot=True)
        self.lng_lt_sma_signal = self.I(return_series, self.data.df, "Lng_Lt_SMA_Signal", plot=False)
        self.chande_momentum = self.I(return_series, self.data.df, "Chande_Momentum", plot=True)
        self.lng_entry_chande_momentum_signal = self.I(return_series, self.data.df, "Lng_Entry_Chande_Momentum_Signal", plot=False)
        self.lng_exit_chande_momentum_signal = self.I(return_series, self.data.df, "Lng_Entry_Chande_Momentum_Signal", plot=False)
        self.vroc = self.I(return_series, self.data.df, "VROC", plot=False)
        self.lng_vroc_signal = self.I(return_series, self.data.df, "Lng_VROC_Signal", plot=True)
        self.lt_sma_slope = self.I(return_series, self.data.df, "Lt_SMA_Slope", plot=False)
        self.lng_lt_SMA_signal = self.I(return_series, self.data.df, "Lng_Lt_SMA_Signal", plot=False)
        self.lng_candle_signal = self.I(return_series, self.data.df, "Lng_Candle_Signal", plot=False)

    def add_market_indicators(self):
        #  Calculate long-term trend
        self.data.df.reset_index(inplace=True)
        self.data.df['Mkt_Lt_SMA'] = calculate_sma(self.data.df['Mkt_Close'], self.lt_sma_window)
        self.data.df.set_index('Date', inplace=True)

        #  Calculate slope
        slope_window = 3
        self.data.df['Mkt_Lt_SMA_Slope'] = self.data.df['Mkt_Lt_SMA'].rolling(window=slope_window).apply(
            lambda y: compute_slope(y), raw=True)

        #  Long-term long signals
        self.data.df['Mkt_Lng_Lt_SMA_Signal'] = np.where(self.data.df['Mkt_Lt_SMA_Slope'] > 0, 1, 0)

    def add_stock_indicators(self):
        #  Calculate long-term trend
        self.data.df.reset_index(inplace=True)
        self.data.df['Lt_SMA'] = calculate_sma(self.data.df['Close'], self.lt_sma_window)
        self.data.df.set_index('Date', inplace=True)

        #  Calculate slope
        slope_window = 3
        self.data.df['Lt_SMA_Slope'] = self.data.df['Lt_SMA'].rolling(window=slope_window).apply(
            lambda y: compute_slope(y), raw=True)

        #  Long-term long signals
        self.data.df['Lng_Lt_SMA_Signal'] = np.where(self.data.df['Lt_SMA_Slope'] > 0, 1, 0)

        #  Calculate Chande Momentum
        self.data.df['Chande_Momentum'] = calculate_cmo(self.data.df['Close'], self.chande_period)

        self.data.df['Chande_Momentum_Slope'] = self.data.df['Chande_Momentum'].rolling(window=slope_window).apply(
            lambda y: compute_slope(y), raw=True)

        self.data.df['Lng_Entry_Chande_Momentum_Slope_Signal'] = np.where((self.data.df['Chande_Momentum_Slope'].shift(1) <= -10) &
                                                                          (self.data.df['Chande_Momentum_Slope'] > -10), 1, 0)

        #  Chande Momentum Signal with lookback period
        self.data.df['Lng_Entry_Chande_Momentum_Signal'] = np.where(
            ((self.data.df['Chande_Momentum'].shift(1) <= self.lng_entry_chande_threshold) &
             (self.data.df['Chande_Momentum'] > self.lng_entry_chande_threshold)), 1, 0)

        self.data.df['Lng_Exit_Chande_Momentum_Signal'] = np.where(
             (self.data.df['Chande_Momentum'].shift(1) <= self.lng_exit_chande_threshold) &
             (self.data.df['Chande_Momentum'] > self.lng_exit_chande_threshold), 1, 0)

        #  VROC
        self.data.df['VROC'] = calculate_vroc(self.data.df['Volume'], self.vroc_period)

        #  VROC Signal with lookback period
        self.data.df['Lng_VROC_Signal'] = np.where(
            ((self.data.df['VROC'].shift(3) <= 0) &
             (self.data.df['VROC'].shift(2) > 0)) |
            ((self.data.df['VROC'].shift(2) <= 0) &
             (self.data.df['VROC'].shift(1) > 0)) |
            ((self.data.df['VROC'].shift(1) <= 0) &
             (self.data.df['VROC'] > 0)), 1, 0)

        # Add candlestick patterns
        self.data.df['THREE_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(self.data.df['Open'], self.data.df['High'],
                                                                       self.data.df['Low'], self.data.df['Close'])
        self.data.df['THREE_LINE_STRIKE'] = talib.CDL3LINESTRIKE(self.data.df['Open'], self.data.df['High'],
                                                                 self.data.df['Low'], self.data.df['Close'])
        self.data.df['MORNING_STAR'] = talib.CDLMORNINGSTAR(self.data.df['Open'], self.data.df['High'],
                                                            self.data.df['Low'], self.data.df['Close'], penetration=0)
        self.data.df['THREE_OUTSIDE_UP'] = talib.CDL3OUTSIDE(self.data.df['Open'], self.data.df['High'],
                                                             self.data.df['Low'], self.data.df['Close'])
        self.data.df['ENGULFING'] = talib.CDLENGULFING(self.data.df['Open'], self.data.df['High'], self.data.df['Low'],
                                                       self.data.df['Close'])

        self.data.df['Lng_Candle_Signal'] = np.where(
            (self.data.df['THREE_WHITE_SOLDIERS'] == 100) |
            (self.data.df['THREE_LINE_STRIKE'] == 100) |
            (self.data.df['MORNING_STAR'] == 100) |
            (self.data.df['THREE_OUTSIDE_UP'] == 100) |
            (self.data.df['ENGULFING'] == 100), 1, 0
        )
        self.data.df.drop(['THREE_WHITE_SOLDIERS', 'THREE_LINE_STRIKE', 'MORNING_STAR', 'THREE_OUTSIDE_UP', 'ENGULFING'],
                          axis=1, inplace=True)

    def next(self):
        #  Warm-up period
        if self.period < self.lt_sma_window:
            self.period += 1
            return

        lng_entry_signals = 0
        lng_exit_signals = 0

        #  Long entry
        if self.mkt_lng_lt_sma_signal[-1] == 1:
            lng_entry_signals += 1

        if self.lng_lt_sma_signal[-1] == 1:
            lng_entry_signals += 1

        if self.lng_entry_chande_momentum_signal[-1] == 1:
            lng_entry_signals += 1

        if self.lng_vroc_signal[-1] == 1:
            lng_entry_signals += 1

        #  Uncomment to add bullish candlestick patterns
        #if self.lng_candle_signal[-1] == 1:
        #     lng_entry_signals += 1

        #  Long exit
        if self.lng_exit_chande_momentum_signal[-1] == 1:
            lng_exit_signals += 1

        #  Trailing stop
        if self.position.size > 0:
            if self.data.df['Close'][-1] > self.lng_max_price:
                self.lng_max_price = self.data.df['Close'][-1]

            if self.data.df['Close'][-1] <= self.lng_max_price * (1 - self.trailing_sl_pct/100):
                lng_exit_signals += 1

        #  Perform trades
        if self.position.size == 0 and lng_entry_signals == 4:
            current_close = self.data.df['Close'][-1]
            self.last_lng_entry_price = current_close
            self.lng_max_price = current_close
            stop_price = round(current_close * (1 - self.lng_sl_pct/100), 4)
            self.buy(sl=stop_price)

        elif self.position.size > 0 and lng_exit_signals >= 1:
            self.last_lng_entry_price = None
            self.lng_max_price = None
            self.position.close()

        self.period += 1


def run_backtest(df, cash=10000, commission=0.002):
    bt = Backtest(df, ChandeVrocStrategy, cash=cash, commission=commission)
    stats = bt.run()
    return bt, stats


def optimize_backtest(df, cash=10000, commission=0.002):
    bt = Backtest(df, ChandeVrocStrategy, cash=cash, commission=commission, trade_on_close=False, exclusive_orders=True)

    #  Optimize strategy parameters
    stats = bt.optimize(chande_period=range(5, 20, 1),
                        vroc_period=range(5, 20, 1),
                        lt_sma_window=range(50, 200, 10),
                        lng_entry_chande_threshold=range(-50, 0, 5),
                        lng_exit_chande_threshold=range(0, 100, 5),
                        trailing_sl_pct=list(np.arange(0.1, 1.0, 0.1)),
                        lng_sl_pct=list(np.arange(0.1, 1.0, 0.1)),
                        maximize='Equity Final [$]',
                        max_tries=100,
                        random_state=0,
                        return_heatmap=False)
    return bt, stats


def print_final_stats(bt_results_df, perform_optimization=False):
    #  Create averages for each performance stat
    avg_return_pc = round(bt_results_df['return_pc'].mean(), 2)
    avg_return_pc_per_month = round(avg_return_pc / 12, 2)
    avg_win_rate = round(bt_results_df['win_rate'].mean(), 2)
    avg_max_drawdown = round(bt_results_df['max_drawdown'].mean(), 2)
    avg_buy_hold_return = round(bt_results_df['buy_hold_return'].mean(), 2)

    print(f"Average return (%): {avg_return_pc}")
    print(f"Average monthly return (%): {avg_return_pc_per_month}")
    print(f"Average win rate : {avg_win_rate}")
    print(f"Average max drawdown: {avg_max_drawdown}")
    print(f"Average buy/hold return: {avg_buy_hold_return}")

    #  Optimized parameters
    if perform_optimization is True:
        chande_period = round(bt_results_df['chande_period'].mean(), 2)
        vroc_period = round(bt_results_df['vroc_period'].mean(), 2)
        lt_sma_window = round(bt_results_df['lt_sma_window'].mean(), 2)
        trailing_sl_pct = round(bt_results_df['trailing_sl_pct'].mean(), 2)
        lng_sl_pct = round(bt_results_df['lng_sl_pct'].mean(), 2)
        lng_entry_chande_threshold = round(bt_results_df['lng_entry_chande_threshold'].mean(), 2)
        lng_exit_chande_threshold = round(bt_results_df['lng_exit_chande_threshold'].mean(), 2)

        print(f"Average Chande Period: {chande_period}")
        print(f"Average VROC Period: {vroc_period}")
        print(f"Average Long-term SMA Window: {lt_sma_window}")
        print(f"Average Trailing Stop Percent: {trailing_sl_pct}")
        print(f"Average Stop Loss Percent: {lng_sl_pct}")
        print(f"Average Long Chande Entry threshold: {lng_entry_chande_threshold}")
        print(f"Average Long Chande Exit threshold: {lng_exit_chande_threshold}")


if __name__ == "__main__":
    #  Create output directories
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Set start- and end dates
    num_years = 5
    end_date_str = datetime.today().strftime('%Y-%m-%d')
    start_date_str = (datetime.today() - timedelta(days=num_years * 365)).strftime('%Y-%m-%d')

    interval = "1hour"
    market_symbol = "QQQM"
    market_prices_df = fetch_prices(market_symbol, interval, start_date_str, end_date_str)
    market_prices_df.rename(columns={'Open': 'Mkt_Open',
                                     'High': 'Mkt_High',
                                     'Low': 'Mkt_Low',
                                     'Close': 'Mkt_Close',
                                     'Volume': 'Mkt_Volume'}, inplace=True)

    # Fetch all NASDAQ 100 symbols
    nasdaq_symbols_df = fetch_nasdaq100_symbols()
    symbols = nasdaq_symbols_df['Ticker'].tolist()

    #  Iterate through symbols
    perform_optimization = False
    bt_run_results_df = pd.DataFrame()
    min_return = 100
    max_return = 0
    for symbol in symbols:
        print(f"Processing symbol {symbol}...")

        #  Fetch prices
        prices_df = fetch_prices(symbol, interval, start_date_str, end_date_str)
        if prices_df is None or len(prices_df) < 200:
            print(f"Not enough prices for {symbol}")
            continue

        #  Merge prices with market prices
        merged_df = pd.merge(prices_df, market_prices_df, left_index=True, right_index=True, how='inner')

        #  Run backtest or optimize
        cash = 10000
        commission = 0.0
        if perform_optimization is True:
            bt, stats = optimize_backtest(merged_df, cash, commission)
            bt_run_results_row = pd.DataFrame(
                {'symbol': [symbol], 'return_pc': [stats['Return [%]']], 'win_rate': [stats['Win Rate [%]']],
                 'max_drawdown': [stats['Max. Drawdown [%]']], 'buy_hold_return': [stats['Buy & Hold Return [%]']],
                 'chande_period': [stats._strategy.chande_period],
                 'vroc_period': [stats._strategy.vroc_period],
                 'lt_sma_window': [stats._strategy.lt_sma_window],
                 'trailing_sl_pct': [stats._strategy.trailing_sl_pct],
                 'lng_sl_pct': [stats._strategy.lng_sl_pct],
                 'lng_entry_chande_threshold': [stats._strategy.lng_entry_chande_threshold],
                 'lng_exit_chande_threshold': [stats._strategy.lng_exit_chande_threshold]})
        else:
            bt, stats = run_backtest(merged_df, cash, commission)
            bt_run_results_row = pd.DataFrame(
                {'symbol': [symbol], 'return_pc': [stats['Return [%]']], 'win_rate': [stats['Win Rate [%]']],
                 'max_drawdown': [stats['Max. Drawdown [%]']], 'buy_hold_return': [stats['Buy & Hold Return [%]']]})
        current_return = stats['Return [%]']
        print(f"Return for {symbol}: {round(current_return, 2)}")

        #  Plot new min/max returns
        if current_return > max_return:
            max_return = current_return
            print(f"===> New max return for {symbol}: {round(current_return, 2)}")
            bt.plot()
        if current_return < min_return:
            min_return = current_return
            print(f"===> New min return for {symbol}: {round(current_return, 2)}")
            bt.plot()

        bt_run_results_df = pd.concat([bt_run_results_df, bt_run_results_row], axis=0, ignore_index=True)

    #  Store backtest results
    bt_run_results_df = bt_run_results_df.sort_values(by=['return_pc'], ascending=[False])
    file_name = "bt_run_results_df.csv"
    path = os.path.join(RESULTS_DIR, file_name)
    bt_run_results_df.to_csv(path)

    #  Plot results
    print_final_stats(bt_run_results_df, perform_optimization)
