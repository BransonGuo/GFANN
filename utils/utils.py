import numpy as np
import pandas as pd
import random
import os
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def timeSeriesSplit(n, train_size, test_size, step_size, gap_size=0, mode='sliding', check_test_size=False):
    '''
    check_test_size: make sure test_size is as specified, not smaller
    '''
    splits = []
    if mode == 'sliding' or mode == 'rolling':
        if mode == 'rolling':
            assert step_size == test_size
        for i in range(train_size, n - test_size - gap_size + step_size, step_size):
            train_idx = np.arange(i - train_size, i)
            test_idx = np.arange(i + gap_size, i + gap_size + test_size)
            if check_test_size:
                if i + gap_size + test_size > n:
                    break
            if i + gap_size >= n:
                break
            splits.append((train_idx, test_idx))
    elif mode == 'expanding':
        for i in range(train_size, n - test_size - gap_size + step_size, step_size):
            train_idx = np.arange(i)
            test_idx = np.arange(i + gap_size, i + gap_size + test_size)
            if check_test_size:
                if i + gap_size + test_size > n:
                    break
            if i + gap_size >= n:
                break
            splits.append((train_idx, test_idx))
    return splits

def winsorize_(signal, threshold=0.05, dynamic=False, window_size=126):
    pos = signal.copy()
    if dynamic:
        pos = pd.DataFrame(pos).astype(np.float64)
        upper_rolling_quantile = pos.rolling(
            window=window_size, min_periods=1
        ).quantile(1-threshold)
        lower_rolling_quantile = pos.rolling(
            window=window_size, min_periods=1
        ).quantile(threshold)

        # Create a boolean mask for values greater than the rolling quantile
        upper_mask = pos > upper_rolling_quantile
        lower_mask = pos < lower_rolling_quantile
        # Use mask instead of in-place assignment to avoid float32/float64 dtype warnings.
        pos = pos.mask(upper_mask.fillna(False), upper_rolling_quantile)
        pos = pos.mask(lower_mask.fillna(False), lower_rolling_quantile)
    else:
        pos[pos > np.quantile(pos, 1-threshold)] = np.float32(np.quantile(pos, 1-threshold))
        pos[pos < np.quantile(pos, threshold)] = np.float32(np.quantile(pos, threshold))
    return pos

def leverage_adj(pos, dynamic=False, window_size=252):
    pos = pd.DataFrame(pos).astype(np.float64)
    if dynamic:
        rolling_max = abs(pos).rolling(window=window_size, min_periods=1).max()
        rolling_max = rolling_max.replace(0, np.nan).ffill().fillna(1.0)
        pos = pos / rolling_max
    else:
        pos = pos / abs(pos).max()
    return pos

def winsorize(signal, threshold=0.05):
    pos = signal.copy()
    pos[pos > np.quantile(pos, 1-threshold)] = np.quantile(pos, 1-threshold)
    pos[pos < np.quantile(pos, threshold)] = np.quantile(pos, threshold)
    return pos

def calc_pos_from_signal(signal, mode='continuous', window_size=60, threshold=0.1):
    pos = signal.copy()
    # thresh = max(abs(np.quantile(pos, 0.1)), abs(np.quantile(pos, 0.9)))
    # pos[pos > np.quantile(pos, 0.95)] = np.quantile(pos, 0.95)
    # pos[pos < np.quantile(pos, 0.05)] = np.quantile(pos, 0.05)
    if mode == 'continuous':
        pos = pd.DataFrame(pos) / pd.DataFrame(abs(pos)).rolling(window_size, min_periods=1).max()
        # pos = pd.DataFrame(pos) / pd.DataFrame(abs(pos)).max()
    elif mode == 'discrete':
        # Convert the array to a pandas Series
        if isinstance(signal, np.ndarray):
            pos_series = pd.Series(pos.flatten())
        else:
            pos_series = signal.copy()
        # Compute the rolling 90th quantile
        upper_rolling_quantile = pos_series.rolling(window=window_size, min_periods=1).quantile(1-threshold)
        lower_rolling_quantile = pos_series.rolling(window=window_size, min_periods=1).quantile(threshold)

        # Create a boolean mask for values greater than the rolling quantile
        upper_mask = pos_series > upper_rolling_quantile
        lower_mask = pos_series < lower_rolling_quantile
        # Update the values in pos using the mask
        pos[upper_mask.fillna(False)] = 1
        pos[~upper_mask.fillna(False)] = 0
        pos[lower_mask.fillna(False)] = -1
    elif mode == 'absolute':
        pos = pd.DataFrame(pos) / pd.DataFrame(abs(pos)).max()
    # pos[pos > 0] = 1
    # pos[pos < 0] = -1
    return pos

def calibrate_signal(signal, y, window):
    pos = signal.copy()
    pos = pos - pd.DataFrame(pos).shift(0).rolling(window, min_periods=1).mean() + pd.DataFrame(y).shift(1).rolling(window, min_periods=1).mean().values
    pos = pos / pd.DataFrame(pos).shift(0).rolling(window, min_periods=1).std() * pd.DataFrame(y).shift(1).rolling(window, min_periods=1).std().values
    return pos

def calc_directional_accuracy(signal: np.ndarray, y: np.ndarray):
    prod = signal * y
    return ((prod > 0).sum() / len(prod)).item()

def calc_ret_series(pos: np.ndarray, y: np.ndarray):
    return pos * y

def calc_ret_series_from_pos(pos: np.ndarray, y: np.ndarray):
    return pos * y

def calc_IC(signal: np.ndarray, y: np.ndarray):
    return np.corrcoef(signal.T, y.T)[1, 0]

def calc_rank_IC(signal: np.ndarray, y: np.ndarray):
    signal_series = pd.Series(np.asarray(signal).reshape(-1))
    y_series = pd.Series(np.asarray(y).reshape(-1))
    valid_mask = signal_series.notna() & y_series.notna()
    signal_rank = signal_series[valid_mask].rank(method="average")
    y_rank = y_series[valid_mask].rank(method="average")
    if len(signal_rank) < 2 or signal_rank.nunique() <= 1 or y_rank.nunique() <= 1:
        return np.nan
    return np.corrcoef(signal_rank, y_rank)[1, 0]

def calc_sharpe_ratio(daily_return: np.ndarray):
    return daily_return.mean() / daily_return.std() * np.sqrt(252)

def calc_max_dd(daily_return: np.ndarray):
    #pct dd
    ret_cumsum = pd.DataFrame(daily_return).cumsum()
    ret_cummax = ret_cumsum.cummax()
    ret_daily_dd = ret_cumsum - ret_cummax
    return (ret_daily_dd/(1+ret_cummax)).min()
    # cumulative_return = (1 + pd.DataFrame(daily_return)).cumprod()
    
    # # Calculate cumulative maximum
    # cummax = cumulative_return.cummax()
    
    # # Calculate daily drawdown as a percentage of the peak
    # drawdown = (cumulative_return - cummax) / cummax
    
    # # Return the minimum value, which represents the maximum drawdown
    # return drawdown.min().values[0]


def calc_abs_max_dd(daily_return: np.ndarray):
    ret_cumsum = pd.DataFrame(daily_return).cumsum()
    ret_cummax = ret_cumsum.cummax()
    ret_daily_dd = ret_cumsum - ret_cummax
    return ret_daily_dd.min()

def calc_turnover(pos: np.ndarray):
    return pd.DataFrame(pos).diff().abs().sum() / len(pos)


def fibonacci_up_to_n(n):
    """
    Generate a list of Fibonacci numbers that are no larger than n.

    Parameters:
    - n: The upper limit for the Fibonacci numbers.

    Returns:
    - A list of Fibonacci numbers up to n.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [1]

    fib_series = [1, 1]
    while True:
        next_fib = fib_series[-1] + fib_series[-2]
        if next_fib > n:
            break
        fib_series.append(next_fib)

    return fib_series
