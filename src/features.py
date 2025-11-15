import numpy as np


def log_returns(prices):
  """
  prices: 1D array of length n, positive
  returns: 1D array of length n
  r[t] = log(prices[t] / prices[t-1])
  r[0] = np.nan
  """
  if len(prices) == 0:
    return np.array([])

  returns = np.empty_like(prices)
  returns[0] = np.nan
  for t in range(1, len(prices)):
    returns[t] = np.log(prices[t] / prices[t - 1])

  return returns


def rolling_mean(x, window):
  """
  x: 1D array (n,)
  returns: 1D array (n,)
  out[t] = mean(x[t-window+1 : t+1])
  out[0:window-1] = np.nan
  Vectorized (use cumulative sum), no Python loop over t.
  """
  if len(x) < window:
    return np.full_like(x, np.nan)

  out = np.empty_like(x)
  cumsum = np.cumsum(np.insert(x, 0, 0))

  out[:window - 1] = np.nan
  out[window - 1:] = (cumsum[window:] - cumsum[:len(x) - window + 1]) / window
  return out


def rolling_std(x, window):
  """
  x: 1D array (n,)
  returns: 1D array (n,)
  out[t] = std(x[t-window+1 : t+1])
  out[0:window-1] = np.nan
  Vectorized (use cumulative sum), no Python loop over t.
  """
  if len(x) < window:
    return np.full_like(x, np.nan)

  out = np.empty_like(x)
  cumsum = np.cumsum(np.insert(x, 0, 0))
  cumsum2 = np.cumsum(np.insert(x**2, 0, 0))

  out[:window - 1] = np.nan
  out[window - 1:] = np.sqrt((cumsum2[window:] - cumsum2[:len(x) - window + 1]) / window -
                             ((cumsum[window:] - cumsum[:len(x) - window + 1]) / window)**2)
  return out


def zscore(x, window):
  """
  z[t] = (x[t] - rolling_mean[t]) / rolling_std[t]
  """
  if len(x) < window:
    return np.full_like(x, np.nan)

  out = (x - rolling_mean(x, window)) / rolling_std(x, window)
  return out


def ewm_mean(x, halflife):
  """
  exponential weighted mean.
  define alpha = ln(2)/halflife
  ew[t] = alpha*x[t] + (1-alpha)*ew[t-1]
  this can be done in a vectorized cumulative form or a fast loop.
  returns 1D array (n,)
  """
  alpha = np.log(2) / halflife
  ew = np.empty_like(x)
  ew[0] = x[0]
  for t in range(1, len(x)):
    ew[t] = alpha * x[t] + (1 - alpha) * ew[t - 1]

  return ew


if __name__ == "__main__":
  prices = np.array([100, 102, 101, 105, 107], dtype=float)
  returns = log_returns(prices)
  print("Log returns:", returns)

  x = np.array([1, 2, 3, 4, 5], dtype=float)
  window = 3
  rm = rolling_mean(x, window)
  print(f"Rolling mean (window={window}):", rm)

  rs = rolling_std(x, window)
  print(f"Rolling std (window={window}):", rs)

  zs = zscore(x, window)
  print(f"Z-score (window={window}):", zs)

  halflife = 2
  ew = ewm_mean(x, halflife)
  print(f"Exponential weighted mean (halflife={halflife}):", ew)