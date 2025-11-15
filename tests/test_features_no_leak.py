import numpy as np

from src.features import log_returns, rolling_mean, rolling_std, zscore, ewm_mean


if __name__ == "__main__":
  prices = 10 + 0.5 * np.random.randn(100)
  np.clip(prices, a_min=1e-3, a_max=None, out=prices)

  window = 5

  returns = log_returns(prices)
  rolling_means = rolling_mean(prices, window)
  rolling_stds = rolling_std(prices, window)
  zscores = zscore(prices, window)
  ewm = ewm_mean(prices, halflife=10)

  for t in range(window, 100):
    assert rolling_means[t] == rolling_mean(prices[:t + 1], window)[t]
    assert zscores[t] == zscore(prices[:t + 1], window)[t]

  print("All tests passed.")