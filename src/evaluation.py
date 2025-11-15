import numpy as np


def mse(y_true, y_pred):
  """ Mean squared error """
  return np.mean((y_true- y_pred)**2)


def r2_score(y_true, y_pred):
  """ 1 - SSE/SST """
  SSE = np.sum((y_true - y_pred)**2)
  SST = np.sum((y_true - np.mean(y_true))**2)

  return 1 - SSE / SST
