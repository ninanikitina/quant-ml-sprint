import numpy as np

def ols_normal_eq(X, y, add_intercept=True, l2=0.0):
  """
  Ordinary Least Squares via normal equations:
      beta = (X^T X + l2 * I)^(-1) X^T y

  Inputs:
      X : (n,d) design matrix
      y : (n,) target vector
      add_intercept : if True, prepend a column of 1s to X internally
      l2 : non-negative float for ridge-style stabilization (0.0 means pure OLS)

  Returns:
      beta : (d,) estimated coefficients (including intercept if used)
      y_hat : (n,) fitted values
      residuals : (n,) y - y_hat
  """
  n, d = X.shape
  assert n == y.shape[0]

  if add_intercept:
      X = np.hstack([np.ones((n, 1)), X])

  A = X.T @ X + l2 * np.identity(d + 1 if add_intercept else d, dtype=X.dtype)
  beta = np.linalg.solve(A, X.T @ y)
  y_hat = X @ beta
  residuals = y - y_hat

  return beta, y_hat, residuals


if __name__ == "__main__":
  # Simple test case
  X = np.array([[1, 2], [2, 3], [3, 4]])
  y = np.array([1, 2, 3])

  beta, y_hat, residuals = ols_normal_eq(X, y, l2=0.001)

  print("Estimated coefficients:", beta)
  print("Fitted values:", y_hat)
  print("Residuals:", residuals)