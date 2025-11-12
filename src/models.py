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


def ols_gd(X, y, lr=1e-2, max_iter=10_000, tol=1e-5, standardize=True):
  """
  Linear regression via gradient descent on MSE.

  Steps:
    1. (optional) standardize columns of X to mean=0, std=1 for faster convergence
    2. initialize beta = 0
    3. repeat:
          grad = -(X^T (y - X beta)) / n
          beta <- beta - lr * grad
       until ||grad|| < tol or max_iter reached

  Returns:
      beta              : learned coefficients (de-standardized if needed)
      losses            : list of loss values per checkpoint
      grad_norm_history : list of gradient norms
  """
  assert X.shape[0] == y.shape[0]

  if standardize:
    std = X.std(axis=0)
    X = X / std

  losses, grad_norm_history = [], []

  beta = np.zeros((X.shape[1]), dtype=X.dtype)
  for _ in range(max_iter):
    grad = -(X.T @ (y - X @ beta)) / X.shape[0]
    beta -= lr * grad

    losses.append(np.linalg.norm(y - X @ beta))
    grad_norm_history.append(np.linalg.norm(grad))

    if np.linalg.norm(grad) <= tol:
      break

  if standardize:
    # De-standardize coefficients
    beta = beta / std

  return beta, losses, grad_norm_history


if __name__ == "__main__":
  # Simple test case
  X = np.array([[1, 2], [2, 3], [3, 4]])
  y = np.array([1, 2, 3])

  beta, y_hat, residuals = ols_normal_eq(X, y, l2=0.001)

  print("Estimated coefficients:", beta)
  print("Fitted values:", y_hat)
  print("Residuals:", residuals)

  X = np.array([[0.5, 1.1, 2.1], [0.5, 1.9, 3.01], [0.9, 3, 4]])
  y = np.array([1, 2, 3])

  beta, losses, grad_norm_history = ols_gd(X, y, tol=1e-6, standardize=True)
  print("Estimated coefficients:\n", beta)
  print("Estimated coefficients from closed formula:\n", ols_normal_eq(X, y, add_intercept=False, l2=0.0001)[0])