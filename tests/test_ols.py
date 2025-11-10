import numpy as np

from src.models import ols_normal_eq


def test_ols():
  true_beta = np.random.randn(10)
  X = np.random.randn(15, 10)
  y = X @ true_beta + 0.001 * np.random.randn(15)

  beta, y_hat = ols_normal_eq(X, y, add_intercept=False, l2=0.0)[:2]

  assert np.linalg.norm(beta - true_beta) < 0.01, "OLS failed to recover true coefficients"
  assert np.all(X.T @ (y - y_hat) < 0.01), "Residuals are not orthogonal to design matrix"


if __name__ == "__main__":
  test_ols()
  print("All tests passed.")
