import numpy as np
import cvxpy as cp


def tracking_qp(R, y, long_only=True):
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = R.shape[1]
    w = cp.Variable(n)
    te = cp.sum_squares(R @ w - y)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Minimize(te), cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except Exception:
        prob.solve(verbose=False)
    if w.value is None:
        x = np.linalg.lstsq(R, y, rcond=None)[0]
        x = np.maximum(x, 0)
        s = x.sum()
        if s <= 0:
            x = np.ones(n) / n
        else:
            x = x / s
        return x
    out = np.asarray(w.value).reshape(-1)
    out = np.maximum(out, 0)
    out = out / out.sum()
    return out


def portfolio_returns(R, w):
    return np.asarray(R, dtype=float) @ np.asarray(w, dtype=float)
