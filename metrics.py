import numpy as np


def annualized_tracking_error(port, bench, periods_per_year=52):
    err = np.asarray(port) - np.asarray(bench)
    return float(np.std(err, ddof=1) * np.sqrt(periods_per_year))


def cum_wealth(rets, start=1.0):
    return start * np.cumprod(1.0 + np.asarray(rets))


def summary(port, bench, label=""):
    ate = annualized_tracking_error(port, bench)
    excess = np.asarray(port) - np.asarray(bench)
    ir = excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(52)
    return {
        "label": label,
        "ATE": ate,
        "mean_excess": float(excess.mean() * 52),
        "IR": float(ir),
        "total_return": float(cum_wealth(port)[-1] - 1.0),
        "bench_return": float(cum_wealth(bench)[-1] - 1.0),
    }
