import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_market, returns, split_sample
from sae import SparseAutoencoder
from allocate import tracking_qp, portfolio_returns
from baselines import lasso_select, cluster_select
from metrics import annualized_tracking_error, cum_wealth, summary

SPARSITY = 0.20
SEED = 11
OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(SEED)

    print("loading weekly NBI + biotech names...")
    px, idx_px = load_market()
    r_stocks = returns(px)
    r_index = returns(idx_px)

    X_tr, y_tr, X_te, y_te = split_sample(r_stocks, r_index)
    names = list(X_tr.columns)
    n = len(names)
    k = max(5, int(round(SPARSITY * n)))
    print(f"universe={n}  target size={k}  train={len(X_tr)}  test={len(X_te)}")

    Xtr = X_tr.values
    ytr = y_tr.values
    Xte = X_te.values
    yte = y_te.values

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xtr_z = (Xtr - mu) / sd

    sae = SparseAutoencoder(
        n_in=n,
        n_hidden=n,
        sparsity=SPARSITY,
        lambda_sparse=2e-3,
        lambda_size=1e-3,
        nu=4.0,
        seed=SEED,
    )
    print("training sparse autoencoder...")
    sae.fit(Xtr_z, epochs=100, batch=16, lr=8e-3, prune_every=25, final_keep=k, verbose=True)
    picked, scores = sae.select_stocks(Xtr_z, names, k=k)
    cols = [names.index(t) for t in picked]
    w_sae = tracking_qp(Xtr[:, cols], ytr)

    print("SAE picks:", ", ".join(picked))
    print("weights:", {t: round(float(w), 4) for t, w in zip(picked, w_sae)})

    lasso_idx, w_lasso = lasso_select(Xtr, ytr, k)
    lasso_names = [names[i] for i in lasso_idx]
    print("LASSO picks:", ", ".join(lasso_names))

    cl_idx, w_cl = cluster_select(Xtr, ytr, k)
    cl_names = [names[i] for i in cl_idx]
    print("cluster picks:", ", ".join(cl_names))

    paths = {
        "index": np.concatenate([ytr, yte]),
        "SAE": np.concatenate([
            portfolio_returns(Xtr[:, cols], w_sae),
            portfolio_returns(Xte[:, cols], w_sae),
        ]),
        "LASSO": np.concatenate([
            portfolio_returns(Xtr[:, lasso_idx], w_lasso),
            portfolio_returns(Xte[:, lasso_idx], w_lasso),
        ]),
        "cluster": np.concatenate([
            portfolio_returns(Xtr[:, cl_idx], w_cl),
            portfolio_returns(Xte[:, cl_idx], w_cl),
        ]),
    }

    dates = X_tr.index.append(X_te.index)
    split_i = len(X_tr)

    rows = []
    for name, series in paths.items():
        if name == "index":
            continue
        rows.append(summary(series[:split_i], paths["index"][:split_i], f"{name} train"))
        rows.append(summary(series[split_i:], paths["index"][split_i:], f"{name} test"))

    tab = pd.DataFrame(rows)
    print("\n" + tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT, "metrics.csv"), index=False)

    pd.Series(dict(zip(picked, w_sae))).sort_values(ascending=False).to_csv(
        os.path.join(OUT, "sae_weights.csv"), header=["weight"]
    )

    wealth = {k: cum_wealth(v) for k, v in paths.items()}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, wealth["index"], color="#2c5aa0", lw=1.8, label="^NBI")
    ax.plot(dates[:split_i], wealth["SAE"][:split_i], color="#8B4513", lw=1.6, label="SAE (in)")
    ax.plot(dates[split_i - 1 :], wealth["SAE"][split_i - 1 :], color="#c0392b", lw=1.6, label="SAE (out)")
    ax.plot(dates[:split_i], wealth["LASSO"][:split_i], color="#1e8449", lw=1.2, alpha=0.85, label="LASSO (in)")
    ax.plot(dates[split_i - 1 :], wealth["LASSO"][split_i - 1 :], color="#58d68d", lw=1.2, alpha=0.85, label="LASSO (out)")
    ax.plot(dates[:split_i], wealth["cluster"][:split_i], color="#148f77", lw=1.1, alpha=0.8, label="cluster (in)")
    ax.plot(dates[split_i - 1 :], wealth["cluster"][split_i - 1 :], color="#76d7c4", lw=1.1, alpha=0.8, label="cluster (out)")
    ax.axvline(dates[split_i - 1], color="gray", ls="--", lw=0.9)
    ax.set_title("cumulative wealth — small portfolio index tracking")
    ax.set_ylabel("growth of $1")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cumulative.png"), dpi=140)
    print(f"\nwrote figures / tables under {OUT}")

    print("\nATE train SAE={:.2%}  LASSO={:.2%}".format(
        annualized_tracking_error(paths["SAE"][:split_i], paths["index"][:split_i]),
        annualized_tracking_error(paths["LASSO"][:split_i], paths["index"][:split_i]),
    ))
    print("ATE test  SAE={:.2%}  LASSO={:.2%}".format(
        annualized_tracking_error(paths["SAE"][split_i:], paths["index"][split_i:]),
        annualized_tracking_error(paths["LASSO"][split_i:], paths["index"][split_i:]),
    ))


if __name__ == "__main__":
    main()
