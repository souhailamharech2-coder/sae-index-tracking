import numpy as np
from sklearn.linear_model import Lasso
from sklearn.cluster import AgglomerativeClustering
from allocate import tracking_qp


def lasso_select(R, y, k, alphas=None):
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if alphas is None:
        alphas = np.logspace(-5, -1, 40)

    best = None
    for a in alphas:
        model = Lasso(alpha=a, positive=True, max_iter=20000, fit_intercept=False)
        model.fit(R, y)
        coef = model.coef_
        nz = np.flatnonzero(coef > 1e-10)
        if len(nz) == 0:
            continue
        score = abs(len(nz) - k)
        te = np.mean((R[:, nz] @ (coef[nz] / coef[nz].sum()) - y) ** 2)
        cand = (score, te, nz, coef)
        if best is None or cand[:2] < best[:2]:
            best = cand

    if best is None:
        corr = np.abs(np.corrcoef(R.T, y)[:-1, -1])
        idx = np.argsort(corr)[::-1][:k]
        return idx, tracking_qp(R[:, idx], y)

    _, _, nz, coef = best
    if len(nz) > k:
        order = np.argsort(coef[nz])[::-1][:k]
        nz = nz[order]
    elif len(nz) < k:
        leftover = [i for i in range(R.shape[1]) if i not in set(nz)]
        corr = np.abs([np.corrcoef(R[:, i], y)[0, 1] for i in leftover])
        add = np.array(leftover)[np.argsort(corr)[::-1][: k - len(nz)]]
        nz = np.concatenate([nz, add])

    w = tracking_qp(R[:, nz], y)
    return nz, w


def cluster_select(R, y, k):
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    corr = np.corrcoef(R.T)
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(np.maximum(2.0 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)

    picked = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if len(members) == 1:
            picked.append(members[0])
            continue
        sub = corr[np.ix_(members, members)]
        score = sub.sum(axis=1)
        picked.append(members[int(np.argmax(score))])

    idx = np.array(picked, dtype=int)
    w = tracking_qp(R[:, idx], y)
    return idx, w
