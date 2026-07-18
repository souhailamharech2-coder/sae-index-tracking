import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


def _relu_grad(x):
    return (x > 0).astype(float)


def student_t_nll(resid, nu=4.0, scale=1.0):
    z = resid / scale
    return 0.5 * (nu + 1.0) * np.log1p((z * z) / nu)


def student_t_grad(resid, nu=4.0, scale=1.0):
    z = resid / scale
    return ((nu + 1.0) * resid) / (nu * scale * scale + resid * resid)


class SparseAutoencoder:
    def __init__(
        self,
        n_in,
        n_hidden=None,
        sparsity=0.2,
        lambda_sparse=1e-3,
        lambda_size=5e-4,
        nu=4.0,
        seed=7,
    ):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden if n_hidden is not None else n_in
        self.sparsity = sparsity
        self.lambda_sparse = lambda_sparse
        self.lambda_size = lambda_size
        self.nu = nu

        lim_e = np.sqrt(6.0 / (n_in + self.n_hidden))
        lim_d = np.sqrt(6.0 / (self.n_hidden + n_in))
        self.W_e = rng.uniform(-lim_e, lim_e, size=(self.n_hidden, n_in))
        self.b_e = np.zeros(self.n_hidden)
        self.W_d = rng.uniform(-lim_d, lim_d, size=(n_in, self.n_hidden))
        self.b_d = np.zeros(n_in)
        self.alive = np.ones(self.n_hidden, dtype=bool)
        self.scale = 1.0

    def encode(self, X):
        z = X @ self.W_e.T + self.b_e
        h = _relu(z)
        h[:, ~self.alive] = 0.0
        return z, h

    def decode(self, h):
        return h @ self.W_d.T + self.b_d

    def forward(self, X):
        z, h = self.encode(X)
        yhat = self.decode(h)
        return z, h, yhat

    def node_importance(self, X):
        _, h = self.encode(X)
        act = np.mean(np.abs(h), axis=0)
        wnorm = np.linalg.norm(self.W_d, axis=0) * np.linalg.norm(self.W_e, axis=1)
        score = act * wnorm
        score[~self.alive] = -np.inf
        return score

    def hard_prune(self, X, keep):
        score = self.node_importance(X)
        order = np.argsort(score)[::-1]
        keep_idx = order[:keep]
        mask = np.zeros(self.n_hidden, dtype=bool)
        mask[keep_idx] = True
        self.alive = mask
        self.W_e[~mask] = 0.0
        self.b_e[~mask] = 0.0
        self.W_d[:, ~mask] = 0.0
        return keep_idx

    def fit(
        self,
        X,
        epochs=80,
        batch=32,
        lr=5e-3,
        prune_every=20,
        final_keep=None,
        verbose=False,
    ):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if final_keep is None:
            final_keep = max(1, int(round(self.sparsity * self.n_hidden)))

        resid0 = X - X.mean(axis=0, keepdims=True)
        self.scale = max(np.std(resid0), 1e-3)

        for ep in range(epochs):
            idx = np.random.permutation(n)
            total = 0.0
            for s in range(0, n, batch):
                batch_idx = idx[s : s + batch]
                xb = X[batch_idx]
                m = xb.shape[0]

                z, h, yhat = self.forward(xb)
                resid = yhat - xb

                recon = np.mean(student_t_nll(resid, self.nu, self.scale))
                mean_act = np.mean(h, axis=0)
                sparse_pen = np.sum(np.abs(mean_act))
                size_pen = np.sum((mean_act - self.sparsity) ** 2)
                loss = recon + self.lambda_sparse * sparse_pen + self.lambda_size * size_pen
                total += loss * m

                dL_dy = student_t_grad(resid, self.nu, self.scale) / m
                dW_d = dL_dy.T @ h
                db_d = dL_dy.sum(axis=0)

                dh = dL_dy @ self.W_d
                dsparse = (self.lambda_sparse / m) * np.sign(mean_act + 1e-12)
                dsize = (2.0 * self.lambda_size / m) * (mean_act - self.sparsity)
                dh = dh + dsparse + dsize
                dh[:, ~self.alive] = 0.0

                dz = dh * _relu_grad(z)
                dW_e = dz.T @ xb
                db_e = dz.sum(axis=0)

                self.W_d -= lr * dW_d
                self.b_d -= lr * db_d
                self.W_e -= lr * dW_e
                self.b_e -= lr * db_e

                self.W_e[~self.alive] = 0.0
                self.b_e[~self.alive] = 0.0
                self.W_d[:, ~self.alive] = 0.0

            if prune_every and (ep + 1) % prune_every == 0 and (ep + 1) < epochs:
                alive_n = int(self.alive.sum())
                target = max(final_keep, int(round(alive_n * 0.7)))
                if target < alive_n:
                    self.hard_prune(X, target)

            if verbose and (ep + 1) % 10 == 0:
                print(f"epoch {ep+1:3d}  loss={total/n:.5f}  alive={self.alive.sum()}")

        self.hard_prune(X, final_keep)
        return self

    def stock_scores(self, X):
        alive_idx = np.where(self.alive)[0]
        W = np.abs(self.W_e[alive_idx])
        return W.sum(axis=0)

    def select_stocks(self, X, names, k=None):
        scores = self.stock_scores(X)
        if k is None:
            k = int(self.alive.sum())
        order = np.argsort(scores)[::-1][:k]
        picked = [names[i] for i in order]
        return picked, scores
