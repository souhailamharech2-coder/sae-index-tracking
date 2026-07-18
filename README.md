# Sparse Autoencoder Index Tracking

Small project implementing the idea from Nikolay Nikolaev's writeup on [deep learning of small portfolios for index tracking](https://www.linkedin.com/pulse/deep-learning-small-portfolios-index-tracking-nikolay-nikolaev/). The goal is partial replication of the NASDAQ Biotech index (^NBI) with a sparse subset of names, instead of holding the whole universe.

I built this to actually understand the two-step setup he describes: use a sparse autoencoder to pick stocks, then solve a constrained quadratic program for the weights. A lot of "deep portfolios" stuff online stops at the network and never shows the allocation step properly.

## Why I made it

Index tracking with a small book is interesting because you get lower trading costs and less junk exposure, but you still want the portfolio to move with the benchmark. The usual LASSO / clustering approaches either overfit the in-sample tracking error or pick names that look correlated historically and then fall apart later.

The SAE angle that stuck with me was: force the network to throw away capacity (soft sparsity + hard pruning), train with a heavy-tailed reconstruction loss so outliers don't dominate, then only hand the surviving names to a QP with long-only and budget constraints. Selection and allocation are separate on purpose. The network is bad at hard constraints; the QP is good at them.

## How it works

Pipeline is intentionally simple:

1. **Data** (`data.py`) — weekly closes for ^NBI and a biotech / healthcare universe from Yahoo. Drop names with holes. Train through 2013-12-30, rest is out of sample (same split idea as the article).
2. **Stock selection** (`sae.py`) — sparse autoencoder over the cross-section of returns. Soft sparsity / size penalty during SGD, then hard prune hidden units by importance until you hit the target cardinality (~20% of the universe). Rank stocks by how strongly they connect into the surviving hidden units.
3. **Allocation** (`allocate.py`) — minimize tracking error `||Rw - y||^2` with `sum(w)=1` and `w >= 0` via cvxpy/OSQP.
4. **Baselines** (`baselines.py`) — nonnegative LASSO tuned to the same portfolio size, and hierarchical clustering with correlation distance `sqrt(2(1-rho))`.
5. **Run** (`run.py`) — trains everything, prints ATE / excess return / IR, writes `output/cumulative.png` and CSVs.

```
returns (train) ──► SAE (select k names) ──► QP weights
                                         │
returns (test)  ─────────────────────────┴──► TE + wealth curves
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

First run downloads prices into `cache/`. Later runs reuse that parquet file unless you delete it.

## What I got

On my last run (64 names that survived cleaning, target size 13):

|          | ATE train | ATE test | total return (test) |
|----------|-----------|----------|---------------------|
| SAE      | 11.6%     | 13.4%    | ~43%                |
| LASSO    | 15.7%     | 15.9%    | ~46%                |
| cluster  | 11.7%     | 14.3%    | ~33%                |
| ^NBI     | —         | —        | ~26%                |

SAE beats the index out of sample and tracks tighter than LASSO. LASSO still edges it on raw OOS return / IR though, so this is not a clean copy of the article's "SAE wins, LASSO overfits" story. Close enough that the machinery is doing something real; not close enough that I'd claim I reproduced his numbers.

## Problems I ran into

**Yahoo data is messy.** Half the "biotech" tickers I first put in the list didn't exist in 2012–2016 (MRNA, CRSP, etc.) or got acquired / delisted (ALXN, SGEN, CELG). yfinance prints a wall of warnings and you end up with a thinner universe than the paper's 79 NBI constituents. Fix was to prune the ticker list to names that actually have continuous history in the window, require ~98% non-null, then drop remaining NaNs. Still not the true historical NBI membership — just a workable proxy.

**Ubuntu blocked system pip.** `pip install` failed with the PEP 668 "externally managed environment" error. Fixed with a local `.venv`. Also needed `pyarrow` for parquet caching; pandas alone wasn't enough.

**I forgot to keep `yte` after a refactor.** I standardized train returns for the SAE and accidentally dropped the test index series binding. Runtime `NameError` right after printing the picks. Dumb, but easy to miss when you're only looking at the selection printout.

**QP can zero out "selected" names.** Long-only tracking error minimization will happily set some weights to exactly 0 (DXCM did this). So you ask for 13 names and effectively hold 12. That's allowed by the constraints; if you want a hard cardinality on the weights themselves you'd need a different optimizer.

**Results don't match the paper's ATE levels.** He reported ~5% annualized TE. I land around 12–13%. Different universe, different surviving names, numpy SAE instead of his Matlab stack. I stopped chasing his exact figure once the OOS behavior looked sensible.

**Publishing to GitHub.** First `git commit` failed because `user.name` / `user.email` weren't set. Then I copy-pasted a remote with the literal placeholder `YOUR_USER` and got `Repository not found`. Fixed the remote to the real account and pushed. Lesson: placeholders in docs are dangerous when you're tired.

## What I learned

- Autoencoders on returns are useful for *structure*, not for *weights*. Once you try to bake budget / no-short constraints into the network, life gets worse. Two-step is the right default here.
- Soft sparsity alone is weak on noisy financial data. The hard prune step (kill hidden units by importance, keep training on what's left) is what actually forces a small model.
- Student-t reconstruction loss is a simple robustness trick that matters. A few biotech blow-ups shouldn't own the gradient.
- Lower in-sample tracking error is not the goal. LASSO can hug the index in sample and still be fine or better OOS on this particular proxy universe — so "overfit" isn't automatic. You have to look at OOS wealth and TE together.
- Data work is most of the project. Model code is short. Cleaning a investable weekly panel is where the time went.

## Layout

```
allocate.py      QP weights
baselines.py     LASSO + clustering
data.py          download / cache / split
sae.py           sparse autoencoder
metrics.py       ATE, wealth, summary stats
run.py           end-to-end backtest
requirements.txt
cache/           ignored, price parquet
output/          ignored, plots + CSVs
```

## Notes / limitations

- Universe is a static list, not point-in-time NBI constituents, so there is survivorship bias.
- No transaction costs, no rebalancing schedule beyond the single train → hold split.
- SAE is plain numpy SGD. Fine for this size; not something I'd drop into production as-is.
- Hyperparameters (`SPARSITY`, prune schedule, Student-t `nu`) were set by hand to match the spirit of the article, not grid-searched.

If you change the ticker list, delete `cache/nbi_weekly.parquet` so it rebuilds.
