# Deep Learning of Small Portfolios for Index Tracking

I made a project that is based from Nikolay Nikolaev's writeup on linkedin on [deep learning of small portfolios for index tracking](https://www.linkedin.com/pulse/deep-learning-small-portfolios-index-tracking-nikolay-nikolaev/). The point of the project is to partially replicate the NASDAQ Biotech index (^NBI) with a sparse subset of names.

I built this this to understand and simulate the two-step setup he describes in his post. He uses a sparse autoencoder to pick stocks, then he solves a constrained quadratic program for the weights. Here's the thing, a lot of "deep portfolios" projects online stop at the network and they never show the allocation step properly. So it get's pretty confusing at times.

## Why did I even make this?

Index Tracking with a small book is intresting because you can lower the trading costs and you get less exposure, which is great! Here's the problem though, you still want the portfolio to move with the benchmark. Standard LASSO/clustering methods don't perform well or give you false data. They either overfit the in-sample tracking error or they pick names that look correlated historically and then blow up later down the line.

The SAE angle that stayed with me was, force the network to waste capacity (hard pruning combined with soft sparsity). Then you train with a heavy-tailed reconstruction loss so outliers don't dominate and ruin your test. Then only give the surviving names to a QP with budget constraints and long-only (You can play around with it and change it up a bit. But that's what worked for me).

Both selection and allocation are kept separate, this is done intentionally! The network is really bad at hard constraints, the QP is good at hard constraints.

## How it works

Pipeline:

1. **Data** data.py) — weekly closes for ^NBI and a biotech / healthcare tickers from Yahoo. Remove names with issues. Trained through 2013-12-30, rest is out of sample (same split as in the article, you can play around with it)

2. **Stock selection** sae.py) — sparse autoencoder over the cross-section of returns. Soft sparsity / size penalty during SGD, then hard prune hidden units by importance until you hit the target cardinality (~20% of the universe). Rank stocks by how strongly they connect into the surviving hidden units.

3. **Allocation** allocate.py) — minimize tracking error ||Rw - y||^2 with sum(w)=1 and w >= 0 via cvxpy/OSQP.

4. **Baselines** baselines.py) — nonnegative LASSO tuned to the same portfolio size, and hierarchical clustering with correlation distance sqrt(2(1-rho)).

5. **Run** run.py) — trains everything, prints ATE / excess return / IR, writes output/cumulative.png and CSVs.

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

## What it outputs

On my last run (64 names that survived cleaning, target size 13):

|          | ATE train | ATE test | total return (test) |

|----------|-----------|----------|---------------------|

| SAE      | 11.6%     | 13.4%    | ~43%                |

| LASSO    | 15.7%     | 15.9%    | ~46%                |

| cluster  | 11.7%     | 14.3%    | ~33%                |

| ^NBI     | —         | —        | ~26%                |

Very intresting! SAE beats the index out of sample and it even tracks tighter than LASSO. LASSO still edges it on raw OSS returns. From what I found this isn't like what the article says (SAE wins and LASSO overfits). But, it is close enough that i'd say I reproduced his numbers.

## The issues I ran into while doing the project

**Yahoo data is really bad.** Half the "biotech" tickers I first put in the list didn't exist in 2012–2016 (MRNA, CRSP, etc.) or they got delisted or acquired (ALXN, SGEN, CELG). yfinance prints a wall of warnings and you end up with a small sample than the paper's 79 NBI constituents. Obv you can get better data and use something like ninjatrader (I'm not sure if they offer the tickers that are being used in this project) (This was a small project to familize myself with yfinance but it ended in me running into a lot of issues) Fix was to prune the ticker list to names that actually have continuous history in the window.

**Ubuntu blocked system pip.** pip install failed with the PEP 668 "externally managed environment" error. Fixed with a local .venv. Also needed pyarrow for parquet caching. pandas alone wasn't enough.

**I forgot to keep yte after a refactor.** I standardized train returns for the SAE and accidentally dropped the test index series binding. Runtime NameError right after printing the picks. Dumb, but easy to miss when you're only looking at the selection printout.

**QP can zero out "selected" names.** Long-only tracking error minimization will happily set some weights to exactly 0 (DXCM did this). So you ask for 13 names and effectively hold 12. That's allowed by the constraints; if you want a hard cardinality on the weights themselves you'd need a different optimizer.

**Results don't match the paper's ATE levels.** In the article he found around 5% annualized TE. But in my test run I found around 12-13%. This happened because of two variable. The first is the data set or tickers used. He used around 70, but I ran around 60 and target size was 13. Meaning the sample size is small. Second reason is the data used, I used free data (Yahoo finance) which is known to be unreliable and you can run into multiple biases (survivorship bias and overfitting, which I ran into). This caused me to have different tickers and different surviving names. I'm using numpy SEA instead of his Matlab stack. This made me stop chasing his exact figure once the OOS behavior looked sensible.

## What I learned from this project

- Autoencoders on returns are quite useful for structure, Again I think it's because of the data that I used. But once you try to bake in budget and having no-short constraints into the network, it becomes harder. As it's mentioned in the article Two-step is the right default here.

- Soft sparsity alone is weak on noisy financial data (Again, it's because I used yahoo finance data, if you use clean data. You won't run into this problem). Use good data = Good model.

- Using a Student-t loss instead of a plain squared error helps a lot with returns. If one stock in the bunch has had a huge one-week jump or dumped a lot, normal MSE training overreacts and then the model starts chasing that move that already happened. So, Student-t treats those big spikes as a "normal" move for this data. That means training stays focused on normal patterns. I used it as a filter like ATR.

- LASSO can hug the index in sample and still be fine or even better OOS on this particular proxy universe. so, overfitting isn't automatic. You have to look at OOS wealth and TE together.

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

- Since, I used yahoo finance and the ticker list is limited meaning it is static and most of them got delisted or merged. So, there is high likelyhood that there is survivorship.

- Another thing, there are no transaction costs, no rebalancing schedule = the returns might be lower. 

- SAE is plain numpy SGD. It's alright for this size and this small project but this is NOT something i'd deploy as is. Needs cleaner Data, more testing and a better split. Something like a 70/30 split.

- The parameters SPARSITY, prune schedule, Student-t nu), I just put what was mentioned in the article. So that it would match. Meaning that it is not grid searched and not optimized.

If you change the ticker list, delete cache/nbi_weekly.parquet. 

If you'd like to add to my project or change it, go ahead!

If you have question or would like to chat, send me an email at souhailamharech2@gmail.com
