import os
import pandas as pd
import yfinance as yf

CACHE = os.path.join(os.path.dirname(__file__), "cache")

BIOTECH = [
    "AMGN", "GILD", "VRTX", "REGN", "BIIB", "ILMN", "INCY", "BMRN", "ALNY",
    "EXEL", "UTHR", "NBIX", "TECH", "IONS", "JAZZ", "HALO", "ACAD", "RARE",
    "SRPT", "PTCT", "LGND", "INSM", "ARWR", "MYGN", "OPK", "CLDX", "GERN",
    "IRWD", "MDGL", "MGNX", "NKTR", "NVCR", "NVAX", "PCRX", "QDEL", "RGEN",
    "TBPH", "TGTX", "XNCR", "ABBV", "MRK", "PFE", "BMY", "LLY", "JNJ", "AZN",
    "NVO", "SNY", "GSK", "TAK", "EDIT", "CDNA", "PACB", "DGX", "LH", "TMO",
    "DHR", "A", "WAT", "IQV", "CRL", "ICLR", "VEEV", "DXCM", "PODD", "ISRG",
    "IDXX", "ALGN", "BAX", "BDX", "SYK", "ZBH", "EW", "CELG", "SHPG", "MYL",
    "PRGO", "ENDP", "MNK", "AGN", "TSRO", "LOXO", "JUNO", "KITE", "BIVV",
]


def _weekly_close(tickers, start, end):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            px = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            px = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("unexpected yfinance columns")
    else:
        px = raw[["Close"]].copy()
        px.columns = [tickers[0] if isinstance(tickers, list) else tickers]

    px = px.sort_index().resample("W-FRI").last()
    return px


def load_market(start="2012-01-06", end="2016-04-19", force=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "nbi_weekly.parquet")
    if os.path.exists(path) and not force:
        panel = pd.read_parquet(path)
        index = panel["^NBI"]
        stocks = panel.drop(columns=["^NBI"])
        return stocks, index

    tickers = sorted(set(BIOTECH))
    stocks = _weekly_close(tickers, start, end)
    index = _weekly_close(["^NBI"], start, end).iloc[:, 0]
    index.name = "^NBI"

    common = stocks.index.intersection(index.index)
    stocks = stocks.loc[common]
    index = index.loc[common]

    keep = [c for c in stocks.columns if stocks[c].notna().mean() > 0.98]
    stocks = stocks[keep].dropna(axis=0, how="any")
    index = index.reindex(stocks.index).dropna()
    stocks = stocks.loc[index.index]

    panel = stocks.copy()
    panel["^NBI"] = index
    panel.to_parquet(path)
    return stocks, index


def returns(prices):
    return prices.pct_change().dropna(how="any")


def split_sample(stock_rets, index_rets, train_end="2013-12-30"):
    train_end = pd.Timestamp(train_end)
    tr = stock_rets.index <= train_end
    te = stock_rets.index > train_end
    return (
        stock_rets.loc[tr],
        index_rets.loc[tr],
        stock_rets.loc[te],
        index_rets.loc[te],
    )
