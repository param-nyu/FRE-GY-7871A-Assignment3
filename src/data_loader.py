"""Daily market data for the 2026 Iran war-risk study.

Pulls the benchmark yield, the macro factors and the credit spreads from FRED
and Yahoo Finance, aligns them on the NYSE trading calendar, and returns daily
changes in the units each variable is reported in.

Running ``python -m src.data_loader`` writes ``data/market/levels.csv`` and
``data/market/changes.csv``. Nothing under data/ is committed.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

from .config import (CHANGE_RULES, FRED_SERIES, MARKET_DIR, SAMPLE_END,
                     SAMPLE_START, YAHOO_SERIES)

# FRED serves any series as CSV without an API key. pandas_datareader wraps the
# same endpoint but adds a dependency on its own rate limiting, so the CSV is
# fetched directly and parsed here.
FRED_CSV = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id={sid}&cosd={start}&coed={end}")

USER_AGENT = "FRE-GY-7871A coursework (NYU Tandon); contact shahparam87@gmail.com"


def _fred_series(sid: str, start, end) -> pd.Series:
    """One FRED series. Holidays are published as '.' and become NaN."""
    url = FRED_CSV.format(sid=sid, start=start, end=end)
    raw = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    raw.raise_for_status()
    frame = pd.read_csv(io.StringIO(raw.content.decode("utf-8")),
                        index_col=0, parse_dates=True)
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce").rename(sid)


def _yahoo_series(yf, ticker: str, start: str, end: str,
                  attempts: int = 3) -> pd.Series:
    """Closing prices for one ticker, retried.

    Yahoo fails intermittently, and a batch download that fails takes every
    ticker in it down at once -- which shows up not as an exception but as four
    silently empty columns. Fetching one ticker at a time with retries means a
    transient failure is loud and local rather than quiet and total.
    """
    last = None
    for attempt in range(attempts):
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                              progress=False, threads=False)
            if raw is None or raw.empty:
                raise ValueError("empty frame returned")
            close = raw["Close"]
            # A single-ticker download still returns a one-column frame under a
            # MultiIndex on recent yfinance versions.
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.rename(ticker)
        except Exception as exc:                        # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{ticker}: download failed after {attempts} attempts "
                       f"({last!r})")


def download_levels(start=SAMPLE_START, end=SAMPLE_END,
                    refresh: bool = False) -> pd.DataFrame:
    """Daily levels of every series, aligned on one index.

    The index is the union of Yahoo trading days, which is the NYSE calendar.
    FRED's rate series follow the SIFMA calendar, so a handful of days exist in
    one and not the other; those become NaN and are handled by the change rules
    rather than being silently filled.
    """
    import yfinance as yf

    cache = MARKET_DIR / "levels.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    # Pull a run-up buffer so the first in-sample day has a previous close to
    # difference against.
    buffered = pd.Timestamp(start) - pd.Timedelta(days=12)

    stop = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    frames = {}
    for ticker in YAHOO_SERIES:
        frames[ticker] = _yahoo_series(yf, ticker, str(buffered.date()), stop)
    levels = pd.DataFrame(frames).sort_index()

    for sid in FRED_SERIES:
        levels[sid] = _fred_series(sid, buffered.date(), end)

    levels = levels.loc[str(buffered.date()):str(end)]
    # Drop days that are holidays everywhere (no series printed at all).
    levels = levels.dropna(how="all")
    levels.to_csv(cache)
    return levels


def compute_changes(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily changes, each variable in the units Table 2 reports it in.

    Rates and spreads are quoted in percent by FRED, so a first difference times
    100 gives basis points. Equities and the dollar index are log changes times
    100, i.e. percent. Oil and gold are dollar changes, matching the paper,
    which reports the oil response in dollars per barrel.
    """
    out = {}
    for col, rule in CHANGE_RULES.items():
        if col not in levels:
            continue
        series = levels[col]
        if rule == "bp":
            out[col] = series.diff() * 100.0
        elif rule == "pct":
            out[col] = np.log(series).diff() * 100.0
        elif rule == "usd":
            out[col] = series.diff()
        else:                                   # pragma: no cover
            raise ValueError(f"unknown change rule {rule!r} for {col}")
    changes = pd.DataFrame(out)

    # A change is only usable when both closes exist. Rows where the benchmark
    # yield is missing cannot enter the estimator at all, so they are dropped;
    # gaps in other series are left as NaN and handled pairwise downstream,
    # which keeps one illiquid series from shrinking the sample for all of them.
    return changes.dropna(subset=["DGS2"])


def load(start=SAMPLE_START, end=SAMPLE_END, refresh: bool = False
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = download_levels(start, end, refresh=refresh)
    changes = compute_changes(levels).loc[str(start):str(end)]
    changes.to_csv(MARKET_DIR / "changes.csv")
    return levels, changes


def coverage_report(levels: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Per-series completeness over the estimation sample."""
    rows = []
    names = {**FRED_SERIES, **YAHOO_SERIES}
    for col in changes.columns:
        series = changes[col]
        rows.append({
            "Series": col,
            "Description": names.get(col, ""),
            "Units": {"bp": "basis points", "pct": "percent",
                      "usd": "dollars"}[CHANGE_RULES[col]],
            "Obs": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Mean": round(float(series.mean()), 4),
            "Std dev": round(float(series.std()), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Downloading market data, {SAMPLE_START} to {SAMPLE_END} ...")
    levels, changes = load()
    print(f"  {len(levels)} calendar rows downloaded, "
          f"{len(changes)} usable trading days in sample\n")
    report = coverage_report(levels, changes)
    print(report.to_string(index=False))
    report.to_csv(MARKET_DIR / "coverage.csv", index=False)

    gaps = report[report.Missing > 0]
    if len(gaps):
        print(f"\n{len(gaps)} series have gaps (holiday calendar mismatches); "
              "these are handled pairwise, not filled.")


if __name__ == "__main__":
    main()
