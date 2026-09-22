"""How much of each variable's movement the war risk factor accounts for.

Under the identifying assumptions, the extra variance a variable shows on war
news days is entirely the war risk factor working through its loading:

    dVar(dx_j) = d_j1^2 * dsigma^2(z1)

and because the benchmark's loading is normalised to one, dsigma^2(z1) is just
the observed shift in the benchmark's own variance. So the war-induced variance
of variable j is estimated as d_j1^2 * [Var_H(dx_1) - Var_L(dx_1)].

Rigobon and Sack are careful that this yields a *lower bound* on the share, not
a point estimate: the shift in war-induced variance must be smaller than the
level of war-induced variance on H days, and the two coincide only if there is
no war news at all on L days. The columns below preserve that reading -- the
share is reported against H-day variance and against whole-sample variance, and
both are lower bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BENCHMARK


def decompose(changes: pd.DataFrame, coefficients: pd.DataFrame,
              high: pd.Series, benchmark: str = BENCHMARK,
              coef_column: str = "d21_omega3") -> pd.DataFrame:
    """Table 3: variances by regime and the share attributable to war risk."""
    flag = high.reindex(changes.index).astype(bool)

    bench = changes[benchmark].dropna()
    bench_flag = flag.reindex(bench.index).astype(bool)
    # Rigobon and Sack measure these variances as the mean squared change,
    # rather than demeaning within each regime (their footnote 12).
    d_var_bench = float((bench[bench_flag] ** 2).mean()
                        - (bench[~bench_flag] ** 2).mean())

    rows = [{
        "variable": benchmark,
        "var_L": float((bench[~bench_flag] ** 2).mean()),
        "var_H": float((bench[bench_flag] ** 2).mean()),
        "var_all": float((bench ** 2).mean()),
        "predicted_change": np.nan,     # the normalisation, not an estimate
        "pct_H": np.nan, "pct_all": np.nan, "d21": 1.0,
    }]

    for row in coefficients.itertuples():
        series = changes[row.variable].dropna()
        f = flag.reindex(series.index).astype(bool)
        var_h = float((series[f] ** 2).mean())
        var_l = float((series[~f] ** 2).mean())
        var_all = float((series ** 2).mean())

        d21 = getattr(row, coef_column)
        predicted = (d21 ** 2) * d_var_bench if np.isfinite(d21) else np.nan

        rows.append({
            "variable": row.variable,
            "var_L": var_l, "var_H": var_h, "var_all": var_all,
            "predicted_change": predicted,
            "pct_H": 100.0 * predicted / var_h if var_h else np.nan,
            "pct_all": 100.0 * predicted / var_all if var_all else np.nan,
            "d21": d21,
        })
    return pd.DataFrame(rows)


def cumulative_share(changes: pd.DataFrame, coefficients: pd.DataFrame,
                     high: pd.Series, benchmark: str = BENCHMARK,
                     coef_column: str = "d21_omega3") -> pd.DataFrame:
    """Share of the variance of the *cumulative* move over the whole window.

    Assuming daily changes are serially independent, the variance of the
    cumulative change over T days is the sum of the daily variances. The war
    risk factor contributes its induced variance on each of the H days, so its
    share of the cumulative variance is n_H * d_j1^2 * dVar(dx_1) over the
    summed daily variance. This is the paper's final column, and the number
    behind its headline claim that war risk explained 13 to 63 percent of
    cumulative movements.
    """
    flag = high.reindex(changes.index).astype(bool)
    bench = changes[benchmark].dropna()
    bf = flag.reindex(bench.index).astype(bool)
    d_var_bench = float((bench[bf] ** 2).mean() - (bench[~bf] ** 2).mean())
    n_high = int(bf.sum())

    rows = []
    for row in coefficients.itertuples():
        series = changes[row.variable].dropna()
        d21 = getattr(row, coef_column)
        total = float((series ** 2).sum())
        war = n_high * (d21 ** 2) * d_var_bench if np.isfinite(d21) else np.nan
        rows.append({
            "variable": row.variable,
            "n_days": len(series),
            "cumulative_change": float(series.sum()),
            "total_variance": total,
            "war_variance": war,
            "pct_cumulative": 100.0 * war / total if total else np.nan,
        })
    return pd.DataFrame(rows)
