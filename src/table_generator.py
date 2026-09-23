"""Build Tables 1, 2 and 3 in the form the paper reports them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (BENCHMARK, BENCHMARK_SHOCK, CHANGE_RULES,
                     FRED_SERIES, IRAQ_2003, OUTPUT_DIR,
                     TABLE_ORDER, YAHOO_SERIES)

NAMES = {**FRED_SERIES, **YAHOO_SERIES}
UNIT_LABEL = {"bp": "bp chg", "pct": "pct chg", "usd": "$ chg"}


# ---------------------------------------------------------------------------
# The sign of the shock
# ---------------------------------------------------------------------------
def benchmark_sign(changes: pd.DataFrame, regimes: pd.DataFrame,
                   benchmark: str = BENCHMARK) -> dict:
    """Which way does the benchmark yield move when war risk rises?

    This is the one thing heteroskedasticity-based identification cannot tell
    us. The estimator recovers the *ratio* of responses, d_j1, but the war risk
    factor is only defined up to sign: "an increase in war risk" means whatever
    direction of z1 we declare it to mean. Rigobon and Sack resolved this by
    assumption, normalising to a 25bp *fall* in the two-year yield because
    flight to quality was the 2003 prior.

    The three-regime split resolves it from data instead. On escalation days --
    days our classifier reads as bad war news from the headline text, never
    from the market reaction -- the mean change in the benchmark yield reveals
    the sign of the war risk factor's effect on it.
    """
    series = changes[benchmark].dropna()
    regime = regimes.regime_3.reindex(series.index)

    bad = series[regime == "H_bad"]
    good = series[regime == "H_good"]
    low = series[regime == "L_baseline"]

    # The contrast between escalation and de-escalation days is the cleaner
    # read than escalation alone, since it differences out anything common to
    # high-news days generally.
    contrast = float(bad.mean() - good.mean()) if len(good) else float(bad.mean())
    sign = 1.0 if contrast >= 0 else -1.0

    from scipy import stats
    if len(bad) > 1 and len(good) > 1:
        t_stat, p_value = stats.ttest_ind(bad, good, equal_var=False)
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        "n_H_bad": len(bad), "n_H_good": len(good), "n_L": len(low),
        "mean_H_bad": float(bad.mean()) if len(bad) else np.nan,
        "mean_H_good": float(good.mean()) if len(good) else np.nan,
        "mean_L": float(low.mean()) if len(low) else np.nan,
        "contrast_bad_minus_good": contrast,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "sign": sign,
        "interpretation": ("war risk UP raises the 2-year yield"
                           if sign > 0 else
                           "war risk UP lowers the 2-year yield"),
    }


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
def table1(regimes: pd.DataFrame, headlines: pd.DataFrame | None = None,
           top_n: int = 25) -> pd.DataFrame:
    """Event timeline: the highest-intensity days and how each was classified."""
    frame = regimes.nlargest(top_n, "intensity").sort_index()
    rows = []
    for day, row in frame.iterrows():
        headline = ""
        if headlines is not None:
            same_day = headlines[headlines.date == day]
            if len(same_day):
                # The headline the classifier scored most strongly in either
                # direction is the most representative one for the day.
                pick = same_day.loc[same_day.direction.abs().idxmax()]
                headline = str(pick.title)[:95]
        rows.append({
            "Date": f"{day:%Y-%m-%d}",
            "Representative headline": headline,
            "News vol (%)": round(float(row.news_volume), 3),
            "Articles": int(row.n_articles),
            "Direction": round(float(row.mean_direction), 2),
            "Regime": row.regime_3,
        })
    return pd.DataFrame(rows)


def regime_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    counts = regimes.regime_3.value_counts()
    total = len(regimes)
    rows = [{"Regime": name,
             "Definition": definition,
             "Days": int(counts.get(name, 0)),
             "Share (%)": round(100 * counts.get(name, 0) / total, 1)}
            for name, definition in [
                ("H_bad", "High war news, escalation (text-classified)"),
                ("H_good", "High war news, de-escalation (text-classified)"),
                ("L_baseline", "Low war news, macro fundamentals"),
            ]]
    rows.append({"Regime": "All", "Definition": "Trading days in sample",
                 "Days": total, "Share (%)": 100.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------
def rank_table(coefficients: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    """Does each pair actually satisfy the condition the estimator needs?

    The identification divides by Var_H(x1) - Var_L(x1). If the benchmark is
    not more volatile on war news days, that denominator is negative and every
    coefficient built on it inverts. Reporting the ratio next to the estimate
    is what separates a result from an artefact.
    """
    rows = []
    for _, row in coefficients.iterrows():
        rows.append({
            "Pair (x1, x2)": f"{benchmark_name} / {NAMES.get(row['variable'], row['variable'])}",
            "Var_H(x1)": round(row["var_H_x1"], 4),
            "Var_L(x1)": round(row["var_L_x1"], 4),
            "Ratio H/L": round(row["variance_ratio_x1"], 2),
            "Rank condition": ("satisfied" if row["variance_ratio_x1"] > 1
                               else "FAILS"),
        })
    return pd.DataFrame(rows)


def table2(coefficients: pd.DataFrame, sign: float,
           shock: float = BENCHMARK_SHOCK, order: list | None = None,
           include_2003: bool = False) -> pd.DataFrame:
    """Structural responses, scaled to a benchmark move in the two-year yield.

    d_j1 is the response of variable j per one unit of the benchmark's own
    response. Multiplying by the signed benchmark shock puts every row in the
    units of "what happens when war risk rises enough to move the two-year
    yield by `shock_bp` basis points in the direction the data says it moves".

    The 2003 column is Rigobon and Sack's published omega_3 estimate, which was
    normalised to a 25bp *fall*. It is reproduced as published, not re-estimated.
    """
    scale = sign * shock
    rows = []
    for name in (order or TABLE_ORDER):
        match = coefficients[coefficients.variable == name]
        if not len(match):
            continue
        row = match.iloc[0]
        entry = {
            "Variable": NAMES.get(name, name),
            "Units": UNIT_LABEL[CHANGE_RULES[name]],
        }
        if include_2003:
            entry["2003 Iraq"] = IRAQ_2003.get(name, np.nan)
        for which, label in (("omega1", "IV w/ w1"), ("omega2", "IV w/ w2"),
                             ("omega3", "IV w/ w3")):
            entry[label] = round(row[f"d21_{which}"] * scale, 3)
            entry[f"{label} |t|"] = round(row[f"t_{which}"], 2)
        entry["Boot |t| (w3)"] = round(row["boot_t_omega3"], 2)
        entry["Boot 95% CI (w3)"] = (
            f"[{row['boot_ci_low_omega3'] * scale:.2f}, "
            f"{row['boot_ci_high_omega3'] * scale:.2f}]")
        rows.append(entry)
    return pd.DataFrame(rows)


def sign_comparison(table_2: pd.DataFrame) -> pd.DataFrame:
    """Does 2026 move the same way as 2003, variable by variable?

    The brief's central empirical claim is that oil, spreads and equities
    behave as they did in 2003 while Treasury yields do not. This puts that
    claim on the face of a table instead of leaving it to prose.
    """
    # itertuples mangles column names containing spaces and slashes, so the
    # frame is walked by label instead.
    rows = []
    for _, row in table_2.iterrows():
        iraq, iran = row["2003 Iraq"], row["IV w/ w3"]
        if not (np.isfinite(iraq) and np.isfinite(iran)):
            continue
        rows.append({
            "Variable": row["Variable"],
            "2003 Iraq": iraq,
            "2026 Iran": iran,
            "Same sign": "yes" if np.sign(iraq) == np.sign(iran) else "NO",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 3
# ---------------------------------------------------------------------------
def table3(decomposition: pd.DataFrame, cumulative: pd.DataFrame) -> pd.DataFrame:
    merged = decomposition.merge(cumulative[["variable", "pct_cumulative"]],
                                 on="variable", how="left")
    rows = []
    for row in merged.itertuples():
        rows.append({
            "Variable": NAMES.get(row.variable, row.variable),
            "Var on L days": _fmt(row.var_L),
            "Var on H days": _fmt(row.var_H),
            "Predicted change in var": ("--" if not np.isfinite(row.predicted_change)
                                        else _fmt(row.predicted_change)),
            "% explained, H days": ("--" if not np.isfinite(row.pct_H)
                                    else round(row.pct_H, 1)),
            "% explained, all days": ("--" if not np.isfinite(row.pct_all)
                                      else round(row.pct_all, 1)),
            "% of cumulative var": ("--" if not np.isfinite(row.pct_cumulative)
                                    else round(row.pct_cumulative, 1)),
        })
    return pd.DataFrame(rows)


def _fmt(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return round(value, 5 if abs(value) < 1 else 3)


def save_all(tables: dict[str, pd.DataFrame]) -> None:
    for name, frame in tables.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def sign_comparison_oil(table_2: pd.DataFrame,
                        coefficients: pd.DataFrame) -> pd.DataFrame:
    """Does 2026 move the same *direction* as 2003, variable by variable?

    The 2003 column was normalised to a 25bp fall in the two-year yield under
    an assumed flight-to-quality sign; the 2026 column is normalised to a
    $1/bbl rise in oil. The magnitudes are therefore not comparable and are not
    compared. What is comparable is the direction each asset moves when war
    risk rises, which is the brief's actual claim: oil, spreads and equities
    behave as they did in 2003, and the Treasury curve does not.
    """
    rows = []
    for _, row in table_2.iterrows():
        name = next((k for k, v in NAMES.items() if v == row["Variable"]), None)
        if name is None or name not in IRAQ_2003:
            continue
        # The published 2003 coefficients already describe the response to an
        # *increase* in war risk (the one large enough to move the two-year
        # yield down 25bp). They are directions as they stand; negating them
        # would invert the comparison.
        iraq_dir = np.sign(IRAQ_2003[name])
        iran_dir = np.sign(row["IV w/ w3"])
        if not np.isfinite(iran_dir) or iran_dir == 0:
            continue
        rows.append({
            "Variable": row["Variable"],
            "2003 direction": "up" if iraq_dir > 0 else "down",
            "2026 direction": "up" if iran_dir > 0 else "down",
            "Same sign": "yes" if iraq_dir == iran_dir else "NO",
        })
    return pd.DataFrame(rows)
