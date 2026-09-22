"""Run the whole study and write every exhibit to outputs/."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import heteroskedasticity_iv as hiv
from . import table_generator as tg
from . import variance_decomposition as vd
from .config import (BENCHMARK, BENCHMARK_SHOCK, CORE_END, CORE_START,
                     INTERIM_DIR, L_DAY_RULE, NOT_REPLICABLE, OUTPUT_DIR,
                     REPLICATION_BENCHMARK, REPLICATION_ORDER, TABLE_ORDER)
from .data_loader import coverage_report, load
from .nlp_tagger import main as tag_main


# ---------------------------------------------------------------------------
# L-day selection
# ---------------------------------------------------------------------------
def select_low_days(regimes: pd.DataFrame, rule: str = L_DAY_RULE) -> pd.Series:
    """Which days form the low-variance comparison set.

    Rigobon and Sack pick low-variance days "as close as possible to, but not
    included in" the war news days, in an equal-sized set (their footnote 7).
    Keeping the two samples adjacent in time limits how much the *other*
    factors can shift between them, which is precisely the assumption the
    identification rests on. Using every remaining day instead buys precision
    at the cost of that adjacency, so it is offered as a robustness check.

    Returns a boolean Series: True = H, False = L, NaN = excluded entirely.
    """
    is_high = (regimes.regime_binary == "H")
    if rule == "all":
        return is_high.astype(bool)

    high_positions = np.flatnonzero(is_high.to_numpy())
    n_days = len(regimes)
    chosen: list[int] = []
    for pos in high_positions:
        # Walk outward from each H day and take the nearest day that is not
        # itself an H day and has not already been claimed.
        for offset in range(1, n_days):
            for candidate in (pos - offset, pos + offset):
                if (0 <= candidate < n_days and not is_high.iloc[candidate]
                        and candidate not in chosen):
                    chosen.append(candidate)
                    break
            else:
                continue
            break

    flag = pd.Series(np.nan, index=regimes.index, dtype="object")
    flag[is_high] = True
    flag.iloc[sorted(chosen)] = False
    return flag


def apply_regime(changes: pd.DataFrame, flag: pd.Series
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Restrict the panel to days that belong to one of the two samples."""
    usable = flag.dropna()
    common = changes.index.intersection(usable.index)
    return changes.loc[common], usable.loc[common].astype(bool)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_regimes(regimes: pd.DataFrame, changes: pd.DataFrame,
                   path=None):
    """News intensity, the regime split, and the benchmark yield beneath it."""
    path = path or OUTPUT_DIR / "figure1_regimes.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             height_ratios=[2, 1.4, 1.6])

    ax = axes[0]
    ax.fill_between(regimes.index, regimes.news_volume, color="#7a2e2e",
                    alpha=0.30, lw=0)
    ax.plot(regimes.index, regimes.news_volume, color="#7a2e2e", lw=1.1)
    high = regimes[regimes.regime_binary == "H"]
    ax.scatter(high.index, high.news_volume, s=18, color="crimson", zorder=4,
               label=f"H: war-news days (n={len(high)})")
    ax.set_ylabel("Share of world\nnews coverage (%)", fontsize=9)
    ax.set_title("Iran conflict news intensity, GDELT", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[1]
    colours = {"H_bad": "#b5271f", "H_good": "#1f6f4a", "L_baseline": "#c9c9c9"}
    for name, colour in colours.items():
        sub = regimes[regimes.regime_3 == name]
        ax.scatter(sub.index, sub.mean_direction, s=14, color=colour,
                   label=f"{name} (n={len(sub)})")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Direction score\n(+ escalation)", fontsize=9)
    ax.set_title("Three-regime split, classified from news text only",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[2]
    bench = changes[BENCHMARK].reindex(regimes.index)
    ax.bar(regimes.index, bench.cumsum(), width=1.0, color="#31507d", alpha=0.85)
    ax.set_ylabel("2-year yield,\ncumulative (bp)", fontsize=9)
    ax.set_title("Benchmark variable: cumulative change in the 2-year Treasury yield",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.2)
    ax.set_xlabel("Date")

    fig.suptitle("Figure 1. War-risk news regimes, 2026 Iran conflict",
                 fontsize=12, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_variance(table_3: pd.DataFrame, path=None):
    """Share of variance attributable to war risk, by variable."""
    path = path or OUTPUT_DIR / "figure2_variance.png"
    frame = table_3[table_3["% explained, H days"] != "--"].copy()
    frame["h"] = pd.to_numeric(frame["% explained, H days"], errors="coerce")
    frame["a"] = pd.to_numeric(frame["% explained, all days"], errors="coerce")
    frame = frame.dropna(subset=["h"]).sort_values("h")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(frame))
    ax.barh(y - 0.2, frame.h, height=0.38, color="#b5271f",
            label="On war-news (H) days")
    ax.barh(y + 0.2, frame.a, height=0.38, color="#31507d",
            label="Over all days")
    ax.set_yticks(y)
    ax.set_yticklabels(frame.Variable, fontsize=8.5)
    ax.set_xlabel("Percent of variance attributable to war risk (lower bound)")
    ax.set_title("Figure 2. Variance decomposition, 2026 Iran war risk",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _estimate(changes: pd.DataFrame, regimes: pd.DataFrame, label: str,
              benchmark: str = BENCHMARK, order: list | None = None):
    flag = select_low_days(regimes)
    panel, high = apply_regime(changes, flag)
    targets = [c for c in (order or TABLE_ORDER) if c in panel.columns]
    coefficients = hiv.estimate_all(panel, benchmark, targets, high)
    print(f"  {label}: n_H={int(high.sum())}, n_L={int((~high).sum())}, "
          f"{len(coefficients)} variables estimated")
    return panel, high, coefficients


def run() -> None:
    print("=" * 72)
    print("Assignment 3: the effects of 2026 Iran war risk on global markets")
    print("=" * 72)

    print("\n[1/5] Market data")
    levels, changes = load()
    coverage = coverage_report(levels, changes)
    coverage.to_csv(OUTPUT_DIR / "table0_data_coverage.csv", index=False)
    print(f"  {len(changes)} trading days, {len(changes.columns)} series")

    print("\n[2/5] News regimes")
    regimes_path = INTERIM_DIR / "regimes.csv"
    if not regimes_path.exists():
        tag_main()
    regimes = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    regimes = regimes.reindex(changes.index).ffill().dropna(subset=["regime_3"])
    counts = regimes.regime_3.value_counts()
    print("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    headlines = None
    scored_path = INTERIM_DIR / "headlines_scored.csv"
    if scored_path.exists():
        headlines = pd.read_csv(scored_path, parse_dates=["date"])

    print("\n[3/5] Heteroskedasticity IV")
    print("  benchmark x1 = WTI crude (see config for why not the 2-year)")
    panel, high, coefficients = _estimate(changes, regimes, "oil, full window")
    coefficients.to_csv(INTERIM_DIR / "coefficients_full.csv", index=False)

    core_mask = ((changes.index >= pd.Timestamp(CORE_START))
                 & (changes.index <= pd.Timestamp(CORE_END)))
    core_changes = changes[core_mask]
    core_regimes = regimes.reindex(core_changes.index).dropna(subset=["regime_3"])
    _, _, core_coefficients = _estimate(core_changes, core_regimes,
                                        "oil, core window")
    core_coefficients.to_csv(INTERIM_DIR / "coefficients_core.csv", index=False)

    # The paper's own specification, reported because its failure is the
    # finding that motivates the change of benchmark.
    _, _, repl_coefficients = _estimate(changes, regimes, "2-year replication",
                                        benchmark=REPLICATION_BENCHMARK,
                                        order=REPLICATION_ORDER)
    repl_coefficients.to_csv(INTERIM_DIR / "coefficients_replication.csv",
                             index=False)

    print("\n[4/5] Rank condition and the sign of the shock")
    rank_oil = tg.rank_table(coefficients, "WTI crude")
    rank_2y = tg.rank_table(repl_coefficients, "2-Year Treasury")
    rank_oil.to_csv(OUTPUT_DIR / "table3b_rank_condition.csv", index=False)
    rank_2y.to_csv(OUTPUT_DIR / "table3c_rank_condition_2y.csv", index=False)
    ratio_oil = float(coefficients.variance_ratio_x1.iloc[0])
    ratio_2y = float(repl_coefficients.variance_ratio_x1.iloc[0])
    print(f"  WTI       Var_H/Var_L = {ratio_oil:5.2f}  "
          f"{'satisfied' if ratio_oil > 1 else 'FAILS'}")
    print(f"  2-Year    Var_H/Var_L = {ratio_2y:5.2f}  "
          f"{'satisfied' if ratio_2y > 1 else 'FAILS'}   "
          f"(paper's 2003 value: 6.19)")

    sign_info = tg.benchmark_sign(changes, regimes, benchmark=BENCHMARK)
    print(f"  oil on escalation days {sign_info['mean_H_bad']:+.2f}, "
          f"de-escalation {sign_info['mean_H_good']:+.2f}, "
          f"contrast {sign_info['contrast_bad_minus_good']:+.2f} "
          f"(t={sign_info['t_stat']:.2f}, p={sign_info['p_value']:.3f})")
    sign_2y = tg.benchmark_sign(changes, regimes,
                                benchmark=REPLICATION_BENCHMARK)
    print(f"  2Y  on escalation days {sign_2y['mean_H_bad']:+.2f}bp, "
          f"de-escalation {sign_2y['mean_H_good']:+.2f}bp, "
          f"contrast {sign_2y['contrast_bad_minus_good']:+.2f}bp "
          f"(t={sign_2y['t_stat']:.2f}, p={sign_2y['p_value']:.3f})")

    sign_table = pd.DataFrame([
        {"Quantity": f"Mean {label} on escalation (H_bad) days",
         "Value": round(info["mean_H_bad"], 2), "Units": units,
         "N": info["n_H_bad"]}
        for label, info, units in (("WTI change", sign_info, "$/bbl"),
                                   ("2Y change", sign_2y, "bp"))
    ] + [
        {"Quantity": f"Mean {label} on de-escalation (H_good) days",
         "Value": round(info["mean_H_good"], 2), "Units": units,
         "N": info["n_H_good"]}
        for label, info, units in (("WTI change", sign_info, "$/bbl"),
                                   ("2Y change", sign_2y, "bp"))
    ] + [
        {"Quantity": f"Contrast (escalation minus de-escalation), {label}",
         "Value": round(info["contrast_bad_minus_good"], 2), "Units": units,
         "N": f"t={info['t_stat']:.2f}, p={info['p_value']:.3f}"}
        for label, info, units in (("WTI", sign_info, "$/bbl"),
                                   ("2Y", sign_2y, "bp"))
    ])
    sign_table.to_csv(OUTPUT_DIR / "table2a_sign_test.csv", index=False)

    print("\n[5/5] Tables and figures")
    sign = sign_info["sign"]
    t1 = tg.table1(regimes, headlines)
    t1b = tg.regime_summary(regimes)
    t2 = tg.table2(coefficients, sign)
    t2_core = tg.table2(core_coefficients, sign)
    t2_repl = tg.table2(repl_coefficients, sign_2y["sign"],
                        shock=tg.BENCHMARK_SHOCK_BP, order=REPLICATION_ORDER,
                        include_2003=True)
    t2b = tg.sign_comparison_oil(t2, coefficients)

    decomposition = vd.decompose(panel, coefficients, high)
    cumulative = vd.cumulative_share(panel, coefficients, high)
    t3 = tg.table3(decomposition, cumulative)

    order = hiv.order_condition(len(TABLE_ORDER) + 1, n_factors=3)
    pd.DataFrame([
        {"Condition": "Trading days in sample", "Value": len(changes)},
        {"Condition": "Days in H (war news)", "Value": int(high.sum())},
        {"Condition": "Days in L (comparison)", "Value": int((~high).sum())},
        {"Condition": "L-day selection rule", "Value": L_DAY_RULE},
        {"Condition": "Var_H/Var_L, WTI benchmark", "Value": round(ratio_oil, 2)},
        {"Condition": "Var_H/Var_L, 2-Year benchmark", "Value": round(ratio_2y, 2)},
        {"Condition": "2003 value for the 2-Year (Rigobon-Sack Table 3)",
         "Value": 6.19},
        {"Condition": f"Order condition N^2-N-2K (N={order['n_vars']}, K={order['n_factors']})",
         "Value": order["statistic"]},
        {"Condition": "Order condition satisfied",
         "Value": "yes" if order["satisfied"] else "no"},
    ]).to_csv(OUTPUT_DIR / "table0b_conditions.csv", index=False)

    tg.save_all({
        "table1_event_timeline": t1,
        "table1b_regime_summary": t1b,
        "table2_cross_asset": t2,
        "table2_cross_asset_core": t2_core,
        "table2c_replication_2y": t2_repl,
        "table2b_sign_comparison": t2b,
        "table3_variance_decomposition": t3,
    })
    figure_regimes(regimes, changes)
    figure_variance(t3)

    print(f"  Table 2 scaled to {BENCHMARK_SHOCK:.0f} $/bbl in WTI")
    print(f"  same direction as 2003 Iraq: "
          f"{int((t2b['Same sign'] == 'yes').sum())}/{len(t2b)} variables")
    print(f"\n  not replicable from free data: {', '.join(NOT_REPLICABLE)}")
    print(f"  exhibits written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
