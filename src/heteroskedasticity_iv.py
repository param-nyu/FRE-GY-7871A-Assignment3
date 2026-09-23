"""Rigobon-Sack heteroskedasticity-based identification of the war risk factor.

The war risk factor is unobservable, so it cannot be put on the right-hand side
of a regression. Identification instead comes from the *second* moments. Write
the reduced form for two financial variables as

    [dx1, dx2]' = D z + mu,        D's first column = [1, d21]'

with the loading of the factor on x1 normalised to unity. If, on a known set of
days H, only the variance of the war risk factor rises -- every other factor
keeping the intensity it has on the remaining days L -- then

    dOmega = Omega_H - Omega_L = dsigma^2(z1) * [[1, d21], [d21, d21^2]]

and d21 is recovered from the shift in second moments alone, without ever
quantifying the factor. See Rigobon and Sack (2003), equations (4) to (12).

Three estimators follow, each implementable as instrumental variables:

    omega_1 : instrument on dx1, signed +1 on H days and -1 on L days
    omega_2 : the same construction using dx2
    omega_3 : the two instrument sets stacked

Standard errors come from the IV asymptotics and, as the brief requires, from a
500-replication bootstrap that resamples within each regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_REPS, BOOTSTRAP_SEED


# ---------------------------------------------------------------------------
# Rank and order conditions
# ---------------------------------------------------------------------------
def rank_condition(dx1: np.ndarray, dx2: np.ndarray,
                   high: np.ndarray) -> dict:
    """Is the covariance ellipse actually rotating between the two regimes?

    The estimator divides by a difference of second moments. If the war news
    days are not genuinely more volatile, that denominator is noise and the
    estimate is meaningless however tight its standard error looks. This
    returns the pieces so the report can show the condition rather than assert
    it.
    """
    hi, lo = high.astype(bool), ~high.astype(bool)
    omega_h = np.cov(np.vstack([dx1[hi], dx2[hi]]))
    omega_l = np.cov(np.vstack([dx1[lo], dx2[lo]]))
    delta = omega_h - omega_l
    return {
        "n_H": int(hi.sum()), "n_L": int(lo.sum()),
        "var_H_x1": float(omega_h[0, 0]), "var_L_x1": float(omega_l[0, 0]),
        "var_H_x2": float(omega_h[1, 1]), "var_L_x2": float(omega_l[1, 1]),
        "cov_H": float(omega_h[0, 1]), "cov_L": float(omega_l[0, 1]),
        "d_var_x1": float(delta[0, 0]), "d_var_x2": float(delta[1, 1]),
        "d_cov": float(delta[0, 1]),
        "det_delta": float(np.linalg.det(delta)),
        "variance_ratio_x1": (float(omega_h[0, 0] / omega_l[0, 0])
                              if omega_l[0, 0] else np.nan),
    }


def order_condition(n_vars: int, n_factors: int) -> dict:
    """Rigobon's counting condition for a system estimated jointly.

    With N variables and K common factors other than the one being identified,
    the shift in the variance-covariance matrix supplies N(N+1)/2 moments while
    the unknowns number N - 1 loadings plus the factor variances. The condition
    quoted in the brief, N^2 - N - 2K > 0, is the usable form. It is reported
    for completeness: the pairwise approach used here and in the paper is
    exactly identified by construction and does not rely on it.
    """
    value = n_vars ** 2 - n_vars - 2 * n_factors
    return {"n_vars": n_vars, "n_factors": n_factors,
            "statistic": value, "satisfied": bool(value > 0)}


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def _moments(dx1: np.ndarray, dx2: np.ndarray, high: np.ndarray) -> tuple:
    hi, lo = high.astype(bool), ~high.astype(bool)
    return (dx1[hi].var(ddof=1), dx1[lo].var(ddof=1),
            dx2[hi].var(ddof=1), dx2[lo].var(ddof=1),
            np.cov(dx1[hi], dx2[hi])[0, 1],
            np.cov(dx1[lo], dx2[lo])[0, 1])


def estimate_moment(dx1, dx2, high, which: str = "omega1") -> float:
    """Closed-form d21 from the shift in second moments.

    omega1 reproduces equation (10), omega2 equation (12). They agree only if
    the identifying assumptions hold exactly, so the gap between them is
    diagnostic rather than cosmetic.
    """
    vh1, vl1, vh2, vl2, ch, cl = _moments(dx1, dx2, high)
    if which == "omega1":
        denom = vh1 - vl1
        return (ch - cl) / denom if denom else np.nan
    if which == "omega2":
        denom = ch - cl
        return (vh2 - vl2) / denom if denom else np.nan
    raise ValueError(which)


def _iv(instrument: np.ndarray, regressor: np.ndarray,
        outcome: np.ndarray) -> tuple[float, float]:
    """Just-identified IV with its asymptotic standard error."""
    num = instrument @ outcome
    den = instrument @ regressor
    if den == 0:
        return np.nan, np.nan
    beta = num / den
    resid = outcome - beta * regressor
    n = len(outcome)
    # Standard just-identified IV variance, White-robust in the score.
    s2 = float((instrument ** 2 @ resid ** 2) / n)
    var = n * s2 / (den ** 2)
    return float(beta), float(np.sqrt(var)) if var > 0 else np.nan


def build_instrument(series: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Equation (8): the variable's change on H days, its negative on L days."""
    return np.where(high.astype(bool), series, -series)


def estimate_iv(dx1, dx2, high, which: str = "omega1") -> tuple[float, float]:
    """d21 by instrumental variables, which is algebraically the moment form."""
    if which == "omega1":
        z = build_instrument(dx1, high)
        return _iv(z, dx1, dx2)
    if which == "omega2":
        z = build_instrument(dx2, high)
        return _iv(z, dx1, dx2)
    if which == "omega3":
        # Stack the two instrument sets, as the paper's omega_3 does. The
        # regression is then overidentified and estimated by two-stage least
        # squares on the stacked moment conditions.
        z = np.column_stack([build_instrument(dx1, high),
                             build_instrument(dx2, high)])
        zx = z.T @ dx1
        zy = z.T @ dx2
        w = np.linalg.pinv(z.T @ z)
        den = zx @ w @ zx
        if den == 0:
            return np.nan, np.nan
        beta = float((zx @ w @ zy) / den)
        resid = dx2 - beta * dx1
        s = z * resid[:, None]
        # Sandwich variance for the overidentified GMM estimator:
        #   Var = (x'ZWZ'x)^-1 (x'ZW S WZ'x) (x'ZWZ'x)^-1,  S = sum z z' u^2.
        # S is the raw sum here, so no further scaling by n belongs anywhere.
        meat = s.T @ s
        var = float((zx @ w @ meat @ w @ zx) / den ** 2)
        return beta, float(np.sqrt(var)) if var > 0 else np.nan
    raise ValueError(which)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap(dx1, dx2, high, which: str = "omega1",
              reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Resample within each regime, preserving the H/L split sizes.

    Resampling the pooled sample would let the regime proportions drift, and
    the regime split is the identifying variation -- so each draw resamples H
    days from H and L days from L, keeping n_H and n_L fixed.
    """
    rng = np.random.default_rng(seed)
    h_idx = np.flatnonzero(high.astype(bool))
    l_idx = np.flatnonzero(~high.astype(bool))
    if len(h_idx) < 3 or len(l_idx) < 3:
        return {"se": np.nan, "reps_ok": 0, "ci_low": np.nan, "ci_high": np.nan}

    draws = np.empty(reps)
    for r in range(reps):
        take = np.concatenate([rng.choice(h_idx, len(h_idx), replace=True),
                               rng.choice(l_idx, len(l_idx), replace=True)])
        flag = np.concatenate([np.ones(len(h_idx), bool),
                               np.zeros(len(l_idx), bool)])
        draws[r] = estimate_iv(dx1[take], dx2[take], flag, which)[0]

    good = draws[np.isfinite(draws)]
    if len(good) < 2:
        return {"se": np.nan, "reps_ok": len(good),
                "ci_low": np.nan, "ci_high": np.nan}
    return {"se": float(good.std(ddof=1)), "reps_ok": int(len(good)),
            "ci_low": float(np.percentile(good, 2.5)),
            "ci_high": float(np.percentile(good, 97.5))}


# ---------------------------------------------------------------------------
# One variable against the benchmark
# ---------------------------------------------------------------------------
def estimate_pair(changes: pd.DataFrame, benchmark: str, target: str,
                  high: pd.Series, reps: int = BOOTSTRAP_REPS) -> dict:
    """Every estimator for one (benchmark, target) pair, on their common days."""
    frame = changes[[benchmark, target]].join(high.rename("H")).dropna()
    dx1 = frame[benchmark].to_numpy(float)
    dx2 = frame[target].to_numpy(float)
    flag = frame["H"].to_numpy(bool)

    row: dict = {"variable": target, "n_obs": len(frame)}
    row.update(rank_condition(dx1, dx2, flag))
    for which in ("omega1", "omega2", "omega3"):
        beta, se = estimate_iv(dx1, dx2, flag, which)
        row[f"d21_{which}"] = beta
        row[f"se_{which}"] = se
        row[f"t_{which}"] = abs(beta / se) if se and np.isfinite(se) else np.nan
        boot = bootstrap(dx1, dx2, flag, which, reps=reps)
        row[f"boot_se_{which}"] = boot["se"]
        row[f"boot_t_{which}"] = (abs(beta / boot["se"])
                                  if boot["se"] and np.isfinite(boot["se"])
                                  else np.nan)
        row[f"boot_ci_low_{which}"] = boot["ci_low"]
        row[f"boot_ci_high_{which}"] = boot["ci_high"]
    return row


def estimate_all(changes: pd.DataFrame, benchmark: str, targets: list[str],
                 high: pd.Series, reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    return pd.DataFrame([estimate_pair(changes, benchmark, t, high, reps)
                         for t in targets if t in changes.columns])
