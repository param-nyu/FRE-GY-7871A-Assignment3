"""Sample definition, series map and estimator settings.

Every choice the analysis treats as fixed lives here, so the methodological
decisions documented in the report are readable in one place.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MARKET_DIR = DATA / "market"
NEWS_DIR = DATA / "news"
INTERIM_DIR = DATA / "interim"
OUTPUT_DIR = ROOT / "outputs"

for _d in (MARKET_DIR, NEWS_DIR, INTERIM_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
# February 28, 2026 is the onset of the conflict and the single highest day of
# global Iran coverage in the whole of 2026 (GDELT: 6.4% of worldwide volume).
SAMPLE_START = dt.date(2026, 2, 28)
SAMPLE_END = dt.date(2026, 9, 22)

# Rigobon and Sack estimate over ten weeks (47 business days, 17 of them war
# news days). Seven months spans escalation, the June ceasefire and the
# aftermath, which strains the assumption that non-war factors are
# homoskedastic across the two regimes. The headline results use the full
# window the brief specifies; this shorter window is the robustness check that
# matches the paper's own design.
CORE_START = dt.date(2026, 2, 28)
CORE_END = dt.date(2026, 5, 15)

# ---------------------------------------------------------------------------
# Financial variables
# ---------------------------------------------------------------------------
# x1, the normalising variable: every structural response is measured relative
# to this one, whose loading on the war risk factor is fixed at unity.
#
# Rigobon and Sack normalise on the two-year Treasury yield, which in 2003 was
# the principal war-risk transmission channel -- its variance was 6.2 times
# higher on war news days than on other days. In 2026 that is not true. The
# two-year is *less* volatile on Iran news days than on other days (a ratio of
# 0.42 to 0.78 depending on the threshold), because the front end is being
# driven by a hawkish Federal Reserve rather than by flight to quality. Since
# identification divides by Var_H - Var_L, a two-year normalisation divides by
# a negative number and inverts every sign.
#
# WTI crude is therefore the benchmark here: it satisfies the rank condition
# comfortably (2.4x to 2.9x) and is the cleanest economic channel through which
# war risk reaches global markets. The two-year specification is still
# estimated and reported, as the diagnostic that motivates the switch.
BENCHMARK = "CL=F"
REPLICATION_BENCHMARK = "DGS2"      # the paper's choice, reported as failing

FRED_SERIES = {
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "DGS3MO": "3-Month Treasury Bill",
    "T10YIE": "Break-even Inflation (10-year)",
    "BAMLC0A4CBBB": "BBB Corporate Spread (OAS)",
    "BAMLH0A0HYM2": "High-Yield Corporate Spread (OAS)",
}

YAHOO_SERIES = {
    "^GSPC": "S&P 500",
    "CL=F": "WTI Crude Oil Futures",
    "GC=F": "Gold Futures",
    "DX-Y.NYB": "U.S. Dollar Index",
}

# How each variable's daily change is computed, and the units it is reported in.
#   "bp"  : first difference of a percent-quoted rate, reported in basis points
#   "pct" : 100 x log change, reported in percent
#   "usd" : first difference of a price, reported in dollars
CHANGE_RULES = {
    "DGS2": "bp", "DGS10": "bp", "DGS3MO": "bp", "T10YIE": "bp",
    "BAMLC0A4CBBB": "bp", "BAMLH0A0HYM2": "bp",
    "^GSPC": "pct", "DX-Y.NYB": "pct",
    "CL=F": "usd", "GC=F": "usd",
}

# Order in which Table 2 and Table 3 list the variables, mirroring the paper.
TABLE_ORDER = [
    "DGS2", "DGS10", "T10YIE", "^GSPC", "BAMLC0A4CBBB", "BAMLH0A0HYM2",
    "GC=F", "DX-Y.NYB", "DGS3MO",
]
# Variable ordering for the paper-faithful specification normalised on the 2Y.
REPLICATION_ORDER = [
    "DGS10", "T10YIE", "^GSPC", "BAMLC0A4CBBB", "BAMLH0A0HYM2",
    "CL=F", "GC=F", "DX-Y.NYB", "DGS3MO",
]

# The paper additionally reports the on-the-run/off-the-run 10-year liquidity
# premium. It is built from proprietary CUSIP-level quotes and has no free
# source, so that row is absent here and the report says so rather than
# substituting a loose proxy.
NOT_REPLICABLE = ["Liquidity Premium (10-year Treasury note)"]

# ---------------------------------------------------------------------------
# Estimator settings
# ---------------------------------------------------------------------------
# Responses are scaled to a benchmark shock. Under the oil normalisation this
# is a $1/bbl move in WTI, chosen because it is round, close to the sample's
# typical daily move, and directly readable. The 2Y replication keeps the
# paper's 25bp scaling so its column is comparable to theirs.
BENCHMARK_SHOCK = 1.0        # $1/bbl in WTI
BENCHMARK_SHOCK_LABEL = "a $1/bbl rise in WTI crude"
BENCHMARK_SHOCK_BP = 25.0    # the paper's 25bp two-year scaling
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260922

# Rigobon and Sack choose low-variance days "as close as possible to, but not
# included in" the war news days, in an equal-sized set, which limits how much
# the other factors can shift between the two samples (their footnote 7).
L_DAY_RULE = "nearest"       # "nearest" (paper) or "all" (every non-H day)

# ---------------------------------------------------------------------------
# 2003 Iraq benchmark: Rigobon and Sack (2003), Table 2, the omega_3 column.
# Reproduced here as published; the 2003 estimates are NOT re-estimated.
# Their normalisation is an increase in war risk large enough to move the
# two-year yield by -25bp.
# ---------------------------------------------------------------------------
IRAQ_2003 = {
    "DGS2": -25.0,            # bp, the paper's normalisation itself
    "DGS10": -26.0,            # bp
    "T10YIE": -11.0,           # bp
    "^GSPC": -3.76,            # percent
    "BAMLC0A4CBBB": 5.0,       # bp
    "BAMLH0A0HYM2": 34.0,      # bp
    "CL=F": 0.77,              # dollars, 12-month futures
    "GC=F": 1.30,              # dollars (not significant, t = 0.26)
    "DX-Y.NYB": -0.44,         # percent, broad index
}
IRAQ_2003_SHOCK_BP = -25.0
