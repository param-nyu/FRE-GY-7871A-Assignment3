# FRE-GY 7871A — Assignment 3

**Evaluating the effects of 2026 Iran war risk on global financial markets.**
A heteroskedasticity-based replication of Rigobon and Sack (2003), *The Effects
of War Risk on U.S. Financial Markets*, applied to the 2026 Iran conflict.

Sample: 140 trading days, February 28 – September 22, 2026.

## Headline result

Implemented as written, **the paper's identification fails in 2026.** The
two-year Treasury yield it normalises on is *less* volatile on Iran-news days
than on other days — a variance ratio of **0.62**, against **6.19** in the 2003
sample. Since the estimator divides by Var_H − Var_L, that denominator is
negative and every coefficient built on it inverts. The same holds for the
10-year, the 3-month bill and breakevens; every risk asset passes cleanly.

That failure is the finding. In 2026 the Treasury curve is not a war-risk
instrument — the front end belongs to a hawkish Fed, and flight-to-quality is
weak enough that Iran news adds less variance to the curve than an ordinary macro
release. Renormalising on WTI crude restores identification.

Of nine cross-asset responses, **one is statistically distinguishable from zero**
(the dollar strengthens, bootstrap t = 2.00). Credit spreads widen as they did in
2003; rates, the dollar and gold all diverge from 2003.

## Running it

```bash
pip install -r requirements.txt
python build_report.py              # full pipeline, then assignment3_report.pdf
python build_report.py --pdf-only   # re-render from cached exhibits
```

Individual stages:

```bash
python -m src.data_loader    # FRED + Yahoo -> data/market/
python -m src.nlp_tagger     # GDELT -> data/interim/regimes.csv  (~15 min cold)
python -m src.analysis       # estimates + every exhibit -> outputs/
```

Everything caches to `data/`, which is not committed. GDELT needs no API key but
throttles hard; the fetcher is deliberately patient and resumes from cache.

## Layout

```
src/data_loader.py              FRED + Yahoo ingestion, alignment, change rules
src/war_lexicon.py              conflict/escalation/de-escalation phrases, anchors
src/nlp_tagger.py               GDELT ingestion, TF-IDF classifier, regime flags
src/heteroskedasticity_iv.py    the three IV estimators, rank/order conditions, bootstrap
src/variance_decomposition.py   war-risk share of variance (lower bounds)
src/table_generator.py          Tables 1-3, sign test, rank-condition tables
src/analysis.py                 orchestration; writes every exhibit
build_report.py                 runs the pipeline and builds the PDF
report/report.md                report source; exhibits injected at build time
AI_USE.md                       AI assistance and scope decisions
```

## Deliverables

* `assignment3_report.pdf` — Tables 1-3, both figures, and the section 8
  evaluation of heteroskedasticity identification against HFI, narrative
  identification, proxy SVARs and sign-restricted VARs.
* `outputs/` — every table as CSV, both figures as PNG.

## Known limitations

The on-the-run/off-the-run liquidity premium in the paper's Table 2 has no free
data source and is omitted. The directional classifier is weak (the two measures
correlate +0.25) and is used only to sign the shock; the headline results rest on
the binary high/low split. With 42 high-news days, statistical power is poor, and
the report leads with that rather than burying it.
