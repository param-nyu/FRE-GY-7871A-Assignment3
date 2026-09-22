# Evaluating the effects of 2026 Iran war risk on global financial markets

## A heteroskedasticity-based replication of Rigobon and Sack (2003)

FRE-GY 7871A, NLP and the Investment Process — Assignment 3 — Param Shah — September 22, 2026

---

## 1. Summary

Rigobon and Sack measured the effect of an unobservable war risk factor on U.S.
markets by exploiting the fact that its variance spikes on days of heavy war news.
This report applies their estimator to the 2026 Iran conflict over 140 trading
days from February 28 to September 22, using a domain-specific NLP classifier on
GDELT news to identify the high-variance days.

The central result is methodological, and it is not the one the brief anticipated.

- **The paper's identification fails in 2026 when implemented as written.** The
  two-year Treasury yield, which Rigobon and Sack normalise on, is **less**
  volatile on Iran-news days than on other days — a variance ratio of 0.62,
  against **6.19** in their 2003 sample. Because the estimator divides by
  Var_H − Var_L, that denominator is negative and every coefficient built on it
  inverts. This holds for the 10-year, the 3-month bill and breakevens too, and
  is stable across every news threshold tested.
- **This failure is the finding.** It is the brief's own thesis in sharper form:
  in 2026 the Treasury curve is not a war-risk instrument. Its variance is
  dominated by monetary policy, and flight-to-quality is weak enough that Iran
  news barely moves the front end relative to macro news.
- **Renormalising on WTI crude restores identification** (variance ratio 1.49 to
  2.9 depending on threshold) and yields interpretable cross-asset responses.
- **Only one of those responses is statistically distinguishable from zero.**
  The dollar strengthens on war risk (bootstrap t = 2.00). Everything else is
  directional evidence at 42 high-news days, and the report treats it as such.

---

## 2. Data

{{COVERAGE}}

<div class="notes">All series are downloaded automatically: FRED for rates and
credit spreads, Yahoo Finance for equities, commodities and the dollar index.
Rates and spreads are first-differenced to basis points; equities and the dollar
are 100 x log changes; oil and gold are dollar changes, following the paper's
own units. FRED and NYSE holiday calendars differ slightly, so missing days are
handled pairwise rather than filled.</div>

Rigobon and Sack also report the on-the-run/off-the-run ten-year liquidity
premium. That series is constructed from proprietary CUSIP-level quotes and has
no free equivalent, so the row is absent here rather than filled with a loose
proxy.

---

## 3. NLP: identifying high-variance days

The estimator needs only a set of days on which the **variance** of war news was
elevated. It does not need the direction of any individual story, which is the
central convenience of the approach. Rigobon and Sack built their 17-day list by
reading newspapers. This report builds the equivalent from GDELT.

### 3.1 Why not an off-the-shelf sentiment model

Two failure modes rule one out, and both were observed directly in the previous
assignment's Fedspeak work.

*General sentiment models grade the wrong axis.* FinBERT scores "inflation
remains elevated" as positive news because it evaluates the economy, not the
policy stance. The same confusion appears here: escalation and sentiment are
distinct dimensions. "US strikes destroy Iranian air defences" is
negative-sentiment and escalatory; "Iran signals willingness to return to talks"
is positive-sentiment and de-escalatory — but the correlation is loose enough
that sentiment is not a usable proxy for war intensity.

*Legacy lexicons have no entry for language coined during this conflict.* Terms
like "energy corridor blockade" or "nuclear threshold enrichment" postdate every
general-purpose dictionary.

### 3.2 The classifier

Two signals are combined, built from different data so that their agreement is a
check rather than an assumption:

1. **Corpus-wide volume.** GDELT timeline queries return the share of *all*
   worldwide coverage matching a query, computed over the full corpus rather
   than a capped sample. One conflict-intensity query gives daily news volume;
   an escalation-flavoured and a de-escalation-flavoured query give direction.
2. **Lexicon and TF-IDF anchors** on a 961-headline corpus. A hand-built list of
   52 conflict, 57 escalation and 42 de-escalation phrases catches what can be
   enumerated; TF-IDF cosine similarity against ten prototypical anchor
   statements catches novel phrasing that merely *resembles* what can be
   enumerated. This is the mechanism for the novel-phrasing problem: a headline
   sharing no exact phrase can still score through shared vocabulary.

The two directional measures correlate **+0.25** across 140 trading days. That is
positive, so they agree in sign, but it is weak, and the report does not lean on
the directional split for anything the binary split can carry.

**The regime split is classified from news text only, never from market
reactions.** Labelling a day "bad war news" because yields rose, then estimating
the yield response to bad war news, would be circular.

{{REGIME_SUMMARY}}

The 30% share of days in H is close to the paper's 17 of 47 business days (36%).

{{FIGURE1}}

## 4. Table 1. Event timeline

{{TABLE1}}

<div class="notes">The 25 highest-intensity days, with the headline the
classifier scored most strongly in either direction. News volume is the share of
worldwide coverage. Direction is positive for escalation. Days showing zero
articles had high corpus-wide volume but no headline captured in the 250-record
monthly samples, which is precisely why the timeline endpoint rather than the
headline corpus is the primary intensity signal.</div>

The timeline tracks the conflict's actual narrative -- the opening escalation in
early March, the Hormuz ultimatum on March 23, the April 8 ceasefire, its
breakdown through May and June, and renewed strikes in July and September.

It also shows the directional classifier's limits plainly. March 31 -- "US
attacks Iranian nuclear site while Tehran hits oil tanker" -- is scored
de-escalatory, which is simply wrong; the day's coverage was dominated by
ceasefire-negotiation vocabulary surrounding the strike, and a bag-of-phrases
classifier cannot tell a strike *during* talks from progress *in* talks. The
April 8 ceasefire, by contrast, is scored correctly and strongly (-3.04).

This is why the binary H/L split carries the identification and the three-regime
split is used only to sign the shock in section 5.3, where its weakness is
reported rather than hidden. **The headline results do not depend on the
direction classifier at all** -- which is precisely the property that makes
heteroskedasticity-based identification attractive here: it needs to know when
war news was loud, not what it said.

---

## 5. Identification, and where it breaks

### 5.1 The estimator

For two variables the reduced form is `[dx1, dx2]' = D z + mu`, with the loading
of the war risk factor on x1 normalised to one. If only the variance of that
factor rises on days H, then

> dOmega = Omega_H − Omega_L = dsigma²(z1) · [[1, d21], [d21, d21²]]

and d21 follows from the shift in second moments without the factor ever being
quantified. Implemented as instrumental variables, the instrument is the
variable's change on H days and its negative on L days (the paper's equation 8);
w1 uses x1, w2 uses x2, and w3 stacks both.

Low-variance days are chosen as an equal-sized set of days nearest to but
excluding the H days, per the paper's footnote 7, which limits how much the
*other* factors can shift between the two samples.

### 5.2 The rank condition

Everything rests on the benchmark being more volatile on war-news days. It is not.

| Benchmark | Var_H | Var_L | Ratio | Status |
|---|---|---|---|---|
| 2-Year Treasury, 2003 (Rigobon-Sack Table 3) | .00594 | .00096 | **6.19** | satisfied |
| 2-Year Treasury, 2026 Iran | 21.74 | 30.62 | **0.62** | **FAILS** |
| 10-Year Treasury, 2026 | 14.62 | 28.71 | 0.51 | FAILS |
| Break-even inflation, 2026 | 4.00 | 5.90 | 0.68 | FAILS |
| 3-Month bill, 2026 | 4.10 | 4.21 | 0.97 | FAILS |
| **WTI crude, 2026** | 27.71 | 18.56 | **1.49** | satisfied |
| High-yield spread, 2026 | 46.79 | 25.12 | 1.86 | satisfied |
| Gold, 2026 | 9538 | 3408 | 2.80 | satisfied |
| Dollar index, 2026 | 0.172 | 0.082 | 2.10 | satisfied |

The split is clean and it is not an artefact: it holds at every news threshold
from the 60th to the 90th percentile, under both mean-squared and demeaned
variance, and under both low-day selection rules.

**Every risk asset passes. Every Treasury instrument fails.** In 2003 the
two-year was the principal channel through which war risk reached markets. In
2026 the front end is driven by a hawkish Federal Reserve, and flight-to-quality
is weak enough that Iran news adds less variance to the curve than an ordinary
macro release does. Normalising on it would divide by a negative number.

**WTI crude is therefore the benchmark** for the headline results. It satisfies
the condition comfortably and is the cleanest economic channel from Middle East
conflict to global markets. The paper's own specification is still estimated and
reported in section 6.3, because its failure is the evidence for this choice.

{{RANK_CONDITION}}

### 5.3 The sign of the shock

Heteroskedasticity identifies the *ratio* of responses, not the sign of the
factor. "An increase in war risk" means whichever direction of z1 we declare it
to mean. Rigobon and Sack resolved this by assumption, normalising to a 25bp
*fall* in the two-year because flight to quality was the 2003 prior. Assuming
the answer is exactly what is at issue in 2026, so the three-regime split is
used to settle it from data instead.

{{SIGN_TEST}}

Escalation days show oil higher by $0.99 and the two-year higher by 2.65bp than
de-escalation days; both contrasts point the same way, and **the two-year rises
on war risk**, as the brief anticipates. But neither contrast is significant
(p = 0.63 and p = 0.18). The sign convention is therefore taken from the data's
direction while acknowledging that the data support it only weakly — a caveat
that propagates to every sign in Table 2.

---

## 6. Table 2. Cross-asset sensitivity

### 6.1 Headline: normalised on WTI crude

{{TABLE2}}

<div class="notes">Structural responses to an increase in war risk large enough
to raise WTI by $1/bbl. Absolute asymptotic t-statistics in the adjacent
columns; the final two columns give the 500-replication bootstrap, which
resamples within each regime so the H/L proportions that carry the
identification are preserved.</div>

Reading the bootstrap column rather than the asymptotic one, **only the dollar
response is distinguishable from zero** (t = 2.00, 95% CI [0.02, 0.23]). The
high-yield spread and gold are the next largest in magnitude but their intervals
straddle zero comfortably. With 42 high-news days, this is the honest resolution
of the exercise: directions worth reporting, magnitudes that mostly are not.

### 6.2 2026 Iran versus 2003 Iraq

{{SIGN_COMPARISON}}

Directions only. The 2003 column was normalised to a 25bp fall in the two-year
under an assumed flight-to-quality sign, the 2026 column to a $1/bbl rise in oil;
the magnitudes are not comparable and are not compared.

**Credit spreads behave exactly as they did in 2003** — both BBB and high-yield
widen on war risk. Everything else diverges. The Treasury curve rises where it
fell in 2003, which is the brief's central prediction and is confirmed. The
dollar strengthens where it weakened, consistent with the United States being a
net energy exporter at roughly 14 million barrels a day rather than the ~6
million of 2003: an oil shock is no longer unambiguously a negative terms-of-
trade shock for the dollar. Gold falls rather than rising, though insignificantly.

Equities are the one result that contradicts the brief, which expected them to
fall as in 2003. The estimated response is mildly *positive* (+0.11% per $1/bbl)
and thoroughly insignificant (t = 0.55, CI [−0.26, 0.83]). The correct reading is
that the S&P response cannot be signed at this sample size, not that equities
rise on war risk.

### 6.3 The paper's specification, reported as failing

{{TABLE2_CORE}}

<div class="notes">Core-crisis window (February 28 to May 15), the robustness
check matching the paper's own ten-week design. Normalised on WTI as above.</div>

The two-year-normalised replication is written to
`outputs/table2c_replication_2y.csv`. Its coefficients are not interpreted here:
they are built on a denominator of Var_H − Var_L = −8.88, and a ratio whose
denominator has the wrong sign inverts every response. It is retained as the
diagnostic that motivates section 5.2, not as an estimate.

---

## 7. Table 3. Variance decomposition

{{TABLE3}}

<div class="notes">Var on L and H days are mean squared daily changes, as in the
paper's footnote 12. The predicted change is d_j1² x [Var_H(oil) − Var_L(oil)].
Shares are <strong>lower bounds</strong>: the shift in war-induced variance must
be smaller than its level on H days, and the two coincide only if there is no
war news at all on L days.</div>

{{FIGURE2}}

War risk accounts for a substantial share of the variance of exactly those assets
whose rank condition it satisfies — gold (82% of all-day variance), the dollar
(75%), high-yield spreads (62%), BBB spreads (45%) — and very little of the
Treasury curve (5% to 10%). This is the same result as section 5.2 seen from the
other side: the war risk factor is a *risk-asset* factor in 2026, not a
rates factor.

Rigobon and Sack found war risk explained 13% to 63% of cumulative variance in
2003. The comparable range here is 2% to 41%, with the top of the range occupied
by gold and the dollar rather than by Treasuries.

---

## 8. Is heteroskedasticity-based identification the right approach?

The brief asks for this evaluation directly. The short answer: it is the right
approach for this problem, and this study's difficulties are not an argument
against it — the method diagnosed its own failure, which is more than most
alternatives would have done.

### 8.1 What it does well

The problem is that war risk is unobservable and markets are simultaneously
determined. Heteroskedasticity-based identification needs only that the analyst
can date the days when the factor was unusually volatile — a far weaker
requirement than quantifying it. Crucially, **the rank condition is checkable**.
Section 5.2 is not a failure of the method; it is the method reporting that its
assumption does not hold for a particular normalisation, which is exactly what a
well-posed identification strategy should do. An approach that silently returned
a plausible-looking number for the two-year would have been worse.

### 8.2 Where it is weak here

*Regime assignment is a researcher choice.* The threshold that splits H from L is
a free parameter. Results were checked from the 60th to the 90th percentile and
the variance-ordering conclusion is stable, but the coefficients move.

*It requires other factors to be homoskedastic across regimes.* Over seven months
spanning escalation, a June ceasefire and its aftermath, this is a strong
assumption. The core-window robustness check exists for this reason.

*Power is poor.* Identification comes from a difference of variances, a
second-moment object estimated from 42 days. Only one of nine responses is
significant, which is a statement about the estimator's efficiency as much as
about the world.

### 8.3 The alternatives

**High-frequency identification / event studies.** Narrow the window to minutes
around each headline so that nothing else can plausibly move prices. This is the
strongest identification available and would be a genuine improvement: with
intraday data the 42 daily observations become hundreds of tightly-identified
headline events, and the homoskedasticity assumption becomes near-trivial over a
30-minute window. Its cost is data — intraday quotes for Treasuries, oil, credit
and FX are expensive, which is why this study is daily — and it measures only the
*announcement* effect, missing risk that builds gradually. **This is the single
best alternative, and the binding constraint is budget rather than method.**

**Narrative identification (Romer and Romer).** Read the record and construct a
shock series directly from documents. Its advantage is that it can distinguish
*kinds* of war news, which the heteroskedasticity approach explicitly collapses
into one factor. Its cost is exactly what Rigobon and Sack were avoiding: it
requires quantifying the news, which reintroduces the judgement the method was
designed to escape, and it is not reproducible across analysts. Notably, the NLP
pipeline in section 3 is a narrative approach in every respect except that it
only classifies *intensity* rather than magnitude — a deliberately narrow use of
narrative information to serve a heteroskedasticity-based estimator.

**Proxy SVAR / external-instrument VAR.** Use an observable correlated with the
shock but not with other structural shocks — oil price jumps around known
military events, or a geopolitical risk index. This buys dynamics: impulse
responses over days and weeks rather than a single contemporaneous coefficient,
which matters because war risk plausibly has persistent effects. Its cost is a
valid instrument, and the obvious candidates here are circular: instrumenting
with oil and then estimating the oil response is not identification. Given that
this study ends up normalising on oil, a proxy SVAR would need a genuinely
external instrument to add anything.

**Sign-restricted structural VAR.** Impose only the signs of responses that
theory is confident about and report the set of admissible models. It is robust
where point identification is fragile — but its weakness is fatal here. The
central finding of this report is that a *sign* flipped between 2003 and 2026:
Treasury yields now rise on war risk. A method that assumes the signs cannot
discover that they changed. Sign restrictions would have imposed the 2003 prior
and returned it.

### 8.4 Verdict

Heteroskedasticity-based identification is well matched to this problem and
should remain the primary approach, with **high-frequency identification the
upgrade to make if intraday data can be obtained**. Sign restrictions are
specifically inappropriate to the question, since the sign change is the result.
A proxy SVAR would be a useful complement for dynamics, but needs an instrument
this study does not have. The practical recommendation is to keep this estimator
and improve its inputs — finer regime dating from intraday news timestamps — rather
than to replace it.

---

## 9. Conclusions

The mechanics of Rigobon and Sack transfer to the 2026 Iran conflict; their
choice of numeraire does not. The two-year Treasury yield was the right
normalisation for a 2003 war scare in which flight to quality dominated, and it
is the wrong one for a 2026 conflict in which the front end belongs to the
Federal Reserve. Re-anchoring on oil restores identification and produces
responses that match 2003 for credit spreads and diverge from it for rates, the
dollar and gold.

The honest summary of the estimates is that they are directionally informative
and statistically weak. One response of nine is significant. The variance
decomposition is firmer, because it rests on second moments measured over the
whole sample rather than on a ratio of differences, and it says clearly that
2026 war risk is a risk-asset factor rather than a rates factor.

---

## 10. Reproducibility

`python build_report.py` runs the entire pipeline and rebuilds this document.
Market data downloads automatically from FRED and Yahoo Finance; news comes from
GDELT's free API with no key. Every table and figure is injected from `outputs/`
at build time, so the report cannot drift from the analysis that produced it. No
data files are committed. AI assistance is documented in `AI_USE.md`.
