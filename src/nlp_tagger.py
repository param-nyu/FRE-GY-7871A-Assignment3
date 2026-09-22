"""Classify each day of the sample as a high- or low-war-news day.

The econometrics need only one thing from the news: a set of days on which the
*variance* of war news was elevated. The direction of any individual story does
not have to be known, which is the central convenience of the identification
strategy. The three-regime extension does need direction, and takes it from the
headline text rather than from the market reaction -- classifying a day as "bad
news" because yields rose, then estimating the yield response to bad news,
would be circular.

Two signals are combined:

* **News volume** -- GDELT's share-of-global-coverage timeline for the conflict
  query. This is the closest available analogue to "days on which war news was
  the primary determinant of asset price movements", which Rigobon and Sack
  established by reading newspapers by hand.
* **Headline intensity** -- a domain lexicon plus TF-IDF cosine similarity to
  prototypical escalation and de-escalation statements, which is what allows
  novel 2026 phrasing to be scored at all.

Running ``python -m src.nlp_tagger`` writes ``data/interim/regimes.csv``.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import requests

from .config import INTERIM_DIR, NEWS_DIR, SAMPLE_END, SAMPLE_START
from .war_lexicon import (CONFLICT_TERMS, DEESCALATION_ANCHORS,
                          DEESCALATION_TERMS, ESCALATION_ANCHORS,
                          ESCALATION_TERMS, GDELT_DEESCALATION_QUERY,
                          GDELT_ESCALATION_QUERY, GDELT_QUERY,
                          GDELT_TIMELINE_QUERY)

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "FRE-GY-7871A coursework (NYU Tandon); contact shahparam87@gmail.com"

# GDELT documents a one-request-per-five-seconds limit but in practice throttles
# a good deal harder than that on sustained querying, returning 429s or a plain
# text "please slow down" body where JSON is expected. Both are treated as
# back-pressure rather than failure: the fetcher waits longer and tries again.
GDELT_PAUSE = 15.0
GDELT_RETRIES = 6
GDELT_THROTTLE_WAIT = 75.0


def _gdelt(params: dict, cache_name: str, refresh: bool = False) -> dict:
    """One GDELT call, cached to disk and retried through the rate limiter.

    Every window is cached under its own filename, so an interrupted run
    resumes where it stopped instead of re-querying from the beginning -- which
    matters when the whole corpus takes several minutes of deliberate waiting.
    """
    cache = NEWS_DIR / cache_name
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    last = None
    for attempt in range(GDELT_RETRIES):
        time.sleep(GDELT_PAUSE if attempt == 0
                   else GDELT_THROTTLE_WAIT * attempt)
        try:
            response = requests.get(GDELT, params={**params, "format": "json"},
                                    timeout=180,
                                    headers={"User-Agent": USER_AGENT})
            if response.status_code == 429:
                last = RuntimeError("429 rate limited")
                continue
            response.raise_for_status()
            body = response.content.decode("utf-8", "replace")
            if not body.lstrip().startswith("{"):
                # Throttle notices come back as prose with a 200 status.
                last = RuntimeError(f"non-JSON body: {body[:80]!r}")
                continue
            payload = json.loads(body)
        except Exception as exc:                         # noqa: BLE001
            last = exc
            continue
        cache.write_text(json.dumps(payload))
        return payload
    raise RuntimeError(f"GDELT failed for {cache_name}: {last!r}")


# ---------------------------------------------------------------------------
# News volume
# ---------------------------------------------------------------------------
def fetch_volume(start=SAMPLE_START, end=SAMPLE_END,
                 refresh: bool = False) -> pd.Series:
    """Daily share of worldwide news coverage matching the conflict query."""
    payload = _gdelt({
        "query": GDELT_TIMELINE_QUERY,
        "mode": "timelinevol",
        "startdatetime": f"{start:%Y%m%d}000000",
        "enddatetime": f"{end:%Y%m%d}235959",
    }, "volume_timeline.json", refresh=refresh)

    points = payload["timeline"][0]["data"]
    index = pd.to_datetime([p["date"][:8] for p in points], format="%Y%m%d")
    return pd.Series([float(p["value"]) for p in points], index=index,
                     name="news_volume")


def fetch_direction_volumes(start=SAMPLE_START, end=SAMPLE_END,
                            refresh: bool = False) -> pd.DataFrame:
    """Daily escalation and de-escalation coverage shares.

    Two timeline queries, each computed over GDELT's full corpus. This is the
    primary directional signal: it sees every matching article rather than the
    250 the article endpoint will return, and costs two requests.
    """
    frames = {}
    for label, query, cache in (
        ("esc_volume", GDELT_ESCALATION_QUERY, "volume_escalation.json"),
        ("deesc_volume", GDELT_DEESCALATION_QUERY, "volume_deescalation.json"),
    ):
        payload = _gdelt({
            "query": query, "mode": "timelinevol",
            "startdatetime": f"{start:%Y%m%d}000000",
            "enddatetime": f"{end:%Y%m%d}235959",
        }, cache, refresh=refresh)
        points = payload["timeline"][0]["data"]
        idx = pd.to_datetime([p["date"][:8] for p in points], format="%Y%m%d")
        frames[label] = pd.Series([float(p["value"]) for p in points], index=idx)
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------
def fetch_headlines(start=SAMPLE_START, end=SAMPLE_END,
                    refresh: bool = False) -> pd.DataFrame:
    """Conflict headlines, fetched in monthly windows.

    GDELT caps a single article query at 250 records and throttles sustained
    querying aggressively -- in practice far below its published one-per-five-
    seconds limit. Weekly windows would need thirty requests and take over an
    hour of deliberate waiting, so the corpus is walked a month at a time and
    the *daily* signals come from the timeline endpoint instead, which is
    computed over GDELT's whole corpus rather than a capped sample. Each window
    is cached separately, so an interrupted run resumes.
    """
    rows = []
    missing: list[str] = []
    window_start = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while window_start <= stop:
        window_end = min(window_start + pd.Timedelta(days=29), stop)
        try:
            payload = _gdelt({
                "query": GDELT_QUERY,
                "mode": "artlist",
                "maxrecords": 250,
                "sort": "hybridrel",
                "startdatetime": f"{window_start:%Y%m%d}000000",
                "enddatetime": f"{window_end:%Y%m%d}235959",
            }, f"articles_{window_start:%Y%m%d}.json", refresh=refresh)
        except RuntimeError as exc:
            # The headline corpus is the *secondary* signal: it feeds the
            # lexicon and TF-IDF classifier, while the daily intensity and
            # direction series come from the timeline endpoint over GDELT's
            # whole corpus. Losing a window degrades the classifier's coverage
            # but does not invalidate the regimes, so a window that will not
            # download is recorded and skipped rather than aborting the run.
            print(f"    {window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}: "
                  f"UNAVAILABLE ({exc})", flush=True)
            missing.append(f"{window_start:%Y-%m-%d}..{window_end:%Y-%m-%d}")
            window_start = window_end + pd.Timedelta(days=1)
            continue

        for article in payload.get("articles", []):
            stamp = article.get("seendate", "")
            if not stamp:
                continue
            rows.append({
                "date": pd.to_datetime(stamp[:8], format="%Y%m%d"),
                "title": article.get("title", ""),
                "domain": article.get("domain", ""),
            })
        print(f"    {window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}: "
              f"{len(payload.get('articles', []))} articles", flush=True)
        window_start = window_end + pd.Timedelta(days=1)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["date", "title"])
    frame.to_csv(NEWS_DIR / "headlines.csv", index=False)
    if missing:
        (NEWS_DIR / "missing_windows.txt").write_text("\n".join(missing))
        print(f"    {len(missing)} window(s) unavailable; recorded in "
              f"data/news/missing_windows.txt")
    return frame


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _phrase_counter(terms: list[str]):
    """Match whole words/phrases, so 'war' does not fire inside 'warehouse'."""
    pattern = re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t in sorted(terms, key=len, reverse=True)),
        re.I)
    return lambda text: len(pattern.findall(text or ""))


count_conflict = _phrase_counter(CONFLICT_TERMS)
count_escalation = _phrase_counter(ESCALATION_TERMS)
count_deescalation = _phrase_counter(DEESCALATION_TERMS)


def anchor_similarity(titles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Max TF-IDF cosine similarity of each headline to each anchor set.

    The lexicon can only match language someone thought to write down. The
    anchors catch headlines that use unfamiliar wording but share vocabulary
    with a prototypical escalation or de-escalation statement -- the novel
    phrasing problem the brief asks the classifier to handle.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    anchors = ESCALATION_ANCHORS + DEESCALATION_ANCHORS
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                          ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(list(titles) + anchors)
    head, anch = matrix[:len(titles)], matrix[len(titles):]
    sim = cosine_similarity(head, anch)
    n_esc = len(ESCALATION_ANCHORS)
    return sim[:, :n_esc].max(axis=1), sim[:, n_esc:].max(axis=1)


def score_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """Per-headline conflict relevance and directional score."""
    out = headlines.copy()
    titles = out.title.fillna("").tolist()

    out["conflict_hits"] = [count_conflict(t) for t in titles]
    out["esc_hits"] = [count_escalation(t) for t in titles]
    out["deesc_hits"] = [count_deescalation(t) for t in titles]
    esc_sim, deesc_sim = anchor_similarity(titles)
    out["esc_sim"], out["deesc_sim"] = esc_sim, deesc_sim

    # Relevance gates the headline in; direction is only meaningful once it is.
    out["is_conflict"] = (out.conflict_hits > 0) | (
        np.maximum(esc_sim, deesc_sim) > 0.10)
    # Lexicon hits and anchor similarity are on different scales, so each is
    # standardised within the corpus before being summed.
    lex = out.esc_hits - out.deesc_hits
    sim = out.esc_sim - out.deesc_sim
    out["direction"] = (_z(lex) + _z(sim)) / 2.0
    return out


def _z(series: pd.Series) -> pd.Series:
    sd = series.std()
    return (series - series.mean()) / sd if sd and np.isfinite(sd) else series * 0.0


# ---------------------------------------------------------------------------
# Daily regimes
# ---------------------------------------------------------------------------
def build_daily(volume: pd.Series, direction_volumes: pd.DataFrame,
                scored: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: intensity, and direction from both measures."""
    conflict = scored[scored.is_conflict]
    daily = conflict.groupby("date").agg(
        n_articles=("title", "size"),
        lexicon_direction=("direction", "mean"),
        esc_hits=("esc_hits", "sum"),
        deesc_hits=("deesc_hits", "sum"),
    )
    daily = daily.join(volume.rename("news_volume"), how="outer")
    daily = daily.join(direction_volumes, how="outer")
    daily = daily.fillna({"news_volume": 0.0, "n_articles": 0,
                          "lexicon_direction": 0.0, "esc_hits": 0,
                          "deesc_hits": 0, "esc_volume": 0.0,
                          "deesc_volume": 0.0})

    # Corpus-wide directional signal: which flavour of coverage dominated that
    # day, on GDELT's whole corpus rather than a capped headline sample.
    daily["volume_direction"] = _z(daily.esc_volume) - _z(daily.deesc_volume)
    return daily


def assign_regimes(daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
                   high_quantile: float = 0.70,
                   direction_cut: float = 0.0) -> pd.DataFrame:
    """Flag H/L days, and split H into escalation and de-escalation.

    `high_quantile` sets how many days are treated as war-news days. Rigobon
    and Sack use 17 of 47 business days, a 36% share; the 70th percentile here
    puts roughly 30% of trading days in H, which keeps the two samples
    comparable in size without diluting the variance shift.
    """
    frame = daily.reindex(trading_days).fillna(
        {"news_volume": 0.0, "n_articles": 0, "lexicon_direction": 0.0,
         "esc_hits": 0, "deesc_hits": 0, "esc_volume": 0.0,
         "deesc_volume": 0.0, "volume_direction": 0.0})

    # Intensity combines how much the world was talking about the conflict with
    # how much of that coverage our classifier recognised as conflict news.
    frame["intensity"] = (_z(frame.news_volume) + _z(frame.n_articles)) / 2.0

    threshold = frame.intensity.quantile(high_quantile)
    frame["regime_binary"] = np.where(frame.intensity >= threshold, "H", "L")

    # Direction combines the corpus-wide volume signal with the lexicon and
    # TF-IDF classifier run on the headline sample. They are built from
    # different data, so their agreement is a check on both; the report
    # records the correlation rather than assuming it.
    frame["mean_direction"] = (_z(frame.volume_direction)
                               + _z(frame.lexicon_direction)) / 2.0

    frame["regime_3"] = np.where(
        frame.regime_binary == "L", "L_baseline",
        np.where(frame.mean_direction >= direction_cut, "H_bad", "H_good"))
    frame["threshold"] = threshold
    return frame


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    from .data_loader import load

    _, changes = load()
    trading_days = changes.index

    print("Fetching GDELT news volume timeline ...")
    volume = fetch_volume()
    print(f"  {len(volume)} daily points, "
          f"{volume.index.min():%Y-%m-%d} to {volume.index.max():%Y-%m-%d}")

    print("Fetching conflict headlines in weekly windows "
          "(rate-limited; ~8 minutes on a cold cache) ...")
    headlines = fetch_headlines()
    print(f"  {len(headlines):,} unique headlines")

    scored = score_headlines(headlines)
    print(f"  {int(scored.is_conflict.sum()):,} classified as conflict news "
          f"({100 * scored.is_conflict.mean():.0f}%)")

    print("Fetching directional coverage timelines ...")
    direction_volumes = fetch_direction_volumes()
    print(f"  escalation and de-escalation series, "
          f"{len(direction_volumes)} days each")

    daily = build_daily(volume, direction_volumes, scored)
    regimes = assign_regimes(daily, trading_days)

    overlap = regimes[["volume_direction", "lexicon_direction"]].dropna()
    if len(overlap) > 2:
        rho = overlap.volume_direction.corr(overlap.lexicon_direction)
        print(f"\n  corr(corpus-volume direction, lexicon+TF-IDF direction) "
              f"= {rho:+.3f} over {len(overlap)} days")
    regimes.to_csv(INTERIM_DIR / "regimes.csv")
    scored.to_csv(INTERIM_DIR / "headlines_scored.csv", index=False)

    counts = regimes.regime_3.value_counts()
    print(f"\nRegimes over {len(regimes)} trading days:")
    print(f"  H (war news)  {int((regimes.regime_binary == 'H').sum()):>4}")
    print(f"  L (other)     {int((regimes.regime_binary == 'L').sum()):>4}")
    for name in ("H_bad", "H_good", "L_baseline"):
        print(f"    {name:<12} {int(counts.get(name, 0)):>4}")

    print("\nTop 12 days by news intensity:")
    top = regimes.nlargest(12, "intensity")
    for day, row in top.iterrows():
        print(f"  {day:%Y-%m-%d}  vol={row.news_volume:6.3f}  "
              f"n={int(row.n_articles):>3}  dir={row.mean_direction:+.2f}  "
              f"{row.regime_3}")


if __name__ == "__main__":
    main()
