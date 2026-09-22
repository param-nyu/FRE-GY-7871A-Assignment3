"""Domain-specific lexicon and anchor statements for 2026 Iran war news.

Two lessons carry over from the Fedspeak work in Assignment 2, and both are the
reason this is hand-built rather than an off-the-shelf sentiment model:

1. General sentiment models score the *valence of events*, not the dimension you
   care about. FinBERT reads "inflation remains elevated" as good news because
   it grades the economy. The same failure appears here: a headline reading
   "Iran signals willingness to return to talks" is positive-sentiment and
   de-escalatory, but "US strikes destroy Iranian air defences" is
   negative-sentiment and *also* an escalation -- sentiment and escalation are
   different axes that happen to correlate loosely.

2. Legacy lexicons have no entry for language coined during this conflict.
   Terms like "energy corridor blockade" or "nuclear threshold enrichment"
   postdate every general-purpose dictionary, which is why the classifier pairs
   an explicit phrase list with TF-IDF similarity to anchor statements: the
   lexicon catches what we can enumerate, the anchors catch novel phrasing that
   is merely *similar* to what we can enumerate.

Note on the regime split: escalation and de-escalation are scored from the
*text*, never from the market reaction. Classifying a day as "bad war news"
because yields rose and then estimating the yield response to bad war news
would be circular.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Intensity: is this a war-news day at all?
# ---------------------------------------------------------------------------
# Terms whose presence marks a headline as being about the conflict itself
# rather than about Iran in general (sport, culture, diaspora, routine
# diplomacy unconnected to the military situation).
CONFLICT_TERMS = [
    "strike", "strikes", "airstrike", "air strike", "missile", "missiles",
    "drone", "drones", "attack", "attacked", "retaliation", "retaliate",
    "war", "warfare", "military", "troops", "deployment", "carrier",
    "escalation", "escalate", "bombing", "bombard", "casualties", "killed",
    "nuclear", "enrichment", "centrifuge", "uranium", "iaea", "safeguards",
    "hormuz", "strait", "tanker", "blockade", "shipping lane", "chokepoint",
    "sanctions", "embargo", "ceasefire", "truce", "armistice", "negotiation",
    "talks", "diplomacy", "ultimatum", "mobilisation", "mobilization",
    "proxy", "houthi", "hezbollah", "irgc", "revolutionary guard",
]

# ---------------------------------------------------------------------------
# Direction: escalation versus de-escalation
# ---------------------------------------------------------------------------
ESCALATION_TERMS = [
    "strike", "strikes", "airstrike", "air strike", "bombard", "bombing",
    "launched", "launches", "missile barrage", "salvo", "retaliation",
    "retaliates", "escalation", "escalates", "escalating", "killed", "deaths",
    "casualties", "invasion", "offensive", "incursion", "shot down",
    "downed", "seized", "seizure", "blockade", "blockades", "closed the strait",
    "closure of hormuz", "mines", "mining", "ultimatum", "deadline",
    "mobilisation", "mobilization", "reinforcements", "carrier strike group",
    "deploys", "deployment", "threatens", "threat", "warns of", "vows revenge",
    "breakout", "weapons-grade", "enrichment surge", "expelled inspectors",
    "withdraws from", "suspends cooperation", "nuclear threshold",
    "energy corridor", "state of emergency", "evacuate", "evacuation",
    "sanctions imposed", "new sanctions", "snapback",
]

DEESCALATION_TERMS = [
    "ceasefire", "cease-fire", "truce", "armistice", "de-escalation",
    "de-escalate", "deescalation", "stand down", "withdrawal", "withdraws troops",
    "pullback", "pull back", "peace talks", "negotiations resume",
    "returns to talks", "diplomatic breakthrough", "agreement reached",
    "deal reached", "framework agreement", "accord", "mediation", "mediated",
    "resumes inspections", "inspectors return", "restraint", "calm",
    "easing tensions", "tensions ease", "sanctions relief", "sanctions lifted",
    "released", "release of detainees", "prisoner exchange", "hotline",
    "reopens", "reopening", "resumed shipping", "safe passage",
    "no further strikes", "halt", "pause", "moratorium",
]

# ---------------------------------------------------------------------------
# Anchor statements for TF-IDF / cosine similarity
# ---------------------------------------------------------------------------
# These are not headlines; they are short prototypical descriptions of the
# states the classifier is trying to recognise. A novel headline that shares no
# exact lexicon phrase can still score highly against an anchor through shared
# vocabulary, which is how the method is meant to cope with new phrasing.
ESCALATION_ANCHORS = [
    "United States military forces conduct strikes on Iranian territory "
    "destroying air defence systems and command control stations",
    "Iran launches ballistic missile and drone salvo at regional bases "
    "causing casualties and prompting interception by allied forces",
    "Iran moves to close the Strait of Hormuz seizing tankers and mining the "
    "energy corridor disrupting global oil shipping through the chokepoint",
    "Iran accelerates uranium enrichment toward weapons grade crossing the "
    "nuclear threshold and expelling international inspectors from its sites",
    "Washington issues an ultimatum and deploys a carrier strike group as "
    "military mobilisation escalates and both sides threaten retaliation",
]

DEESCALATION_ANCHORS = [
    "Iran and the United States agree a ceasefire halting strikes as both "
    "sides stand down military forces and observe a truce",
    "Negotiations resume with a mediated framework agreement and diplomats "
    "report a breakthrough toward a comprehensive nuclear accord",
    "International inspectors return to Iranian nuclear facilities as Tehran "
    "resumes cooperation with safeguards and suspends enrichment",
    "Shipping resumes safely through the Strait of Hormuz as the blockade is "
    "lifted and tankers are released restoring energy corridor traffic",
    "Sanctions relief is announced and detainees are released in a prisoner "
    "exchange as tensions ease and the crisis calms",
]

# The GDELT query used to pull the corpus. Kept here so the report can state
# exactly what was searched rather than describing it loosely.
GDELT_QUERY = (
    "Iran (strike OR strikes OR missile OR drone OR nuclear OR enrichment "
    "OR Hormuz OR tanker OR ceasefire OR sanctions OR retaliation OR war) "
    "sourcelang:english"
)

# A narrower query for the daily intensity timeline. GDELT's timeline endpoint
# rejects very long boolean queries, so the volume series uses a compact form
# and the article corpus uses the full one above.
GDELT_TIMELINE_QUERY = "Iran (strike OR missile OR Hormuz OR ceasefire OR nuclear) sourcelang:english"


# ---------------------------------------------------------------------------
# Directional timeline queries
# ---------------------------------------------------------------------------
# GDELT's timeline endpoint reports the share of *all* worldwide coverage
# matching a query, computed over its entire corpus rather than a 250-record
# sample. Running one escalation-flavoured and one de-escalation-flavoured
# query therefore gives a daily directional signal built on far more articles
# than any feasible headline download, at the cost of two requests instead of
# thirty. The headline corpus is still collected, and is what the lexicon and
# TF-IDF classifier run on; the two directional measures are then cross-checked
# against one another in the report.
GDELT_ESCALATION_QUERY = (
    "Iran (strike OR strikes OR missile OR drone OR attack OR retaliation "
    "OR escalation OR bombing OR blockade) sourcelang:english"
)
GDELT_DEESCALATION_QUERY = (
    "Iran (ceasefire OR truce OR talks OR negotiations OR agreement "
    "OR diplomacy OR deescalation OR withdrawal) sourcelang:english"
)
