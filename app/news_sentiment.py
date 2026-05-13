#!/usr/bin/env python3
"""
app/news_sentiment.py

Runtime news sentiment module for the trading engine.
Lazy-loads the model selected by sentiment_benchmark.py; falls back to lexicon scoring.
"""

import pathlib
from datetime import datetime, timezone, timedelta

_sentiment_pipe = None
_best_model     = None

_MODEL_FILE = pathlib.Path(__file__).parent / "data" / "best_sentiment_model.txt"


POSITIVE_WORDS = {
    # Earnings & revenue
    "beat", "beats", "exceeded", "exceeds", "exceed", "surpassed", "outperformed",
    "outperform", "topped", "top", "record", "records", "highest", "all-time",
    "strong", "stronger", "strongest", "robust", "solid", "stellar", "impressive",
    "better-than-expected", "above-expectations", "blowout",
    # Growth & momentum
    "surge", "surges", "surged", "soar", "soars", "soared", "spike", "spikes",
    "spiked", "rally", "rallies", "rallied", "jump", "jumps", "jumped", "climb",
    "climbs", "climbed", "rise", "rises", "rose", "gain", "gains", "gained",
    "advance", "advances", "advanced", "accelerate", "accelerates", "accelerated",
    "boom", "booms", "booming", "momentum", "breakout",
    # Guidance & outlook
    "raises", "raised", "raise", "lifted", "lifts", "lift", "increased", "increases",
    "increase", "upgraded", "upgrade", "upgrades", "initiated", "positive", "optimistic",
    "bullish", "upbeat", "constructive", "favorable", "confident", "confidence",
    "guidance-raise", "raised-guidance", "upside",
    # Deals & expansion
    "acquisition", "acquires", "acquired", "merger", "partnership", "partnerships",
    "deal", "deals", "contract", "contracts", "won", "wins", "win", "awarded",
    "expanded", "expands", "expansion", "launches", "launch", "launched", "new",
    "breakthrough", "innovation", "innovative", "patent", "approved", "approval",
    # Profitability
    "profit", "profits", "profitable", "profitability", "margin", "margins",
    "earnings", "income", "revenue", "revenues", "cash", "dividend", "dividends",
    "buyback", "buybacks", "repurchase", "yield", "return", "returns", "growth",
    "high", "upward", "up", "boost", "boosts", "boosted",
    # Analyst actions
    "overweight", "buy", "outperform", "conviction", "reiterated", "maintain",
    "target-raise", "price-target-increase",
}

NEGATIVE_WORDS = {
    # Earnings misses
    "miss", "misses", "missed", "missed-estimates", "below-expectations",
    "disappoints", "disappoint", "disappointed", "disappointing", "shortfall",
    "worse-than-expected", "weak", "weaker", "weakest", "poor", "soft",
    "sluggish", "lackluster", "underwhelming", "below",
    # Decline & momentum
    "drop", "drops", "dropped", "fall", "falls", "fell", "fallen", "decline",
    "declines", "declined", "plunge", "plunges", "plunged", "tumble", "tumbles",
    "tumbled", "crash", "crashes", "crashed", "collapse", "collapses", "collapsed",
    "slump", "slumps", "slumped", "slide", "slides", "slid", "sink", "sinks",
    "sank", "dip", "dips", "dipped", "retreat", "retreats", "retreated",
    "loss", "losses", "low", "lower", "lowest", "down", "downward",
    # Guidance cuts
    "cut", "cuts", "cutting", "reduced", "reduces", "reduce", "lowered", "lowers",
    "lower", "slashed", "slashes", "slash", "trimmed", "trims", "trim",
    "downgraded", "downgrade", "downgrades", "missed-guidance", "warned", "warning",
    "warns", "warn", "cautious", "caution", "concerned", "concerns", "worry",
    "worries", "uncertain", "uncertainty", "bearish", "negative", "pessimistic",
    # Legal & regulatory
    "lawsuit", "lawsuits", "sued", "sues", "sue", "litigation", "investigation",
    "probe", "probes", "investigated", "charges", "charged", "fine", "fined",
    "fines", "penalty", "penalties", "violation", "violations", "fraud", "scandal",
    "recall", "recalls", "recalled", "ban", "banned", "bans", "blocked", "rejected",
    "denied", "denied-approval", "failed", "fails", "fail",
    # Operational distress
    "layoffs", "layoff", "job-cuts", "restructuring", "restructure", "downsizing",
    "bankrupt", "bankruptcy", "default", "defaults", "defaulted", "debt",
    "overlevered", "dilution", "dilutive", "write-down", "writedown", "write-off",
    "impairment", "goodwill-impairment", "restatement", "restated", "restates",
    "accounting", "irregularities", "halted", "halt", "suspended", "suspension",
    # Analyst actions
    "underweight", "sell", "underperform", "avoid", "reduce", "exit",
    "price-target-cut", "target-cut", "downside",
}


def _normalize_label(label: str) -> float:
    label = label.lower()
    if any(x in label for x in ["pos", "up", "bull", "1", "optimis"]):
        return 1.0
    if any(x in label for x in ["neg", "down", "bear", "pessim"]):
        return -1.0
    return 0.0


def _lexicon_score(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return float(pos - neg) / float(total)


def _load_model():
    global _sentiment_pipe, _best_model

    if _best_model is not None:
        return  # already initialized

    if not _MODEL_FILE.exists():
        print("[NEWS] best_sentiment_model.txt not found — using lexicon fallback.")
        _best_model = "lexicon"
        _sentiment_pipe = None
        return

    model_name = _MODEL_FILE.read_text().strip()
    _best_model = model_name

    if not model_name or model_name.lower() == "lexicon":
        _sentiment_pipe = None
        print("[NEWS] Lexicon sentiment active (no model selected).")
        return

    try:
        from transformers import pipeline as hf_pipeline
        print(f"[NEWS] Loading sentiment model: {model_name} ...")
        _sentiment_pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            device=-1,
            truncation=True,
            max_length=512,
        )
        print(f"[NEWS] Sentiment model ready: {model_name}")
    except Exception as e:
        print(f"[NEWS] Model load failed ({e}), falling back to lexicon.")
        _sentiment_pipe = None
        _best_model = "lexicon"


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_news_scores(symbols: list, api, lookback_minutes: int = 480) -> dict:
    """
    Fetch recent news for symbols and return {sym: score} where score ∈ [-1.0, 1.0].
    Symbols with no articles return 0.0.
    On any failure returns {sym: 0.0 for sym in symbols}.
    """
    _load_model()

    try:
        now_utc   = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(minutes=lookback_minutes)

        raw_news = api.get_news(
            symbols=symbols,
            start=_rfc3339(start_utc),
            end=_rfc3339(now_utc),
            limit=50,
        )

        n_articles = len(raw_news) if raw_news else 0
        print(f"[NEWS] {n_articles} articles fetched for {len(symbols)} symbols")

        if not raw_news:
            return {sym: 0.0 for sym in symbols}

        headlines = [getattr(a, "headline", "") or "" for a in raw_news]

        if _sentiment_pipe is not None:
            try:
                raw_scores   = _sentiment_pipe(headlines, batch_size=32, truncation=True, max_length=512)
                article_scores = [_normalize_label(r["label"]) for r in raw_scores]
            except Exception as e:
                print(f"[NEWS] Model inference failed ({e}), falling back to lexicon.")
                article_scores = [_lexicon_score(h) for h in headlines]
        else:
            article_scores = [_lexicon_score(h) for h in headlines]

        sym_buckets: dict = {sym: [] for sym in symbols}
        for article, score in zip(raw_news, article_scores):
            for s in (getattr(article, "symbols", []) or []):
                if s in sym_buckets:
                    sym_buckets[s].append(score)

        result = {}
        for sym in symbols:
            bucket = sym_buckets[sym]
            result[sym] = float(max(-1.0, min(1.0, sum(bucket) / len(bucket)))) if bucket else 0.0

        return result

    except Exception as e:
        print(f"[NEWS] fetch failed: {e}")
        return {sym: 0.0 for sym in symbols}
