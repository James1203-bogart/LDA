"""
Sentiment analysis helpers using NLTK VADER.

Provides a simple document-level sentiment analyzer that returns
compound/pos/neu/neg scores for each text.
"""

import logging
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


RATING_SENTIMENT_LABELS = {
    'negative': 'Negative',
    'neutral': 'Neutral',
    'positive': 'Positive',
}


def classify_rating(rating):
    """Classify a 1-5 evaluation rating as negative, neutral, or positive."""
    if pd.isna(rating):
        return None

    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None

    if value < 1 or value > 5:
        return None
    if value <= 2:
        return 'negative'
    if value == 3:
        return 'neutral'
    return 'positive'


def analyze_rating_constructs(dataframe, excluded_columns=None):
    """Summarize numeric 1-5 evaluation constructs.

    A construct is included when every non-empty value is within the 1-5
    rating scale. Identifier-like columns are ignored to avoid treating row
    numbers or year levels as evaluation constructs.
    """
    excluded = set(excluded_columns or [])
    constructs = {}
    for column in dataframe.columns:
        if column in excluded or _looks_like_identifier(column):
            continue

        raw_values = dataframe[column].dropna()
        values = pd.to_numeric(raw_values, errors='coerce')
        valid_values = values.dropna()
        if (valid_values.empty or len(valid_values) != len(raw_values)
            or not valid_values.between(1, 5).all()):
            continue

        labels = valid_values.map(classify_rating)
        counts = labels.value_counts().to_dict()
        mean = round(float(valid_values.mean()), 2)
        constructs[str(column)] = {
            'mean': mean,
            'classification': _classify_construct_mean(mean),
            'response_count': int(len(valid_values)),
            'counts': {
                key: int(counts.get(key, 0))
                for key in RATING_SENTIMENT_LABELS
            },
        }

    return constructs


def _looks_like_identifier(column):
    """Return whether a column name strongly suggests metadata, not a rating."""
    name = str(column).strip().lower().replace('-', '_').replace(' ', '_')
    return any(token in name.split('_') for token in ('id', 'year', 'age'))


def _classify_construct_mean(mean):
    """Classify a construct mean without making a fractional score positive."""
    if mean <= 2:
        return 'negative'
    if mean < 4:
        return 'neutral'
    return 'positive'


def _ensure_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception as exc:
            logger.warning('Could not download vader_lexicon: %s', exc)


def analyze_sentiments(texts):
    """Return a list of sentiment dicts (compound, pos, neu, neg) for texts.

    texts: iterable of strings
    returns: list of dicts matching SentimentIntensityAnalyzer.polarity_scores
    """
    _ensure_vader()
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        logger.warning('Failed to initialize SentimentIntensityAnalyzer: %s', e)
        return [None] * len(texts)

    results = []
    for t in texts:
        try:
            scores = sia.polarity_scores(t if t is not None else '')
            results.append(scores)
        except Exception as exc:
            logger.warning('Sentiment analysis failed for text: %s', exc)
            results.append(None)

    return results
