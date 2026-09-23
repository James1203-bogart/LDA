"""
Text Preprocessing Module
Handles cleaning, tokenization, stop-word removal, and lemmatization
of student feedback text data.
"""

import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


def _ensure_nltk_resources():
    """Download required NLTK resources if they are missing, without crashing."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as exc:
                logger.warning('Could not download NLTK resource %s: %s', package, exc)


_ensure_nltk_resources()

# Custom stop words relevant to student feedback surveys
CUSTOM_STOP_WORDS = {
    'also', 'would', 'could', 'one', 'like', 'get', 'make', 'much',
    'really', 'thing', 'even', 'well', 'many', 'us', 'go', 'na',
    'im', 'dont', 'cant', 'didnt', 'doesnt', 'isnt', 'wasnt',
    'havent', 'hasnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
    'etc', 'eg', 'ie', 'vs', 'shall', 'upon', 'thus', 'hence',
    'thereby', 'thereafter', 'whereas', 'whereby', 'wherein',
}


class TextPreprocessor:
    """Preprocesses student feedback text for LDA topic modeling."""

    def __init__(self, extra_stop_words=None):
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as exc:
            logger.warning('Could not initialize WordNet lemmatizer: %s', exc)
            self.lemmatizer = None
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as exc:
            logger.warning('Could not load English stopwords: %s', exc)
            self.stop_words = set()
        self.stop_words.update(CUSTOM_STOP_WORDS)
        if extra_stop_words:
            self.stop_words.update(extra_stop_words)

    def clean_text(self, text):
        """Remove special characters, numbers, URLs, emails, and extra whitespace."""
        if not isinstance(text, str):
            return ''
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        """Tokenize text into individual words."""
        try:
            return word_tokenize(text)
        except Exception:
            return re.findall(r"[a-zA-Z']+", text.lower())

    def remove_stopwords(self, tokens):
        """Remove stop words from token list."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def lemmatize(self, tokens):
        """Lemmatize tokens to their base forms."""
        if not self.lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text):
        """Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

    def preprocess_corpus(self, texts):
        """Preprocess a list of text documents."""
        return [self.preprocess(text) for text in texts]

    def get_preprocessing_stats(self, original_texts, processed_texts):
        """Return statistics about the preprocessing step."""
        total_docs = len(original_texts)
        empty_after = sum(1 for t in processed_texts if len(t) == 0)
        avg_tokens_before = sum(len(str(t).split()) for t in original_texts) / max(total_docs, 1)
        avg_tokens_after = sum(len(t) for t in processed_texts) / max(total_docs, 1)

        all_tokens = [token for doc in processed_texts for token in doc]
        vocab_size = len(set(all_tokens))

        return {
            'total_documents': total_docs,
            'empty_documents_after_preprocessing': empty_after,
            'avg_words_before': round(avg_tokens_before, 2),
            'avg_tokens_after': round(avg_tokens_after, 2),
            'vocabulary_size': vocab_size,
            'total_tokens': len(all_tokens),
        }
