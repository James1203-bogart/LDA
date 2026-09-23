"""
Topic Modeling Module
Implements Latent Dirichlet Allocation (LDA) using scikit-learn.
"""

import logging
import numpy as np
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config import (
    DEFAULT_NUM_TOPICS, LDA_ITERATIONS, RANDOM_STATE,
    MIN_TOPICS, MAX_TOPICS
)

logger = logging.getLogger(__name__)


def _compute_coherence_npmi(components, feature_names, texts, top_n=10):
    """Compute NPMI-based coherence score for topics.

    For each topic, takes the top_n words and computes the average
    normalized pointwise mutual information among all word pairs,
    using co-occurrence within documents.
    """
    doc_word_sets = [set(doc) for doc in texts]
    n_docs = len(doc_word_sets)
    if n_docs == 0:
        return 0.0, [0.0] * components.shape[0]

    word_doc_freq = Counter()
    pair_doc_freq = Counter()

    all_top_words = set()
    topics_top_words = []
    for topic_idx in range(components.shape[0]):
        top_indices = components[topic_idx].argsort()[:-top_n - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_top_words.append(top_words)
        all_top_words.update(top_words)

    for doc_words in doc_word_sets:
        present = doc_words & all_top_words
        for w in present:
            word_doc_freq[w] += 1
        present_list = sorted(present)
        for i in range(len(present_list)):
            for j in range(i + 1, len(present_list)):
                pair_doc_freq[(present_list[i], present_list[j])] += 1

    eps = 1e-12
    per_topic_coherence = []

    for top_words in topics_top_words:
        npmi_scores = []
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                w1, w2 = sorted([top_words[i], top_words[j]])
                p_w1 = word_doc_freq.get(w1, 0) / n_docs
                p_w2 = word_doc_freq.get(w2, 0) / n_docs
                p_pair = pair_doc_freq.get((w1, w2), 0) / n_docs

                if p_pair < eps or p_w1 < eps or p_w2 < eps:
                    npmi = -1.0
                else:
                    pmi = np.log(p_pair / (p_w1 * p_w2 + eps) + eps)
                    npmi = pmi / (-np.log(p_pair + eps))
                npmi_scores.append(npmi)

        per_topic_coherence.append(float(np.mean(npmi_scores)) if npmi_scores else 0.0)

    overall = float(np.mean(per_topic_coherence)) if per_topic_coherence else 0.0
    return overall, per_topic_coherence


class LDATopicModeler:
    """Implements LDA topic modeling on preprocessed student feedback."""

    def __init__(self):
        self.vectorizer = None
        self.dtm = None
        self.model = None
        self.num_topics = None
        self.processed_texts = None
        self.feature_names = None

    def build_corpus(self, processed_texts, min_df=2, max_df=0.5):
        """Build document-term matrix from preprocessed tokens."""
        self.processed_texts = processed_texts
        text_strings = [' '.join(doc) for doc in processed_texts]

        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'(?u)\b\w+\b',
        )
        self.dtm = self.vectorizer.fit_transform(text_strings)
        self.feature_names = list(self.vectorizer.get_feature_names_out())

        return {
            'dictionary_size': len(self.feature_names),
            'corpus_size': self.dtm.shape[0],
            'total_word_occurrences': int(self.dtm.sum()),
        }

    def train_model(self, num_topics=None, passes=None, iterations=None):
        """Train the LDA model."""
        if self.dtm is None:
            raise ValueError("Corpus not built. Call build_corpus() first.")

        self.num_topics = num_topics or DEFAULT_NUM_TOPICS
        max_iter = iterations or LDA_ITERATIONS

        self.model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=RANDOM_STATE,
            max_iter=max_iter,
            learning_method='online',
            n_jobs=-1,
        )
        self.model.fit(self.dtm)
        return self.get_topics()

    def get_topics(self, num_words=10):
        """Get topics with their top words and weights."""
        if self.model is None:
            raise ValueError("Model not trained.")

        topics = []
        for idx in range(self.num_topics):
            topic_dist = self.model.components_[idx]
            topic_dist = topic_dist / topic_dist.sum()
            top_indices = topic_dist.argsort()[:-num_words - 1:-1]

            words = [self.feature_names[i] for i in top_indices]
            weights = [round(float(topic_dist[i]), 4) for i in top_indices]

            topics.append({
                'topic_id': idx,
                'words': words,
                'weights': weights,
                'word_weight_pairs': list(zip(words, weights)),
            })
        return topics

    def get_document_topics(self):
        """Get topic distribution for each document."""
        if self.model is None:
            raise ValueError("Model not trained.")

        doc_topic_dist = self.model.transform(self.dtm)
        return [
            [(tid, round(float(prob), 4)) for tid, prob in enumerate(dist)]
            for dist in doc_topic_dist
        ]

    def get_dominant_topics(self):
        """Get the dominant topic for each document."""
        doc_topic_dist = self.model.transform(self.dtm)
        return [
            {
                'document_index': i,
                'dominant_topic': int(np.argmax(dist)),
                'probability': round(float(dist[np.argmax(dist)]), 4),
            }
            for i, dist in enumerate(doc_topic_dist)
        ]

    def get_topic_distribution_summary(self):
        """Get summary of how documents are distributed across topics."""
        dominant = self.get_dominant_topics()
        distribution = Counter(d['dominant_topic'] for d in dominant)
        total = len(dominant)
        return [
            {
                'topic_id': tid,
                'document_count': distribution.get(tid, 0),
                'percentage': round(distribution.get(tid, 0) / max(total, 1) * 100, 2),
            }
            for tid in range(self.num_topics)
        ]

    def find_optimal_topics(self, min_topics=None, max_topics=None, step=1):
        """Find optimal number of topics by evaluating coherence and perplexity."""
        if self.dtm is None:
            raise ValueError("Corpus not built.")

        min_t = min_topics or MIN_TOPICS
        max_t = max_topics or MAX_TOPICS

        results = []
        for k in range(min_t, max_t + 1, step):
            model = LatentDirichletAllocation(
                n_components=k,
                random_state=RANDOM_STATE,
                max_iter=LDA_ITERATIONS,
                learning_method='online',
                n_jobs=-1,
            )
            model.fit(self.dtm)

            perplexity = model.perplexity(self.dtm)
            coherence, _ = _compute_coherence_npmi(
                model.components_, self.feature_names, self.processed_texts
            )

            results.append({
                'num_topics': k,
                'coherence_score': round(float(coherence), 4),
                'perplexity_score': round(float(perplexity), 4),
            })
            logger.info(f"k={k}: coherence={coherence:.4f}, perplexity={perplexity:.4f}")

        best = max(results, key=lambda x: x['coherence_score'])
        return {
            'results': results,
            'optimal_k': best['num_topics'],
            'best_coherence': best['coherence_score'],
        }
