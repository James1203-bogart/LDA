"""
Evaluation Module
Computes coherence scores, perplexity, and other evaluation metrics
for the LDA topic model.
"""

from modules.topic_modeling import _compute_coherence_npmi
from collections import Counter, defaultdict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_topic_assignments(y_true, topic_ids):
    """Score LDA topic assignments against labels using leave-one-out alignment.

    LDA topic IDs are arbitrary cluster IDs, so each held-out topic is mapped
    to the most common gold label for that topic among the remaining rows.
    """
    if len(y_true) != len(topic_ids):
        raise ValueError("Gold labels and topic assignments must have the same length.")
    if len(y_true) < 2:
        raise ValueError("At least two labeled feedback rows are needed to evaluate topics.")
    if any(not label for label in y_true) or any(topic_id is None for topic_id in topic_ids):
        raise ValueError("Gold labels and topic assignments must not be empty.")

    topic_label_counts = defaultdict(Counter)
    all_label_counts = Counter(y_true)
    for label, topic_id in zip(y_true, topic_ids):
        topic_label_counts[topic_id][label] += 1

    predictions = []
    for true_label, topic_id in zip(y_true, topic_ids):
        topic_counts = topic_label_counts[topic_id].copy()
        topic_counts[true_label] -= 1
        if topic_counts[true_label] == 0:
            del topic_counts[true_label]

        candidate_counts = topic_counts
        if not candidate_counts:
            candidate_counts = all_label_counts.copy()
            candidate_counts[true_label] -= 1
            if candidate_counts[true_label] == 0:
                del candidate_counts[true_label]

        highest_count = max(candidate_counts.values())
        predictions.append(sorted(
            label for label, count in candidate_counts.items()
            if count == highest_count
        )[0])

    labels = sorted(set(y_true) | set(predictions))
    per_class = precision_recall_fscore_support(
        y_true, predictions, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, predictions, labels=labels, average='macro', zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, predictions, labels=labels, average='weighted', zero_division=0
    )
    return {
        'accuracy': round(float(accuracy_score(y_true, predictions)), 4),
        'macro_precision': round(float(macro[0]), 4),
        'macro_recall': round(float(macro[1]), 4),
        'macro_f1': round(float(macro[2]), 4),
        'weighted_precision': round(float(weighted[0]), 4),
        'weighted_recall': round(float(weighted[1]), 4),
        'weighted_f1': round(float(weighted[2]), 4),
        'labels': labels,
        'per_class': [
            {
                'label': label,
                'precision': round(float(per_class[0][index]), 4),
                'recall': round(float(per_class[1][index]), 4),
                'f1': round(float(per_class[2][index]), 4),
                'support': int(per_class[3][index]),
            }
            for index, label in enumerate(labels)
        ],
        'confusion_matrix': confusion_matrix(
            y_true, predictions, labels=labels
        ).tolist(),
    }


class ModelEvaluator:
    """Evaluates LDA model quality using standard metrics."""

    def __init__(self, model, dtm, feature_names, texts):
        self.model = model
        self.dtm = dtm
        self.feature_names = feature_names
        self.texts = texts

    def coherence_score(self):
        """Calculate NPMI-based topic coherence score."""
        overall, _ = _compute_coherence_npmi(
            self.model.components_, self.feature_names, self.texts
        )
        return round(overall, 4)

    def per_topic_coherence(self):
        """Calculate coherence for each individual topic."""
        _, per_topic = _compute_coherence_npmi(
            self.model.components_, self.feature_names, self.texts
        )
        return [round(c, 4) for c in per_topic]

    def perplexity_score(self):
        """Calculate perplexity of the model on the corpus.

        Lower perplexity indicates better predictive performance.
        """
        return round(float(self.model.perplexity(self.dtm)), 4)

    def full_evaluation(self):
        """Run all evaluation metrics and return a summary."""
        coherence = self.coherence_score()
        per_topic = self.per_topic_coherence()
        perplexity = self.perplexity_score()

        return {
            'coherence_c_v': coherence,
            'coherence_u_mass': 'N/A',
            'per_topic_coherence': per_topic,
            'perplexity': perplexity,
            'num_topics': self.model.n_components,
            'interpretation': self._interpret_scores(coherence, perplexity),
        }

    def _interpret_scores(self, coherence, perplexity):
        """Provide human-readable interpretation of scores."""
        interpretations = []

        if coherence >= 0.10:
            interpretations.append(
                "NPMI coherence score indicates good topic quality — topics are semantically meaningful."
            )
        elif coherence >= 0.0:
            interpretations.append(
                "NPMI coherence score indicates moderate topic quality — some topics may overlap."
            )
        else:
            interpretations.append(
                "NPMI coherence score is low — consider adjusting the number of topics or preprocessing."
            )

        interpretations.append(
            f"Perplexity is {perplexity:.2f}. Lower values indicate better predictive performance."
        )

        return interpretations
