"""
Evaluation Module
Computes coherence scores, perplexity, and other evaluation metrics
for the LDA topic model.
"""

from modules.topic_modeling import _compute_coherence_npmi


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
