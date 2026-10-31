# LDA
# LDA
# LDA

## Enhanced LDA evaluation

The Compare page can calculate accuracy, macro/weighted precision, recall, and
F1 for the LDA model's dominant-topic assignments. Upload a manual-coding file,
select the feedback-text and manual-theme columns, and match the feedback rows
to the analyzed comments.

Because LDA topic IDs are arbitrary, the evaluation aligns each topic to the
most common manual theme from the other matched rows (leave-one-out), then
scores the held-out row. This uses the model's actual topic assignments and
does not use reviewer-entered topic names as predictions. Coherence and
perplexity remain separate unsupervised topic-quality metrics.
