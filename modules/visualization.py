"""
Visualization Module
Generates word clouds, topic charts, and distribution visualizations.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def generate_wordcloud(word_weights, title="Topic", width=800, height=400):
    """Generate a word cloud image as base64 string."""
    freq_dict = {word: weight for word, weight in word_weights}
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap='viridis',
        max_words=50,
        prefer_horizontal=0.7,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_topic_bar_chart(topics, num_words=10):
    """Generate horizontal bar charts of top words per topic."""
    n_topics = len(topics)
    cols = min(n_topics, 3)
    rows = (n_topics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_topics == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, topic in enumerate(topics):
        words = topic['words'][:num_words][::-1]
        weights = topic['weights'][:num_words][::-1]
        axes[i].barh(words, weights, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(words))))
        axes[i].set_title(f"Topic {topic['topic_id']}", fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Weight')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_distribution_pie_chart(distribution_summary):
    """Generate a pie chart of document distribution across topics."""
    labels = [f"Topic {d['topic_id']}" for d in distribution_summary]
    sizes = [d['document_count'] for d in distribution_summary]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.85,
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title('Document Distribution Across Topics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_coherence_line_chart(optimization_results):
    """Generate a line chart showing coherence vs number of topics."""
    results = optimization_results['results']
    ks = [r['num_topics'] for r in results]
    coherences = [r['coherence_score'] for r in results]
    perplexities = [r['perplexity_score'] for r in results]
    optimal_k = optimization_results['optimal_k']

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    color1 = '#2196F3'
    ax1.set_xlabel('Number of Topics (k)', fontsize=12)
    ax1.set_ylabel('NPMI Coherence Score', color=color1, fontsize=12)
    ax1.plot(ks, coherences, 'o-', color=color1, linewidth=2, markersize=8, label='Coherence')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')

    color2 = '#FF9800'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Perplexity', color=color2, fontsize=12)
    ax2.plot(ks, perplexities, 's--', color=color2, linewidth=2, markersize=8, label='Perplexity')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Topic Number Optimization', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_per_topic_coherence_bar(per_topic_coherence):
    """Generate a bar chart of coherence per topic."""
    labels = [f"Topic {i}" for i in range(len(per_topic_coherence))]
    colors = ['#4CAF50' if c >= 0.1 else '#FF9800' if c >= 0.0 else '#F44336'
              for c in per_topic_coherence]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(labels, per_topic_coherence, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('NPMI Coherence Score', fontsize=12)
    ax.set_title('Per-Topic Coherence Scores', fontsize=14, fontweight='bold')
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (>=0.1)')
    ax.axhline(y=0.0, color='orange', linestyle='--', alpha=0.5, label='Moderate (>=0.0)')
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
