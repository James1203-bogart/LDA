"""
Student Feedback Topic Modeling System
Flask Web Application
Using Latent Dirichlet Allocation (LDA)
"""

import os
import io
import json
import logging
import re
import uuid
import pickle
from datetime import datetime
import pandas as pd
import requests
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_file
)
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS,
    SECRET_KEY, MAX_CONTENT_LENGTH, DEFAULT_NUM_TOPICS,
    MIN_TOPICS, MAX_TOPICS, MAX_DOCUMENTS
)
from modules.preprocessing import TextPreprocessor
from modules.topic_modeling import LDATopicModeler
from modules.pretrained_model import PretrainedFeedbackModel
from modules.evaluation import ModelEvaluator
from modules.visualization import (
    generate_topic_bar_chart,
    generate_distribution_pie_chart, generate_coherence_line_chart,
    generate_per_topic_coherence_bar
)
from modules.sentiment import analyze_sentiments, analyze_rating_constructs
from modules.pretrained_model import PRETRAINED_CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('VERCEL') == '1'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Ensure directories exist. Vercel serverless functions are read-only for the app bundle,
# so fall back to /tmp when the project directory is not writable.
for directory in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        pass

# Persist analysis state so links keep working across restarts and Render instances.
if os.environ.get('VERCEL') == '1':
    ANALYSIS_STORE_DIR = '/tmp/analysis_store'
else:
    ANALYSIS_STORE_DIR = os.path.join(os.path.dirname(__file__), 'analysis_store')

try:
    os.makedirs(ANALYSIS_STORE_DIR, exist_ok=True)
except OSError:
    ANALYSIS_STORE_DIR = '/tmp'

analysis_store = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _analysis_path(analysis_id):
    return os.path.join(ANALYSIS_STORE_DIR, f'{analysis_id}.pkl')


def get_analysis(analysis_id):
    if analysis_id in analysis_store:
        return analysis_store[analysis_id]

    path = _analysis_path(analysis_id)
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'rb') as fh:
            data = pickle.load(fh)
        analysis_store[analysis_id] = data
        return data
    except Exception as e:
        logger.warning('Failed to load analysis %s: %s', analysis_id, e)
        return None


def set_analysis(analysis_id, analysis_data):
    analysis_store[analysis_id] = analysis_data
    os.makedirs(ANALYSIS_STORE_DIR, exist_ok=True)
    try:
        with open(_analysis_path(analysis_id), 'wb') as fh:
            pickle.dump(analysis_data, fh)
    except Exception as e:
        logger.warning('Failed to persist analysis %s: %s', analysis_id, e)


def get_completed_analyses():
    analysis_ids = set(analysis_store)
    try:
        analysis_ids.update(
            entry.name[:-4]
            for entry in os.scandir(ANALYSIS_STORE_DIR)
            if entry.is_file() and entry.name.endswith('.pkl')
        )
    except OSError:
        pass

    completed = []
    for analysis_id in analysis_ids:
        analysis = get_analysis(analysis_id)
        if not analysis or not analysis.get('topics') or not analysis.get('distribution'):
            continue

        try:
            modified_at = os.path.getmtime(_analysis_path(analysis_id))
        except OSError:
            modified_at = 0

        completed.append({
            'analysis_id': analysis_id,
            'filename': analysis.get('filename') or 'Untitled dataset',
            'row_count': analysis.get('row_count') or analysis.get('text_count', 0),
            'topic_count': len(analysis['topics']),
            'completed_at': datetime.fromtimestamp(modified_at) if modified_at else None,
            'sort_key': modified_at,
        })

    return sorted(completed, key=lambda item: item['sort_key'], reverse=True)


def get_dataframe_from_analysis(analysis):
    return analysis.get('dataframe')


def extract_google_sheet_id(url):
    """Extract the spreadsheet ID from a Google Sheets or Google Forms URL."""
    m = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    return None


# ─── Routes ──────────────────────────────────────────────────────────

@app.route('/results')
def results_list():
    return render_template('results_list.html', analyses=get_completed_analyses())


@app.route('/')
def index():
    """Dashboard / Home page."""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload student feedback dataset (file or Google Sheets URL)."""
    if request.method == 'POST':
        upload_mode = request.form.get('upload_mode', 'file')

        df = None
        filename = None
        filepath = None

        # ── Google Sheets / Google Forms import ──────────────────
        if upload_mode == 'google':
            sheet_url = request.form.get('google_url', '').strip()
            if not sheet_url:
                flash('Please paste a Google Sheets URL.', 'danger')
                return redirect(request.url)

            sheet_id = extract_google_sheet_id(sheet_url)
            if not sheet_id:
                flash('Invalid Google Sheets URL. Please paste a URL like '
                      'https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/...', 'danger')
                return redirect(request.url)

            # Optionally extract gid (sheet tab)
            gid_match = re.search(r'[?&#]gid=(\d+)', sheet_url)
            gid = gid_match.group(1) if gid_match else '0'

            export_url = (
                f'https://docs.google.com/spreadsheets/d/{sheet_id}'
                f'/export?format=csv&gid={gid}'
            )

            try:
                resp = requests.get(export_url, timeout=30)
                resp.raise_for_status()
                # Google returns text/html for private sheets
                ct = resp.headers.get('Content-Type', '')
                if 'text/csv' not in ct and 'application/csv' not in ct:
                    flash('Could not fetch the spreadsheet. Make sure it is shared as '
                          '"Anyone with the link" can view.', 'danger')
                    return redirect(request.url)

                df = pd.read_csv(io.StringIO(resp.text))
                filename = f'google_sheet_{sheet_id[:8]}.csv'

            except requests.RequestException as e:
                logger.error('Google Sheets fetch error: %s', e)
                flash('Failed to fetch from Google Sheets. Check the URL and make sure '
                      'the sheet is publicly shared.', 'danger')
                return redirect(request.url)
            except Exception as e:
                logger.error('Google Sheets parse error: %s', e)
                flash(f'Error reading spreadsheet data: {str(e)}', 'danger')
                return redirect(request.url)

        # ── Local file upload ────────────────────────────────────
        else:
            if 'file' not in request.files:
                flash('No file selected.', 'danger')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No file selected.', 'danger')
                return redirect(request.url)

            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload CSV or Excel files.', 'danger')
                return redirect(request.url)

            filename = secure_filename(file.filename)
            
            # Read file directly into memory (for Render.com compatibility)
            try:
                file_stream = file.stream
                file_stream.seek(0)
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_stream)
                else:
                    df = pd.read_excel(file_stream)
            except Exception as e:
                logger.error(f'File read error: {str(e)}')
                flash(f'Error reading file: {str(e)}', 'danger')
                return redirect(request.url)

        # ── Common: store analysis ───────────────────────────────
        if df is None or df.empty:
            flash('The dataset is empty. Please try a different source.', 'danger')
            return redirect(request.url)

        analysis_id = uuid.uuid4().hex
        analysis_data = {
            'filename': filename,
            'dataframe': df,
            'columns': list(df.columns),
            'row_count': len(df),
        }
        set_analysis(analysis_id, analysis_data)

        source_label = 'Google Sheets' if upload_mode == 'google' else 'file'
        flash(f'Dataset loaded from {source_label} successfully! ({len(df)} rows)', 'success')
        return redirect(url_for('select_column', analysis_id=analysis_id))

    return render_template('upload.html')


@app.route('/select-column/<analysis_id>', methods=['GET', 'POST'])
def select_column(analysis_id):
    """Select which column contains the feedback text."""
    analysis = get_analysis(analysis_id)
    if not analysis:
        flash('Analysis not found. Please upload a file first.', 'danger')
        return redirect(url_for('upload'))

    df = get_dataframe_from_analysis(analysis)
    if df is None:
        flash('Invalid analysis data.', 'danger')
        return redirect(url_for('upload'))

    if request.method == 'POST':
        feedback_column = request.form.get('feedback_column')
        if feedback_column not in df.columns:
            flash('Invalid column selected.', 'danger')
            return redirect(request.url)

        analysis['feedback_column'] = feedback_column
        analysis['rating_constructs'] = analyze_rating_constructs(
            df, excluded_columns={feedback_column}
        )

        # Extract texts
        texts = df[feedback_column].dropna().astype(str).tolist()
        original_text_count = len(texts)
        if original_text_count > MAX_DOCUMENTS:
            texts = texts[:MAX_DOCUMENTS]
            flash(
                f'Dataset has {original_text_count} non-empty entries. '
                f'Only the first {MAX_DOCUMENTS} documents will be processed.',
                'warning'
            )

        analysis['raw_texts'] = texts
        analysis['text_count'] = len(texts)
        analysis['original_text_count'] = original_text_count

        # Preview
        analysis['preview'] = texts[:10]
        
        # Persist to session
        set_analysis(analysis_id, analysis)

        flash(f'Column "{feedback_column}" selected. {len(texts)} feedback entries found.', 'success')
        return redirect(url_for('preprocess', analysis_id=analysis_id))

    # Show column selection with previews
    column_previews = {}
    for col in df.columns:
        sample_vals = df[col].dropna().head(3).astype(str).tolist()
        column_previews[col] = sample_vals

    return render_template(
        'select_column.html',
        analysis_id=analysis_id,
        columns=df.columns.tolist(),
        column_previews=column_previews,
        filename=analysis['filename'],
        row_count=analysis['row_count'],
    )


@app.route('/preprocess/<analysis_id>', methods=['GET', 'POST'])
def preprocess(analysis_id):
    """Preprocess the selected feedback text."""
    analysis = get_analysis(analysis_id)
    if not analysis or 'raw_texts' not in analysis:
        flash('Please upload a file and select a column first.', 'danger')
        return redirect(url_for('upload'))

    if request.method == 'POST':
        # Get optional extra stop words from user
        extra_sw = request.form.get('extra_stopwords', '').strip()
        extra_stop_words = set()
        if extra_sw:
            extra_stop_words = {w.strip().lower() for w in extra_sw.split(',') if w.strip()}

        try:
            preprocessor = TextPreprocessor(extra_stop_words=extra_stop_words)
            processed = preprocessor.preprocess_corpus(analysis['raw_texts'])

            # Remove empty documents
            raw_and_processed = [
                (raw, proc) for raw, proc in zip(analysis['raw_texts'], processed)
                if len(proc) > 0
            ]
            if not raw_and_processed:
                flash('All documents were empty after preprocessing. Try different settings.', 'danger')
                return redirect(request.url)

            analysis['filtered_raw'] = [r for r, _ in raw_and_processed]
            analysis['processed_texts'] = [p for _, p in raw_and_processed]
            analysis['preprocessing_stats'] = preprocessor.get_preprocessing_stats(
                analysis['raw_texts'], processed
            )
            analysis['extra_stopwords'] = list(extra_stop_words)

            # Preview of preprocessing
            analysis['preprocess_preview'] = [
                {
                    'original': raw,
                    'tokens': ' '.join(proc),
                }
                for raw, proc in list(raw_and_processed)[:5]
            ]
            # Run document-level sentiment analysis on the filtered raw texts.
            try:
                analysis['sentiments'] = analyze_sentiments(analysis['filtered_raw'])
            except Exception as e:
                logger.warning('Sentiment analysis during preprocess failed: %s', e)
                analysis['sentiments'] = []
            
            # Persist preprocessing results to session
            set_analysis(analysis_id, analysis)
            
        except Exception as e:
            logger.exception('Preprocessing error: %s', e)
            flash(f'An error occurred during text preprocessing: {str(e)}', 'danger')
            return redirect(request.url)

        flash('Text preprocessing completed successfully!', 'success')
        return redirect(url_for('model_config', analysis_id=analysis_id))

    return render_template(
        'preprocess.html',
        analysis_id=analysis_id,
        text_count=analysis['text_count'],
        preview=analysis.get('preview', []),
    )


@app.route('/model-config/<analysis_id>', methods=['GET', 'POST'])
def model_config(analysis_id):
    """Apply the bundled classifier and generate supporting topic details."""
    analysis = get_analysis(analysis_id)
    if not analysis or 'processed_texts' not in analysis:
        flash('Please complete preprocessing first.', 'danger')
        return redirect(url_for('upload'))

    if request.method == 'POST':
        try:
            classifier = PretrainedFeedbackModel()
            analysis['pretrained_predictions'] = classifier.classify_documents(
                analysis.get('filtered_raw', analysis.get('raw_texts', []))
            )
            analysis['category_summary'] = classifier.summarize(analysis['pretrained_predictions'])

            predefined_topics = classifier.build_predefined_topics()
            analysis['topics'] = predefined_topics
            analysis['num_topics'] = len(predefined_topics)

            category_order = list(PRETRAINED_CATEGORIES.keys())
            category_counts = {category: 0 for category in category_order}
            for prediction in analysis['pretrained_predictions']:
                category = prediction.get('category')
                if category in category_counts:
                    category_counts[category] += 1

            analysis['distribution'] = [
                {
                    'topic_id': idx,
                    'document_count': category_counts.get(category, 0),
                    'percentage': round(
                        category_counts.get(category, 0) / max(len(analysis['pretrained_predictions']), 1) * 100,
                        2,
                    ),
                }
                for idx, category in enumerate(category_order, start=1)
            ]
            analysis['dominant_topics'] = [
                {
                    'document_index': idx,
                    'dominant_topic': category_order.index(pred['category']) + 1,
                    'probability': pred['confidence'],
                    'topic': category_order.index(pred['category']) + 1,
                }
                for idx, pred in enumerate(analysis['pretrained_predictions'])
            ]
            analysis['predefined_results'] = True
            analysis['evaluation'] = {
                'coherence_c_v': 'N/A',
                'coherence_u_mass': 'N/A',
                'perplexity': 'N/A',
                'interpretation': [
                    'This feedback model uses a predefined student-feedback taxonomy instead of a data-driven LDA model.',
                    'The system groups responses into 3 main categories and their subtopics for management review. '
                    'Rating-based sentiment is computed separately from the numeric feedback items.'
                ],
            }

            try:
                if analysis.get('sentiments'):
                    from collections import defaultdict
                    topic_sents = defaultdict(list)
                    for idx, pred in enumerate(analysis['pretrained_predictions']):
                        if idx >= len(analysis['sentiments']):
                            continue
                        score = analysis['sentiments'][idx]
                        if score is None:
                            continue
                        category_idx = category_order.index(pred['category']) + 1
                        topic_sents[category_idx].append(score.get('compound', 0.0))
                    analysis['per_topic_sentiment'] = {
                        topic_id: round(sum(values) / len(values), 4) if values else None
                        for topic_id, values in topic_sents.items()
                    }
            except Exception as e:
                logger.warning('Failed to compute per-topic sentiment: %s', e)

            set_analysis(analysis_id, analysis)
            flash('Pre-defined student feedback taxonomy applied with 3 categories and subtopics.', 'success')
            return redirect(url_for('results', analysis_id=analysis_id))

        except ValueError as e:
            logger.error('Invalid input in model config: %s', e)
            flash(f'Invalid input: {str(e)}', 'danger')
            return redirect(request.url)
        except Exception as e:
            logger.error('Error training LDA model: %s', e)
            flash(f'Error training model: {str(e)}. Please try again with different settings.', 'danger')
            return redirect(request.url)

    stats = analysis.get('preprocessing_stats') or {
        'total_documents': analysis.get('text_count', 0),
        'empty_documents_after_preprocessing': 0,
        'avg_words_before': 0,
        'avg_tokens_after': 0,
        'vocabulary_size': 0,
        'total_tokens': 0,
    }

    return render_template(
        'model_config.html',
        analysis_id=analysis_id,
        stats=stats,
        preview=analysis.get('preprocess_preview', []),
        default_topics=3,
        min_topics=3,
        max_topics=3,
    )


@app.route('/results/<analysis_id>')
def results(analysis_id):
    """Display topic modeling results and visualizations."""
    analysis = get_analysis(analysis_id)
    if not analysis or not analysis.get('topics') or not analysis.get('distribution'):
        flash('No results available. Please run the model first.', 'danger')
        return redirect(url_for('upload'))

    topics = analysis.get('topics', [])
    distribution = analysis.get('distribution', [])
    evaluation = analysis.get('evaluation', {})

    if not topics or not distribution:
        flash('No results available. Please run the model first.', 'danger')
        return redirect(url_for('upload'))

    # Generate visualizations
    bar_chart = generate_topic_bar_chart(topics)
    pie_chart = generate_distribution_pie_chart(distribution)

    per_topic_chart = None
    if 'evaluation' in analysis and 'per_topic_coherence' in analysis['evaluation']:
        per_topic_chart = generate_per_topic_coherence_bar(analysis['evaluation']['per_topic_coherence'])

    optimization_chart = None
    if 'optimization' in analysis:
        optimization_chart = generate_coherence_line_chart(analysis['optimization'])

    # Sentiment summary
    sentiments = analysis.get('sentiments', [])
    rating_constructs = analysis.get('rating_constructs', {})
    sentiment_summary = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
    for s in sentiments:
        if not s:
            continue
        sentiment_summary['total'] += 1
        c = s.get('compound', 0.0)
        if c >= 0.05:
            sentiment_summary['positive'] += 1
        elif c <= -0.05:
            sentiment_summary['negative'] += 1
        else:
            sentiment_summary['neutral'] += 1

    # Document-topic table for preview
    dominant = analysis.get('dominant_topics', [])
    doc_topic_preview = []
    for d in dominant[:20]:
        idx = d.get('document_index')
        if idx is None:
            idx = d.get('index')
        sentiment_val = None
        try:
            if idx is not None and idx < len(sentiments) and sentiments[idx]:
                sentiment_val = sentiments[idx].get('compound')
        except Exception:
            sentiment_val = None

        preview_text = ''
        if isinstance(idx, int) and 0 <= idx < len(analysis['filtered_raw']):
            raw = analysis['filtered_raw'][idx]
            preview_text = raw[:150] + ('...' if len(raw) > 150 else '')

        doc_topic_preview.append({
            'index': idx,
            'text': preview_text,
            'topic': d.get('dominant_topic') or d.get('topic'),
            'probability': d.get('probability'),
            'sentiment': sentiment_val,
        })

    per_topic_sentiment = analysis.get('per_topic_sentiment', {})

    # Group ALL documents (not just the top-20 preview) by topic and by
    # sentiment bucket, so we can show "who said what" per topic/mood.
    MAX_PER_BUCKET = 30  # cap so the page doesn't balloon on huge datasets
    topic_sentiment_docs = {}
    for d in dominant:
        idx = d.get('document_index')
        if idx is None:
            idx = d.get('index')
        topic_id = d.get('dominant_topic')
        if topic_id is None:
            topic_id = d.get('topic')
        if idx is None or topic_id is None:
            continue
        if not isinstance(idx, int) or idx < 0 or idx >= len(analysis['filtered_raw']):
            continue

        sc = None
        try:
            if idx < len(sentiments) and sentiments[idx]:
                sc = sentiments[idx].get('compound')
        except Exception:
            sc = None
        if sc is None:
            continue

        if sc >= 0.05:
            bucket = 'positive'
        elif sc <= -0.05:
            bucket = 'negative'
        else:
            bucket = 'neutral'

        bucket_map = topic_sentiment_docs.setdefault(
            topic_id, {'positive': [], 'neutral': [], 'negative': []}
        )
        entries = bucket_map[bucket]
        raw = analysis['filtered_raw'][idx]
        entries.append({
            'index': idx,
            'text': raw[:200] + ('...' if len(raw) > 200 else ''),
            'score': sc,
        })

    # Keep counts before trimming, then trim each bucket's display list
    topic_sentiment_counts = {
        t: {k: len(v) for k, v in buckets.items()}
        for t, buckets in topic_sentiment_docs.items()
    }
    for buckets in topic_sentiment_docs.values():
        for k in buckets:
            buckets[k] = buckets[k][:MAX_PER_BUCKET]

    return render_template(
        'results.html',
        analysis_id=analysis_id,
        topics=topics,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        per_topic_chart=per_topic_chart,
        optimization_chart=optimization_chart,
        evaluation=analysis.get('evaluation', {}),
        distribution=analysis.get('distribution', []),
        optimization=analysis.get('optimization'),
        doc_topic_preview=doc_topic_preview,
        num_topics=analysis['num_topics'],
        preprocessing_stats=analysis['preprocessing_stats'],
        per_topic_sentiment=per_topic_sentiment,
        sentiment_summary=sentiment_summary,
        rating_constructs=rating_constructs,
        category_summary=analysis.get('category_summary', []),
        pretrained_predictions=analysis.get('pretrained_predictions', []),
        topic_sentiment_docs=topic_sentiment_docs,
        topic_sentiment_counts=topic_sentiment_counts,
    )


@app.route('/compare/<analysis_id>', methods=['GET', 'POST'])
def compare(analysis_id):
    """Compare manual coding and collect human topic evaluations."""
    analysis = get_analysis(analysis_id)
    if not analysis or 'topics' not in analysis:
        flash('Please run the model first.', 'danger')
        return redirect(url_for('upload'))

    if request.method == 'POST':
        if request.form.get('action') == 'save_human_review':
            assessments = {'accurate', 'partially_accurate', 'inaccurate'}
            reviewer = request.form.get('reviewer', '').strip()
            reviews = {}
            if not reviewer:
                flash('Enter the reviewer name or role before saving evaluations.', 'danger')
                return redirect(request.url)

            for topic in analysis['topics']:
                topic_id = topic['topic_id']
                field_suffix = str(topic_id)
                label = request.form.get(f'label_{field_suffix}', '').strip()
                assessment = request.form.get(f'assessment_{field_suffix}', '')
                notes = request.form.get(f'notes_{field_suffix}', '').strip()
                if not label or assessment not in assessments:
                    flash('Provide a topic label and valid accuracy assessment for every topic.', 'danger')
                    return redirect(request.url)
                reviews[str(topic_id)] = {
                    'label': label,
                    'assessment': assessment,
                    'notes': notes,
                    'reviewer': reviewer,
                    'reviewed_at': datetime.now().isoformat(timespec='seconds'),
                }

            analysis['human_review'] = {
                'reviewer': reviewer,
                'topics': reviews,
            }
            set_analysis(analysis_id, analysis)
            flash('Human topic evaluations saved.', 'success')
            return redirect(request.url)

        if 'manual_file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['manual_file']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Invalid file. Upload CSV or Excel with manual coding.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)

        try:
            file_stream = file.stream
            file_stream.seek(0)
            if filename.endswith('.csv'):
                manual_df = pd.read_csv(file_stream)
            else:
                manual_df = pd.read_excel(file_stream)
        except Exception as e:
            logger.error(f'Manual file read error: {str(e)}')
            flash(f'Error reading file: {e}', 'danger')
            return redirect(request.url)

        analysis['manual_df'] = manual_df
        analysis['manual_columns'] = list(manual_df.columns)
        set_analysis(analysis_id, analysis)
        flash('Manual coding file uploaded.', 'success')

    examples_by_topic = {}
    for document in analysis.get('dominant_topics', []):
        topic_id = document.get('dominant_topic')
        if topic_id is None:
            topic_id = document.get('topic')
        index = document.get('document_index', document.get('index'))
        raw_texts = analysis.get('filtered_raw', [])
        if topic_id is None or not isinstance(index, int) or not 0 <= index < len(raw_texts):
            continue
        text = str(raw_texts[index])
        examples_by_topic.setdefault(topic_id, []).append({
            'text': text[:500] + ('...' if len(text) > 500 else ''),
            'probability': document.get('probability', 0),
        })

    human_review = analysis.get('human_review', {})
    human_review_topics = []
    for topic in analysis['topics']:
        topic_id = topic['topic_id']
        examples = sorted(
            examples_by_topic.get(topic_id, []),
            key=lambda example: example['probability'] or 0,
            reverse=True,
        )[:3]
        human_review_topics.append({
            'topic': topic,
            'examples': examples,
            'review': human_review.get('topics', {}).get(str(topic_id), {}),
        })

    return render_template(
        'compare.html',
        analysis_id=analysis_id,
        topics=analysis['topics'],
        has_manual='manual_df' in analysis,
        manual_columns=analysis.get('manual_columns', []),
        human_review_topics=human_review_topics,
        human_reviewer=human_review.get('reviewer', ''),
    )


@app.route('/export/<analysis_id>/<format_type>')
def export_report(analysis_id, format_type):
    """Export results report."""
    analysis = get_analysis(analysis_id)
    if not analysis or 'topics' not in analysis:
        flash('No results to export.', 'danger')
        return redirect(url_for('upload'))

    if format_type == 'csv':
        # Export topics as CSV
        rows = []
        for topic in analysis['topics']:
            for word, weight in topic['word_weight_pairs']:
                rows.append({
                    'Topic': topic['topic_id'],
                    'Word': word,
                    'Weight': weight,
                })
        df = pd.DataFrame(rows)
        filepath = os.path.join(RESULTS_FOLDER, f'topics_{analysis_id}.csv')
        df.to_csv(filepath, index=False)
        return send_file(filepath, as_attachment=True, download_name='lda_topics_report.csv')

    elif format_type == 'excel':
        filepath = os.path.join(RESULTS_FOLDER, f'report_{analysis_id}.xlsx')
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Topics sheet
            topic_rows = []
            for topic in analysis['topics']:
                for word, weight in topic['word_weight_pairs']:
                    topic_rows.append({
                        'Topic': topic['topic_id'],
                        'Word': word,
                        'Weight': weight,
                    })
            pd.DataFrame(topic_rows).to_excel(writer, sheet_name='Topics', index=False)

            # Distribution sheet
            pd.DataFrame(analysis['distribution']).to_excel(
                writer, sheet_name='Distribution', index=False
            )

            # Evaluation sheet
            eval_data = analysis.get('evaluation', {})
            eval_rows = [
                {'Metric': 'Coherence (c_v)', 'Value': eval_data.get('coherence_c_v', '')},
                {'Metric': 'Coherence (u_mass)', 'Value': eval_data.get('coherence_u_mass', '')},
                {'Metric': 'Perplexity', 'Value': eval_data.get('perplexity', '')},
                {'Metric': 'Number of Topics', 'Value': eval_data.get('num_topics', '')},
            ]
            pd.DataFrame(eval_rows).to_excel(writer, sheet_name='Evaluation', index=False)

            # Rating-based sentiment for numerical evaluation constructs
            rating_rows = []
            for construct, summary in analysis.get('rating_constructs', {}).items():
                rating_rows.append({
                    'Construct': construct,
                    'Mean Rating': summary['mean'],
                    'Classification': summary['classification'].title(),
                    'Responses': summary['response_count'],
                    'Negative (1-2)': summary['counts']['negative'],
                    'Neutral (3)': summary['counts']['neutral'],
                    'Positive (4-5)': summary['counts']['positive'],
                })
            pd.DataFrame(rating_rows).to_excel(
                writer, sheet_name='Rating Constructs', index=False
            )

            # Document-topic assignments
            dominant = analysis.get('dominant_topics', [])
            doc_rows = []
            for d in dominant:
                idx = d['document_index']
                if idx is None:
                    idx = d.get('index')
                if not isinstance(idx, int) or idx < 0 or idx >= len(analysis['filtered_raw']):
                    continue
                doc_rows.append({
                    'Document Index': idx,
                    'Feedback': analysis['filtered_raw'][idx][:500],
                    'Dominant Topic': d['dominant_topic'],
                    'Probability': d['probability'],
                })
            pd.DataFrame(doc_rows).to_excel(writer, sheet_name='Documents', index=False)

            # Optimization results if available
            if 'optimization' in analysis:
                pd.DataFrame(analysis['optimization']['results']).to_excel(
                    writer, sheet_name='Optimization', index=False
                )

        return send_file(filepath, as_attachment=True, download_name='lda_topic_modeling_report.xlsx')

    flash('Invalid export format.', 'danger')
    return redirect(url_for('results', analysis_id=analysis_id))


@app.route('/about')
def about():
    """About the system."""
    return render_template('about.html')


# ─── Error handlers ──────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 300MB.', 'danger')
    return redirect(url_for('upload'))


@app.errorhandler(500)
def internal_error(e):
    logger.exception('Unhandled server error: %s', e)
    return render_template('500.html'), 500


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
