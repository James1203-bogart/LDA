"""
Student Feedback Topic Modeling System
Flask Web Application
Using Latent Dirichlet Allocation (LDA)
"""

import os
import io
import json
import hashlib
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
from modules.evaluation import ModelEvaluator
from modules.visualization import (
    generate_topic_bar_chart,
    generate_distribution_pie_chart, generate_coherence_line_chart,
    generate_per_topic_coherence_bar
)
from modules.sentiment import analyze_sentiments, analyze_rating_constructs
from modules.pretrained_model import PRETRAINED_TOPIC_NAMES, PretrainedFeedbackModel


def _apply_pretrained_topic_names(analysis):
    names_by_id = (
        dict(enumerate(PRETRAINED_TOPIC_NAMES, start=1))
        if analysis.get('predefined_results') else {}
    )
    topic_names = {}
    for topic in analysis.get('topics', []):
        topic_id = topic.get('topic_id')
        topic_name = names_by_id.get(topic_id) or topic.get('display_name')
        if not topic_name:
            top_words = topic.get('words') or [
                word for word, _ in topic.get('word_weight_pairs', [])
            ]
            topic_name = ', '.join(top_words[:3]) or f'Topic {topic_id}'
        topic['display_name'] = topic_name
        topic_names[topic_id] = topic_name

    for item in analysis.get('distribution', []):
        topic_name = topic_names.get(item.get('topic_id'))
        if topic_name:
            item['display_name'] = topic_name


def _apply_pretrained_categories(analysis):
    if 'category_summary' in analysis and 'pretrained_predictions' in analysis:
        return False

    classifier = PretrainedFeedbackModel()
    predictions = classifier.classify_documents(analysis.get('filtered_raw', []))
    analysis['pretrained_predictions'] = predictions
    analysis['category_summary'] = classifier.summarize(predictions)
    return True


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


def discard_analysis(analysis_id):
    """Remove a newly uploaded analysis when an existing result is reused."""
    analysis_store.pop(analysis_id, None)
    try:
        os.remove(_analysis_path(analysis_id))
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning('Failed to remove duplicate analysis %s: %s', analysis_id, e)


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


def dataframe_fingerprint(dataframe):
    """Return a stable fingerprint for the contents and schema of an uploaded dataset."""
    digest = hashlib.sha256()
    digest.update(repr(list(dataframe.columns)).encode('utf-8'))
    digest.update(repr([str(dtype) for dtype in dataframe.dtypes]).encode('utf-8'))
    digest.update(str(dataframe.shape).encode('ascii'))
    digest.update(pd.util.hash_pandas_object(dataframe, index=True).values.tobytes())
    return digest.hexdigest()


def find_matching_completed_analysis(analysis, num_topics):
    """Find a prior result for the same file and analysis settings."""
    dataframe = get_dataframe_from_analysis(analysis)
    if dataframe is None:
        return None

    fingerprint = analysis.get('dataset_fingerprint') or dataframe_fingerprint(dataframe)
    stopwords = frozenset(analysis.get('extra_stopwords', []))

    for completed in get_completed_analyses():
        candidate = get_analysis(completed['analysis_id'])
        if not candidate or candidate.get('filename') != analysis.get('filename'):
            continue
        if candidate.get('feedback_column') != analysis.get('feedback_column'):
            continue
        if candidate.get('num_topics') != num_topics:
            continue
        if frozenset(candidate.get('extra_stopwords', [])) != stopwords:
            continue

        candidate_fingerprint = candidate.get('dataset_fingerprint')
        candidate_dataframe = get_dataframe_from_analysis(candidate)
        if not candidate_fingerprint and candidate_dataframe is not None:
            candidate_fingerprint = dataframe_fingerprint(candidate_dataframe)
        if candidate_fingerprint == fingerprint:
            return completed['analysis_id']

    return None


def extract_google_sheet_id(url):
    """Extract the spreadsheet ID from a Google Sheets or Google Forms URL."""
    m = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    return None


def get_human_evaluation_language_pack(language='multilingual'):
    """Return the translated UI labels for the human evaluation section."""
    normalized = (language or 'multilingual').strip().lower()
    packs = {
        'multilingual': {
            'title': 'Human Evaluation of Topics',
            'intro': (
                'Ask qualified faculty members or academic personnel to review the top words and '
                'representative feedback for each topic. Record a concise label and judge how well '
                'it represents the comments. This expert review complements automated coherence '
                'scores; it is not a substitute for them.'
            ),
            'reviewer_label': 'Reviewer name or academic role',
            'language_label': 'Evaluation language',
            'topic_label': 'Human-assigned topic label',
            'assessment_label': 'Does the label represent the feedback?',
            'representative_feedback': 'Representative feedback',
            'no_examples': 'No representative comments are available for this topic.',
            'notes_label': 'Reviewer notes',
            'select_assessment': 'Select an assessment',
            'accurate': 'Accurate',
            'partially_accurate': 'Partially accurate',
            'inaccurate': 'Inaccurate',
            'save_button': 'Save Human Evaluation',
            'view_results': 'Back to Results',
            'last_evaluated': 'Last evaluated:',
        },
        'tagalog': {
            'title': 'Pagsusuri ng Tao sa mga Paksa',
            'intro': (
                'Hilingin sa mga kwalipikadong faculty member o akademikong tauhan na suriin ang mga '
                'pinaka-kaugnay na salita at halimbawang tugon para sa bawat paksa. Itala ang maikling '
                'label at husgahan kung gaano ito nauugnay sa mga komento. Ang ekspertong pagsusuring ito '
                'ay sumusuporta sa mga awtomatikong coherence score; hindi ito pamalit sa mga ito.'
            ),
            'reviewer_label': 'Pangalan ng reviewer o papel sa akademya',
            'language_label': 'Wika ng pagsusuri',
            'topic_label': 'Label ng paksa na ibinigay ng tao',
            'assessment_label': 'Sinasalamin ba ng label ang feedback?',
            'representative_feedback': 'Kinatawan na feedback',
            'no_examples': 'Walang available na representatibong komento para sa paksang ito.',
            'notes_label': 'Mga tala ng reviewer',
            'select_assessment': 'Pumili ng pagmamarka',
            'accurate': 'Tumpak',
            'partially_accurate': 'Bahagyang tumpak',
            'inaccurate': 'Hindi tumpak',
            'save_button': 'I-save ang Pagsusuri ng Tao',
            'view_results': 'Bumalik sa mga Resulta',
            'last_evaluated': 'Huling sinusuri:',
        },
        'english': {
            'title': 'Human Evaluation of Topics',
            'intro': (
                'Ask qualified faculty members or academic personnel to review the top words and '
                'representative feedback for each topic. Record a concise label and judge how well '
                'it represents the comments. This expert review complements automated coherence '
                'scores; it is not a substitute for them.'
            ),
            'reviewer_label': 'Reviewer name or academic role',
            'language_label': 'Evaluation language',
            'topic_label': 'Human-assigned topic label',
            'assessment_label': 'Does the label represent the feedback?',
            'representative_feedback': 'Representative feedback',
            'no_examples': 'No representative comments are available for this topic.',
            'notes_label': 'Reviewer notes',
            'select_assessment': 'Select an assessment',
            'accurate': 'Accurate',
            'partially_accurate': 'Partially accurate',
            'inaccurate': 'Inaccurate',
            'save_button': 'Save Human Evaluation',
            'view_results': 'Back to Results',
            'last_evaluated': 'Last evaluated:',
        },
        'german': {
            'title': 'Menschliche Bewertung von Themen',
            'intro': (
                'Bitten Sie qualifizierte Fakultätsmitglieder oder akademische Mitarbeiter, die '
                'wichtigsten Wörter und repräsentativen Feedbacks für jedes Thema zu prüfen. Erfassen '
                'Sie ein kurzes Label und beurteilen Sie, wie gut es die Kommentare wiedergibt. Diese '
                'Expertenbewertung ergänzt automatisierte Kohärenzwerte; sie ersetzt sie nicht.'
            ),
            'reviewer_label': 'Name des Gutachters oder akademische Rolle',
            'language_label': 'Bewertungssprache',
            'topic_label': 'Vom Menschen zugewiesenes Themenlabel',
            'assessment_label': 'Spiegelt das Label das Feedback wider?',
            'representative_feedback': 'Repräsentatives Feedback',
            'no_examples': 'Für dieses Thema sind keine repräsentativen Kommentare verfügbar.',
            'notes_label': 'Anmerkungen des Gutachters',
            'select_assessment': 'Bewertung auswählen',
            'accurate': 'Korrekt',
            'partially_accurate': 'Teilweise korrekt',
            'inaccurate': 'Unkorrekt',
            'save_button': 'Menschliche Bewertung speichern',
            'view_results': 'Zurück zu den Ergebnissen',
            'last_evaluated': 'Zuletzt bewertet:',
        },
        'russian': {
            'title': 'Человеческая оценка тем',
            'intro': (
                'Попросите квалифицированных преподавателей или академический персонал проверить '
                'ключевые слова и репрезентативные отзывы для каждой темы. Запишите краткое название '
                'и оцените, насколько оно соответствует комментариям. Эта экспертная оценка дополняет '
                'автоматические показатели связности; она не заменяет их.'
            ),
            'reviewer_label': 'Имя рецензента или академическая роль',
            'language_label': 'Язык оценки',
            'topic_label': 'Название темы, заданное человеком',
            'assessment_label': 'Отражает ли название отзыв?',
            'representative_feedback': 'Репрезентативный отзыв',
            'no_examples': 'Для этой темы нет репрезентативных комментариев.',
            'notes_label': 'Заметки рецензента',
            'select_assessment': 'Выберите оценку',
            'accurate': 'Точно',
            'partially_accurate': 'Частично верно',
            'inaccurate': 'Неверно',
            'save_button': 'Сохранить человеческую оценку',
            'view_results': 'Назад к результатам',
            'last_evaluated': 'Последняя оценка:',
        },
        'arabic': {
            'title': 'التقييم البشري للمواضيع',
            'intro': (
                'اطلب من أعضاء هيئة التدريس المؤهلين أو الموظفين الأكاديميين مراجعة الكلمات الأساسية '
                'والتعليقات التمثيلية لكل موضوع. سجل تسمية مختصرة وقم بتقييم مدى تمثيلها للتعليقات. '
                'هذا التقييم الخبرائي يكمل درجات الترابط الآلية؛ ولا يحل محلها.'
            ),
            'reviewer_label': 'اسم المراجع أو الدور الأكاديمي',
            'language_label': 'لغة التقييم',
            'topic_label': 'تسمية الموضوع المعينة من قبل الإنسان',
            'assessment_label': 'هل التسمية تمثل الملاحظات؟',
            'representative_feedback': 'ملاحظات تمثيلية',
            'no_examples': 'لا توجد تعليقات تمثيلية متاحة لهذا الموضوع.',
            'notes_label': 'ملاحظات المراجع',
            'select_assessment': 'اختر التقييم',
            'accurate': 'دقيق',
            'partially_accurate': 'دقيق جزئيًا',
            'inaccurate': 'غير دقيق',
            'save_button': 'حفظ التقييم البشري',
            'view_results': 'العودة إلى النتائج',
            'last_evaluated': 'آخر تقييم:',
        },
    }
    return packs.get(normalized, packs['multilingual'])


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
            'dataset_fingerprint': dataframe_fingerprint(df),
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
    """Train a topic model using the uploaded feedback."""
    analysis = get_analysis(analysis_id)
    if not analysis or 'processed_texts' not in analysis:
        flash('Please complete preprocessing first.', 'danger')
        return redirect(url_for('upload'))

    processed_texts = analysis['processed_texts']
    max_topics = min(MAX_TOPICS, len(processed_texts))
    min_topics = min(MIN_TOPICS, max_topics)

    if request.method == 'POST':
        try:
            num_topics = int(request.form.get('num_topics', DEFAULT_NUM_TOPICS))
            if not min_topics <= num_topics <= max_topics:
                raise ValueError(
                    f'Choose between {min_topics} and {max_topics} topics for this dataset.'
                )

            matching_analysis_id = find_matching_completed_analysis(analysis, num_topics)
            if matching_analysis_id:
                discard_analysis(analysis_id)
                flash('Existing results found for this same file. Showing the saved results.', 'success')
                return redirect(url_for('results', analysis_id=matching_analysis_id))

            modeler = LDATopicModeler()
            modeler.build_corpus(processed_texts, min_df=1, max_df=1.0)
            analysis['topics'] = modeler.train_model(num_topics=num_topics)
            analysis['num_topics'] = num_topics
            analysis['dominant_topics'] = modeler.get_dominant_topics()
            classifier = PretrainedFeedbackModel()
            analysis['pretrained_predictions'] = classifier.classify_documents(
                analysis.get('filtered_raw', [])
            )
            analysis['category_summary'] = classifier.summarize(
                analysis['pretrained_predictions']
            )

            from collections import Counter, defaultdict
            topic_counts = Counter(
                item['dominant_topic'] for item in analysis['dominant_topics']
            )
            analysis['distribution'] = [
                {
                    'topic_id': topic_id,
                    'document_count': topic_counts.get(topic_id, 0),
                    'percentage': round(
                        topic_counts.get(topic_id, 0) / max(len(processed_texts), 1) * 100,
                        2,
                    ),
                }
                for topic_id in range(num_topics)
            ]

            evaluator = ModelEvaluator(
                modeler.model,
                modeler.dtm,
                modeler.feature_names,
                processed_texts,
            )
            analysis['evaluation'] = evaluator.full_evaluation()

            topic_sentiments = defaultdict(list)
            for item in analysis['dominant_topics']:
                index = item['document_index']
                if index < len(analysis.get('sentiments', [])):
                    score = analysis['sentiments'][index]
                    if score is not None:
                        topic_sentiments[item['dominant_topic']].append(
                            score.get('compound', 0.0)
                        )
            analysis['per_topic_sentiment'] = {
                topic_id: round(sum(scores) / len(scores), 4)
                for topic_id, scores in topic_sentiments.items()
            }
            analysis.pop('baseline_evaluation', None)
            analysis.pop('predefined_results', None)
            set_analysis(analysis_id, analysis)
            flash(f'Topic model trained on your feedback with {num_topics} topics.', 'success')
            return redirect(url_for('results', analysis_id=analysis_id))

        except ValueError as e:
            flash(f'Invalid model configuration: {str(e)}', 'danger')
            return redirect(request.url)
        except Exception as e:
            logger.exception('Error training LDA model: %s', e)
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
        default_topics=min(DEFAULT_NUM_TOPICS, max_topics),
        min_topics=min_topics,
        max_topics=max_topics,
    )


@app.route('/results/<analysis_id>')
def results(analysis_id):
    """Display topic modeling results and visualizations."""
    analysis = get_analysis(analysis_id)
    if not analysis or not analysis.get('topics') or not analysis.get('distribution'):
        flash('No results available. Please run the model first.', 'danger')
        return redirect(url_for('upload'))

    if _apply_pretrained_categories(analysis):
        set_analysis(analysis_id, analysis)
    _apply_pretrained_topic_names(analysis)
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
    pretrained_predictions = analysis.get('pretrained_predictions', [])
    category_predictions = []
    for idx, text in enumerate(analysis.get('filtered_raw', [])):
        prediction = (
            pretrained_predictions[idx]
            if idx < len(pretrained_predictions) else {}
        )
        display_text = str(text)
        category_predictions.append({
            'index': idx,
            'text': display_text[:500] + ('...' if len(display_text) > 500 else ''),
            'category': prediction.get('category'),
            'subtopic': prediction.get('subtopic'),
            'confidence': prediction.get('confidence'),
        })

    doc_topic_preview = []
    for d in dominant[:20]:
        idx = d.get('document_index')
        if idx is None:
            idx = d.get('index')
        topic_id = d.get('dominant_topic')
        if topic_id is None:
            topic_id = d.get('topic')
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
            'topic': topic_id,
            'probability': d.get('probability'),
            'sentiment': sentiment_val,
            'category': pretrained_predictions[idx].get('category')
            if isinstance(idx, int) and 0 <= idx < len(pretrained_predictions) else None,
            'subtopic': pretrained_predictions[idx].get('subtopic')
            if isinstance(idx, int) and 0 <= idx < len(pretrained_predictions) else None,
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
        category_predictions=category_predictions,
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

    _apply_pretrained_topic_names(analysis)

    if request.method == 'POST':
        if request.form.get('action') == 'save_human_review':
            assessments = {'accurate', 'partially_accurate', 'inaccurate'}
            allowed_languages = {'multilingual', 'tagalog', 'english', 'german', 'russian', 'arabic'}
            reviewer = request.form.get('reviewer', '').strip()
            evaluation_language = request.form.get('evaluation_language', 'multilingual').strip().lower()
            reviews = {}
            if not reviewer:
                flash('Enter the reviewer name or role before saving evaluations.', 'danger')
                return redirect(request.url)
            if evaluation_language not in allowed_languages:
                flash('Select a valid evaluation language.', 'danger')
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
                'language': evaluation_language,
                'topics': reviews,
            }
            set_analysis(analysis_id, analysis)
            try:
                save_human_evaluation_excel(analysis_id, analysis)
            except Exception:
                logger.exception('Failed to create human evaluation Excel file for %s', analysis_id)
                flash(
                    'Human topic evaluations were saved, but the Excel file could not be created.',
                    'warning',
                )
            else:
                flash('Human topic evaluations saved. Download the Excel file below.', 'success')
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

    human_review_language = human_review.get('language', 'multilingual')
    human_eval_text = get_human_evaluation_language_pack(human_review_language)

    model_comparison = None
    if analysis.get('processed_texts') and analysis.get('num_topics'):
        baseline_evaluation = analysis.get('baseline_evaluation')
        if baseline_evaluation is None:
            baseline_modeler = LDATopicModeler()
            baseline_modeler.build_corpus(
                analysis['processed_texts'], min_df=1, max_df=1.0
            )
            baseline_modeler.train_model(
                num_topics=analysis['num_topics'],
                learning_method='batch',
            )
            baseline_evaluation = ModelEvaluator(
                baseline_modeler.model,
                baseline_modeler.dtm,
                baseline_modeler.feature_names,
                analysis['processed_texts'],
            ).full_evaluation()
            analysis['baseline_evaluation'] = baseline_evaluation
            set_analysis(analysis_id, analysis)

        model_comparison = {
            'current': analysis.get('evaluation', {}),
            'baseline': baseline_evaluation,
        }

    return render_template(
        'compare.html',
        analysis_id=analysis_id,
        topics=analysis['topics'],
        has_manual='manual_df' in analysis,
        manual_columns=analysis.get('manual_columns', []),
        human_review_topics=human_review_topics,
        human_reviewer=human_review.get('reviewer', ''),
        human_review_language=human_review_language,
        human_eval_text=human_eval_text,
        model_comparison=model_comparison,
    )


@app.route('/export/<analysis_id>/<format_type>')
def export_report(analysis_id, format_type):
    """Export results report."""
    analysis = get_analysis(analysis_id)
    if not analysis or 'topics' not in analysis:
        flash('No results to export.', 'danger')
        return redirect(url_for('upload'))

    if format_type == 'human-evaluation':
        if not analysis.get('human_review', {}).get('topics'):
            flash('No saved human evaluation is available to export.', 'warning')
            return redirect(url_for('compare', analysis_id=analysis_id))
        filepath = save_human_evaluation_excel(analysis_id, analysis)
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f'human_evaluation_{analysis_id}.xlsx',
        )

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


def save_human_evaluation_excel(analysis_id, analysis):
    """Write saved human topic evaluations to an Excel workbook."""
    human_review = analysis['human_review']
    topic_names = {
        str(topic['topic_id']): topic.get('display_name') or f"Topic {topic['topic_id']}"
        for topic in analysis['topics']
    }
    language_pack = get_human_evaluation_language_pack(
        human_review.get('language', 'multilingual')
    )
    assessment_names = {
        'accurate': language_pack['accurate'],
        'partially_accurate': language_pack['partially_accurate'],
        'inaccurate': language_pack['inaccurate'],
    }
    rows = [
        {
            'Dataset': analysis.get('filename', ''),
            'Reviewer': human_review.get('reviewer', ''),
            'Evaluation Language': human_review.get('language', 'multilingual'),
            'Topic ID': topic_id,
            'LDA Topic': topic_names.get(topic_id, f'Topic {topic_id}'),
            'Human Label': review.get('label', ''),
            'Assessment': assessment_names.get(
                review.get('assessment'), review.get('assessment', '')
            ),
            'Notes': review.get('notes', ''),
            'Reviewed At': review.get('reviewed_at', ''),
        }
        for topic_id, review in human_review['topics'].items()
    ]

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filepath = os.path.join(RESULTS_FOLDER, f'human_evaluation_{analysis_id}.xlsx')
    pd.DataFrame(rows).to_excel(filepath, sheet_name='Human Evaluation', index=False)
    return filepath


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
