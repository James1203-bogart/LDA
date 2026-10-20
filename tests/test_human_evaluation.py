import app as flask_app


def sample_analysis():
    return {
        'topics': [
            {
                'topic_id': 0,
                'word_weight_pairs': [('teaching', 0.4), ('feedback', 0.2)],
            },
            {
                'topic_id': 1,
                'word_weight_pairs': [('facilities', 0.5), ('library', 0.3)],
            },
        ],
        'dominant_topics': [
            {'document_index': 0, 'dominant_topic': 0, 'probability': 0.7},
            {'document_index': 1, 'dominant_topic': 0, 'probability': 0.9},
            {'document_index': 2, 'dominant_topic': 1, 'probability': 0.8},
        ],
        'filtered_raw': [
            'The instructor explains clearly.',
            'Teaching feedback is helpful.',
            'The library needs more study spaces.',
        ],
    }


def test_compare_displays_representative_comments(monkeypatch):
    analysis = sample_analysis()
    monkeypatch.setattr(flask_app, 'get_analysis', lambda _: analysis)
    flask_app.app.config['TESTING'] = True

    response = flask_app.app.test_client().get('/compare/test-analysis')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Human Evaluation of Topics' in page
    assert 'Teaching feedback is helpful.' in page
    assert 'The library needs more study spaces.' in page
    assert 'No BERT model was run.' in page
    assert 'BERT (simulated demo)' in page
    assert '0.35 (fictional)' in page
    assert page.index('Manual Coding') < page.index('Model Comparison')


def test_compare_displays_keyword_names_for_lda_topics(monkeypatch):
    analysis = sample_analysis()
    monkeypatch.setattr(flask_app, 'get_analysis', lambda _: analysis)
    flask_app.app.config['TESTING'] = True

    response = flask_app.app.test_client().get('/compare/test-analysis')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Topic 0: teaching, feedback' in page
    assert 'Topic 1: facilities, library' in page


def test_compare_shows_and_caches_reproducible_batch_baseline(monkeypatch):
    analysis = sample_analysis()
    analysis.update({
        'processed_texts': [['instructor', 'clear'], ['library', 'space']],
        'num_topics': 2,
        'evaluation': {'coherence_c_v': 0.12, 'perplexity': 45.0},
    })
    model_runs = []

    class FakeModeler:
        def build_corpus(self, texts, min_df, max_df):
            assert texts == analysis['processed_texts']

        def train_model(self, num_topics, learning_method):
            model_runs.append((num_topics, learning_method))
            self.model = object()
            self.dtm = object()
            self.feature_names = ['clear', 'space']

    class FakeEvaluator:
        def __init__(self, model, dtm, feature_names, texts):
            assert texts == analysis['processed_texts']

        def full_evaluation(self):
            return {'coherence_c_v': 0.08, 'perplexity': 52.0}

    persisted = []
    monkeypatch.setattr(flask_app, 'get_analysis', lambda _: analysis)
    monkeypatch.setattr(flask_app, 'set_analysis', lambda key, value: persisted.append(key))
    monkeypatch.setattr(flask_app, 'LDATopicModeler', FakeModeler)
    monkeypatch.setattr(flask_app, 'ModelEvaluator', FakeEvaluator)
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()

    response = client.get('/compare/test-analysis')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Baseline LDA (batch)' in page
    assert 'reference baseline, not a recovered historical model' in page
    assert '0.12' in page
    assert '0.08' in page
    assert model_runs == [(2, 'batch')]
    assert analysis['baseline_evaluation']['perplexity'] == 52.0
    assert persisted == ['test-analysis']

    second_response = client.get('/compare/test-analysis')
    assert second_response.status_code == 200
    assert model_runs == [(2, 'batch')]


def test_compare_displays_pretrained_topic_names_for_saved_analyses(monkeypatch):
    analysis = {
        'predefined_results': True,
        'topics': [
            {'topic_id': 1, 'word_weight_pairs': []},
            {'topic_id': 2, 'word_weight_pairs': []},
            {'topic_id': 3, 'word_weight_pairs': []},
        ],
        'dominant_topics': [],
        'filtered_raw': [],
    }
    monkeypatch.setattr(flask_app, 'get_analysis', lambda _: analysis)
    flask_app.app.config['TESTING'] = True

    response = flask_app.app.test_client().get('/compare/test-analysis')

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Topic 1: Management of Teaching and Learning' in page
    assert 'Topic 2: Content Knowledge, Pedagogy, Technology' in page
    assert 'Topic 3: Commitment and Transparency' in page


def test_compare_saves_human_evaluations(monkeypatch):
    analysis = sample_analysis()
    persisted = []
    monkeypatch.setattr(flask_app, 'get_analysis', lambda _: analysis)
    monkeypatch.setattr(flask_app, 'set_analysis', lambda analysis_id, data: persisted.append(data.copy()))
    flask_app.app.config['TESTING'] = True

    response = flask_app.app.test_client().post(
        '/compare/test-analysis',
        data={
            'action': 'save_human_review',
            'reviewer': 'Faculty reviewer',
            'label_0': 'Teaching quality',
            'assessment_0': 'accurate',
            'notes_0': 'Comments consistently describe instruction.',
            'label_1': 'Learning facilities',
            'assessment_1': 'partially_accurate',
            'notes_1': 'Comments focus mostly on the library.',
        },
    )

    assert response.status_code == 302
    assert persisted
    assert analysis['human_review']['reviewer'] == 'Faculty reviewer'
    assert analysis['human_review']['topics']['0']['assessment'] == 'accurate'
    assert analysis['human_review']['topics']['1']['label'] == 'Learning facilities'
    assert analysis['human_review']['topics']['1']['notes'] == 'Comments focus mostly on the library.'
    assert analysis['human_review']['topics']['0']['reviewed_at']


def test_compare_rejects_invalid_human_assessment(monkeypatch):
    analysis = sample_analysis()
    persisted = []
    monkeypatch.setattr(flask_app, 'get_analysis', lambda _: analysis)
    monkeypatch.setattr(flask_app, 'set_analysis', lambda *args: persisted.append(args))
    flask_app.app.config['TESTING'] = True

    response = flask_app.app.test_client().post(
        '/compare/test-analysis',
        data={
            'action': 'save_human_review',
            'reviewer': 'Faculty reviewer',
            'label_0': 'Teaching quality',
            'assessment_0': 'accurate',
            'label_1': 'Facilities',
            'assessment_1': 'unknown',
        },
    )

    assert response.status_code == 302
    assert not persisted
    assert 'human_review' not in analysis
