import io
import tempfile
import unittest

import pandas as pd

import app as flask_app

app = flask_app.app
analysis_store = flask_app.analysis_store
set_analysis = flask_app.set_analysis
get_analysis = flask_app.get_analysis


class ModelConfigRouteTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.original_store_dir = flask_app.ANALYSIS_STORE_DIR
        self.temp_store = tempfile.TemporaryDirectory()
        flask_app.ANALYSIS_STORE_DIR = self.temp_store.name
        analysis_store.clear()

    def tearDown(self):
        analysis_store.clear()
        flask_app.ANALYSIS_STORE_DIR = self.original_store_dir
        self.temp_store.cleanup()

    def test_model_config_route_trains_and_redirects_without_configuration_page(self):
        analysis_store['abc123'] = {
            'processed_texts': [['hello', 'world']],
            'raw_texts': ['hello world'],
            'preprocess_preview': [{'original': 'hello world', 'tokens': 'hello world'}],
        }

        response = self.client.get('/model-config/abc123')

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/results/abc123')
        self.assertEqual(get_analysis('abc123')['num_topics'], 5)

    def test_analysis_state_persists_across_restart(self):
        analysis = {'processed_texts': [['hello', 'world']], 'text_count': 1}
        set_analysis('persisted-id', analysis)

        analysis_store.clear()
        reloaded = get_analysis('persisted-id')

        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded['text_count'], 1)

    def test_model_config_generates_topics_from_uploaded_feedback(self):
        fruit_docs = [
            ['apple', 'orange', 'banana', 'fruit', 'sweet'],
            ['apple', 'banana', 'fruit', 'fresh', 'sweet'],
            ['orange', 'banana', 'fruit', 'fresh', 'ripe'],
            ['apple', 'orange', 'fruit', 'ripe', 'sweet'],
        ]
        analysis_store['fruit-analysis'] = {
            'processed_texts': fruit_docs,
            'filtered_raw': [' '.join(doc) for doc in fruit_docs],
            'sentiments': [],
            'preprocessing_stats': {},
            'baseline_evaluation': {'coherence_c_v': 1.0, 'perplexity': 0.0},
        }

        response = self.client.post(
            '/model-config/fruit-analysis',
            data={'num_topics': '2'},
        )

        self.assertEqual(response.status_code, 302)
        analysis = get_analysis('fruit-analysis')
        topic_words = {
            word
            for topic in analysis['topics']
            for word in topic['words']
        }
        self.assertEqual(analysis['num_topics'], 5)
        self.assertNotIn('baseline_evaluation', analysis)
        self.assertEqual(len(analysis['pretrained_predictions']), len(fruit_docs))
        self.assertEqual(
            sum(category['document_count'] for category in analysis['category_summary']),
            len(fruit_docs),
        )
        self.assertTrue(topic_words)
        self.assertTrue(topic_words.issubset({word for doc in fruit_docs for word in doc}))
        self.assertNotIn('classroom', topic_words)
        self.assertNotIn('curriculum', topic_words)

        results_response = self.client.get('/results/fruit-analysis')
        self.assertEqual(results_response.status_code, 200)
        results_page = results_response.get_data(as_text=True)
        self.assertIn('apple', results_page)
        self.assertIn('Feedback Categories', results_page)
        self.assertIn('Predicted Categories', results_page)
        self.assertIn('Predicted Category', results_page)
        self.assertIn('Match Score', results_page)
        self.assertIn(analysis['pretrained_predictions'][0]['category'], results_page)
        self.assertIn(analysis['pretrained_predictions'][0]['subtopic'], results_page)
        self.assertIn(
            f"{analysis['pretrained_predictions'][0]['confidence']:.4f}",
            results_page,
        )
        self.assertIn('Category / Subtopic', results_page)
        self.assertIn('Topic 0:', results_page)

    def test_model_config_reuses_results_for_same_file_and_settings(self):
        dataframe = pd.DataFrame({
            'feedback': ['clear explanations', 'helpful teaching'],
        })
        fingerprint = flask_app.dataframe_fingerprint(dataframe)
        set_analysis('existing-results', {
            'filename': 'feedback.csv',
            'dataframe': dataframe,
            'dataset_fingerprint': fingerprint,
            'feedback_column': 'feedback',
            'extra_stopwords': [],
            'num_topics': 5,
            'topics': [{'topic_id': 0, 'words': ['teaching']}],
            'distribution': [{'topic_id': 0, 'document_count': 2}],
        })
        analysis_store['new-upload'] = {
            'filename': 'feedback.csv',
            'dataframe': dataframe.copy(),
            'dataset_fingerprint': fingerprint,
            'feedback_column': 'feedback',
            'extra_stopwords': [],
            'processed_texts': [['clear', 'explanations'], ['helpful', 'teaching']],
            'preprocessing_stats': {},
        }

        response = self.client.post(
            '/model-config/new-upload',
            data={'num_topics': '2'},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/results/existing-results')
        self.assertIsNone(get_analysis('new-upload'))

    def test_upload_runs_feedback_analysis_and_redirects_to_five_topic_results(self):
        csv_content = (
            'year_level,feedback\n'
            '"1st Year","Clear teaching helps students understand difficult lessons"\n'
            '"2nd Year","Helpful examples make science concepts easier to learn"\n'
            '"3rd Year","More assignments improve writing and research skills"\n'
            '"4th Year","Classroom discussions encourage thoughtful student questions"\n'
            '"1st Year","Better course materials support learning and academic progress"\n'
        )

        response = self.client.post(
            '/upload',
            data={
                'upload_mode': 'file',
                'file': (io.BytesIO(csv_content.encode('utf-8')), 'feedback.csv'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 302)
        self.assertRegex(response.headers['Location'], r'^/results/[0-9a-f]{32}$')
        analysis_id = response.headers['Location'].rsplit('/', 1)[-1]
        analysis = get_analysis(analysis_id)
        self.assertEqual(analysis['feedback_column'], 'feedback')
        self.assertEqual(analysis['num_topics'], 5)
        self.assertEqual(len(analysis['topics']), 5)
        self.assertEqual(len(analysis['distribution']), 5)
        results_response = self.client.get(response.headers['Location'])
        self.assertEqual(results_response.status_code, 200)
        results_page = results_response.get_data(as_text=True)
        self.assertIn('<div class="step-circle">1</div>', results_page)
        self.assertIn('<div class="step-circle">5</div>', results_page)

    def test_same_filename_with_changed_file_contents_is_not_reused(self):
        previous = pd.DataFrame({'feedback': ['clear explanations']})
        current = pd.DataFrame({'feedback': ['poor explanations']})
        set_analysis('existing-results', {
            'filename': 'feedback.csv',
            'dataframe': previous,
            'feedback_column': 'feedback',
            'extra_stopwords': [],
            'num_topics': 2,
            'topics': [{'topic_id': 0, 'words': ['teaching']}],
            'distribution': [{'topic_id': 0, 'document_count': 1}],
        })
        current_analysis = {
            'filename': 'feedback.csv',
            'dataframe': current,
            'feedback_column': 'feedback',
            'extra_stopwords': [],
        }

        self.assertIsNone(
            flask_app.find_matching_completed_analysis(current_analysis, num_topics=2)
        )


if __name__ == '__main__':
    unittest.main()
