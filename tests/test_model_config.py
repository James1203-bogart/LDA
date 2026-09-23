import os
import shutil
import unittest

from app import app, analysis_store, set_analysis, get_analysis, ANALYSIS_STORE_DIR


class ModelConfigRouteTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        analysis_store.clear()
        if os.path.exists(ANALYSIS_STORE_DIR):
            shutil.rmtree(ANALYSIS_STORE_DIR)

    def tearDown(self):
        analysis_store.clear()
        if os.path.exists(ANALYSIS_STORE_DIR):
            shutil.rmtree(ANALYSIS_STORE_DIR)

    def test_model_config_renders_when_preprocessing_stats_are_missing(self):
        analysis_store['abc123'] = {
            'processed_texts': [['hello', 'world']],
            'raw_texts': ['hello world'],
            'preprocess_preview': [{'original': 'hello world', 'tokens': 'hello world'}],
        }

        response = self.client.get('/model-config/abc123')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'LDA Model Configuration', response.data)

    def test_analysis_state_persists_across_restart(self):
        analysis = {'processed_texts': [['hello', 'world']], 'text_count': 1}
        set_analysis('persisted-id', analysis)

        analysis_store.clear()
        reloaded = get_analysis('persisted-id')

        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded['text_count'], 1)


if __name__ == '__main__':
    unittest.main()
