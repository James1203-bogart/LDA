from modules.preprocessing import TextPreprocessor


def test_preprocessor_handles_missing_nltk_resources():
    preprocessor = TextPreprocessor()
    tokens = preprocessor.preprocess('This is a test sentence for preprocessing.')
    assert isinstance(tokens, list)
    assert len(tokens) > 0
