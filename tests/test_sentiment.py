import pandas as pd

from modules.sentiment import analyze_rating_constructs, classify_rating


def test_classify_rating_uses_requested_boundaries():
    assert classify_rating(1) == 'negative'
    assert classify_rating(2) == 'negative'
    assert classify_rating(3) == 'neutral'
    assert classify_rating(4) == 'positive'
    assert classify_rating(5) == 'positive'


def test_rating_constructs_summarize_scale_and_ignore_metadata():
    data = pd.DataFrame({
        'student_id': [1, 2, 3],
        'teaching_quality': [1, 3, 5],
        'course_support': [4, 5, 4],
        'comment': ['bad', 'okay', 'good'],
    })

    constructs = analyze_rating_constructs(data, excluded_columns={'comment'})

    assert set(constructs) == {'teaching_quality', 'course_support'}
    assert constructs['teaching_quality']['classification'] == 'neutral'
    assert constructs['teaching_quality']['counts'] == {
        'negative': 1,
        'neutral': 1,
        'positive': 1,
    }
    assert constructs['course_support']['classification'] == 'positive'


def test_pretrained_management_topics_cover_time_and_class_feedback():
    from modules.pretrained_model import PRETRAINED_CATEGORIES, PretrainedFeedbackModel

    management = PRETRAINED_CATEGORIES['Academic Support and Management']
    management_keys = {key.lower() for key in management}

    assert any('time' in key or 'schedule' in key for key in management_keys)
    assert any('class' in key for key in management_keys)

    classifier = PretrainedFeedbackModel()
    result = classifier.classify(
        'The class schedule is too crowded and time management is poor for students.'
    )

    assert result['category'] == 'Academic Support and Management'


def test_pretrained_topics_have_requested_non_overlapping_display_names():
    from modules.pretrained_model import PretrainedFeedbackModel

    topics = PretrainedFeedbackModel.build_predefined_topics()

    assert [topic['display_name'] for topic in topics] == [
        'Management of Teaching and Learning',
        'Content Knowledge, Pedagogy, Technology',
        'Commitment and Transparency',
    ]