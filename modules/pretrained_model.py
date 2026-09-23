"""Bundled, inference-only classifier for student feedback."""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PRETRAINED_CATEGORIES = {
    'Teaching and Learning': {
        'Instruction Quality': 'clear explanations, teaching methods, class delivery, teaching pace, teacher communication, class engagement',
        'Class Delivery and Content': 'course content, lessons, curriculum, class discussion, learning activities, class examples, practical applications',
        'Assessment and Feedback': 'grading, exams, quizzes, rubrics, assignments, projects, feedback, evaluation',
    },
    'Academic Support and Management': {
        'Time and Class Scheduling': 'time management, class schedule, class hours, timetable, scheduling, deadline pressure, class conflicts, workload, class timing',
        'Student Support': 'tutorials, help, office hours, mentoring, questions, guidance, academic support, advising',
        'Learning Progress': 'difficulty, practice, understanding, confidence, improvement, knowledge gain, academic progress, performance',
    },
    'Facilities and Learning Environment': {
        'Facilities and Resources': 'classroom, laboratory, equipment, ventilation, lighting, library, facilities, learning materials, resources',
        'Digital Learning': 'online platform, learning management system, website, navigation, technology, assignment submission, virtual learning',
        'Engagement and Activities': 'hands-on activities, experiments, group work, discussions, participation, creative activities, field activities',
    },
}



class PretrainedFeedbackModel:
    """Classify feedback using a fixed taxonomy without fitting to uploaded data."""

    def __init__(self):
        self.labels = []
        self.prototype_texts = []
        for category, subtopics in PRETRAINED_CATEGORIES.items():
            for subtopic, description in subtopics.items():
                self.labels.append((category, subtopic))
                self.prototype_texts.append(description)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.prototype_matrix = self.vectorizer.fit_transform(self.prototype_texts)

    def classify(self, text):
        """Return the best category and subtopic for one feedback response."""
        matrix = self.vectorizer.transform([text or ''])
        scores = cosine_similarity(matrix, self.prototype_matrix)[0]
        best_index = int(np.argmax(scores))
        category, subtopic = self.labels[best_index]
        category_scores = [
            score for score, (label, _) in zip(scores, self.labels)
            if label == category
        ]
        return {
            'category': category,
            'subtopic': subtopic,
            'confidence': round(float(scores[best_index]), 4),
            'category_score': round(float(max(category_scores)), 4),
        }

    def classify_documents(self, texts):
        """Classify documents without changing the bundled model."""
        return [self.classify(text) for text in texts]

    @staticmethod
    def summarize(predictions):
        """Return counts for categories and their subtopics."""
        counts = Counter((item['category'], item['subtopic']) for item in predictions)
        return [
            {
                'category': category,
                'subtopics': [
                    {'name': subtopic, 'document_count': counts[(category, subtopic)]}
                    for subtopic in subtopics
                ],
                'document_count': sum(counts[(category, subtopic)] for subtopic in subtopics),
            }
            for category, subtopics in PRETRAINED_CATEGORIES.items()
        ]

    @staticmethod
    def build_predefined_topics():
        """Create the fixed topic cards used for student feedback reporting."""
        topics = []
        for topic_id, (category, subtopics) in enumerate(PRETRAINED_CATEGORIES.items(), start=1):
            words = []
            for subtopic, description in subtopics.items():
                for token in description.split(','):
                    cleaned = token.strip().lower()
                    if cleaned and cleaned not in words:
                        words.append(cleaned)
            words = words[:10]
            topics.append({
                'topic_id': topic_id,
                'category': category,
                'words': words,
                'weights': [1.0 for _ in words],
                'word_weight_pairs': [(word, 1.0) for word in words],
            })
        return topics