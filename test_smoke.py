"""Quick smoke test for core modules."""
import sys
sys.path.insert(0, '.')

from modules.preprocessing import TextPreprocessor
from modules.topic_modeling import LDATopicModeler
from modules.evaluation import ModelEvaluator

texts = [
    'The teacher explains the lessons clearly and gives real-life examples',
    'I wish the instructor would use more visual aids and presentations',
    'The course is interesting but the workload is too heavy',
    'Math class is very difficult and the professor goes too fast',
    'The English class helped me improve my writing skills',
    'The laboratory activities are very engaging and help us apply what we learned',
    'The classroom is too hot and makes it hard to concentrate',
    'I like that the teacher gives us clear rubrics for our projects',
    'The online learning platform is confusing and hard to navigate',
    'Grammar lessons are repetitive from high school',
]

pp = TextPreprocessor()
processed = pp.preprocess_corpus(texts)
stats = pp.get_preprocessing_stats(texts, processed)
print('Preprocessing stats:', stats)

modeler = LDATopicModeler()
corpus_stats = modeler.build_corpus(processed, min_df=1, max_df=0.9)
print('Corpus stats:', corpus_stats)

topics = modeler.train_model(num_topics=3)
print(f'Topics found: {len(topics)}')
for t in topics:
    print(f"  Topic {t['topic_id']}: {t['words'][:5]}")

evaluator = ModelEvaluator(modeler.model, modeler.dtm, modeler.feature_names, modeler.processed_texts)
e = evaluator.full_evaluation()
print(f"Coherence: {e['coherence_c_v']}")
print(f"Perplexity: {e['perplexity']}")
print('ALL TESTS PASSED')
