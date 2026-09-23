import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# For Render.com compatibility: only use filesystem for exports, not uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

if os.environ.get('VERCEL') == '1':
    UPLOAD_FOLDER = '/tmp/uploads'
    RESULTS_FOLDER = '/tmp/results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(32).hex())

MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300MB max upload

# LDA defaults
MAX_DOCUMENTS = 3000
DEFAULT_NUM_TOPICS = 5
MIN_TOPICS = 2
MAX_TOPICS = 20
LDA_PASSES = 15
LDA_ITERATIONS = 100
RANDOM_STATE = 42

# Environment detection
IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production'
IS_RENDER = os.environ.get('RENDER') == 'true'
