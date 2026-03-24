import os

# Root folder of the project (AGROTREE/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Crop_recommendation.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'crop_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'model_metrics.json')