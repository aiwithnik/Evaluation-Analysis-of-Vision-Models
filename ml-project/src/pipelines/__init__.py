from .ingestion_pipeline import main as run_ingestion
from .preprocess_pipeline import main as run_preprocessing
from .feature_pipeline import main as run_feature_engineering
from .training_pipeline import main as run_training
from .evaluation_pipeline import main as run_evaluation

__all__ = [
    "run_ingestion",
    "run_preprocessing",
    "run_feature_engineering",
    "run_training",
    "run_evaluation",
]
