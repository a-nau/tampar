import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from src.tools.compute_similarity_scores import main as main_simscores
from src.tools.predict_tampering import main as main_predtamp

logger = logging.getLogger(__name__)


class TestTamperingPrediction(unittest.TestCase):
    def test_simscores_computation(self):
        logger.info("Testing Similarity Score Computation")
        df = main_simscores()

    def test_tampering_prediction(self):
        logger.info("Testing Tampering Prediction")
        df = main_predtamp()
