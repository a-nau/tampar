import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from src.tools.create_distorted_images import main

logger = logging.getLogger(__name__)


class TestCreateDistortedDataset(unittest.TestCase):
    def test_create_dataset(self):
        logger.info("Testing Dataset Creation")
        main(distortion_values=[-0.02, 0.04])
