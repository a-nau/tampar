import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from detectron2.engine import launch

from src.tools.train_maskrcnn import main, parse_args

logger = logging.getLogger(__name__)


class TestTrainCNN(unittest.TestCase):
    def test_train_cnn(self):
        logger.info("Testing training of CNN")
        args = parse_args([])
        args.config_file = (ROOT / "src/maskrcnn/configs/test.yaml").as_posix()
        args.num_gpus = 0
        args.num_machines = 1
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=0,
            dist_url=args.dist_url,
            args=(args,),
        )
