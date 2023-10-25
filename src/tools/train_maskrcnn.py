import logging
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", UserWarning)
ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger

import src.maskrcnn.data.register_datasets  # noqa
from src.maskrcnn.config import get_maskrcnn_cfg_defaults
from src.maskrcnn.engine.train_loop import Trainer

logger = logging.getLogger("maskrcnn")


def setup(args):
    cfg = get_cfg()
    get_maskrcnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.freeze:
        cfg.freeze()
    default_setup(cfg, None)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskrcnn"
    )
    return cfg


def parse_args(args):
    parser = default_argument_parser()
    parser.add_argument("--gpus", default="0", help="Set GPUs that should be used")
    args = parser.parse_args(args)
    print("Command Line Args:", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.freeze = True
    return args


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,  # default 1 in default_argument_parser
            dist_url=args.dist_url,
            args=(args,),
        )
    except Exception as e:
        logger.exception(f"TERMINATED: {repr(e)}")
