#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

from detectron2.config import CfgNode
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_keypoint_head, build_model  # noqa

from src.maskrcnn.modeling.backbone.dla import build_dla_from_vision_fpn_backbone

logger = logging.getLogger(__name__)


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge([640, 800], max_size=1200),
            T.RandomContrast(intensity_min=0.9, intensity_max=1.1),
            T.RandomBrightness(intensity_min=0.9, intensity_max=1.1),
            T.RandomSaturation(intensity_min=0.9, intensity_max=1.1),
            T.RandomLighting(scale=255),
        ]

        return build_detection_train_loader(
            cfg, mapper=DatasetMapper(cfg, True, augmentations=augs)
        )

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=DatasetMapper(
                cfg,
                False,
            ),
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        return COCOEvaluator(
            dataset_name,
            tasks=("bbox", "segm", "keypoints"),
            output_dir=output_dir,
            kpt_oks_sigmas=[0.05] * 5 + [0.1] + [0.05] * 2,
        )
