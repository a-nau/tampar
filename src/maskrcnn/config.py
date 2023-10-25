from detectron2.config import CfgNode as CN


def get_maskrcnn_cfg_defaults(cfg):
    """
    # -*- coding: utf-8 -*-
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

    This function has been modified to accept some new custom configuration keys.
    (Remark in the form "This key ...")

    Customize the detectron2 cfg to include some new keys and default values
    for Mesh R-CNN.
    """
    cfg.MODEL.FREEZE_BACKBONE_COMPLETE = False
    cfg.MODEL.FREEZE_RPN = False
    cfg.MODEL.FREEZE_BOX_HEAD = False
    cfg.MODEL.FREEZE_MASK_HEAD = False

    cfg.MODEL.DLA = CN()
    cfg.MODEL.DLA.TYPE = "dla34"
    cfg.MODEL.DLA.TRICKS = False
    cfg.MODEL.RESNETS.TORCHVISION = True

    # Add necessary defaults when running in Docker: https://detectron2.readthedocs.io/en/latest/modules/config.html
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    cfg.SOLVER.BASE_LR_END = 0.0
    cfg.SOLVER.NUM_DECAYS = 3
    cfg.SOLVER.RESCALE_INTERVAL = False

    return cfg
