import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import einops
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.simsac.models.our_models.SimSaC import SimSaC_Model
from src.simsac.utils.plot import flow_to_image, show_flow
from src.tampering.utils import get_side_surface_patches, numpy2torch, rescale

ckpt_names = [
    "synthetic.pth",
    "synth_then_joint_synth_changesim.pth",
    "synth_then_joint_synth_cmu.pth",
]


class SimSaC:
    _instances: dict = {}

    def __init__(self, ckpt_name: str = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimSaC_Model(
            batch_norm=True,
            pyramid_type="VGG",
            div=1.0,
            evaluation=True,
            consensus_network=False,
            cyclic_consistency=True,
            dense_connection=True,
            decoder_inputs="corr_flow_feat",
            refinement_at_all_levels=False,
            refinement_at_adaptive_reso=True,
            num_class=2,
            use_pac=False,
            vpr_candidates=False,
        )
        ckpt_name = ckpt_names[0] if ckpt_name == "" else ckpt_name
        pretrained_dict = torch.load((ROOT / "simsac" / "weight" / ckpt_name))
        self.model.load_state_dict(pretrained_dict["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, ckpt_name: str = ""):
        ckpt_name = ckpt_names[0] if ckpt_name == "" else ckpt_name
        if cls._instances.get(ckpt_name, None) is None:
            cls._instances[ckpt_name] = cls(ckpt_name)
        return cls._instances[ckpt_name]

    def inference(self, im1: np.ndarray, im2: np.ndarray):
        # Transform
        transform = transforms.Compose([transforms.ToTensor()])
        resize_transform = transforms.Resize((256, 256))
        im1 = transform(im1).unsqueeze(0)
        im1_256 = resize_transform(im1)
        im2 = transform(im2).unsqueeze(0)
        im2_256 = resize_transform(im2)
        flow, change = self.model(
            im1.to(self.device),
            im2.to(self.device),
            im1_256.to(self.device),
            im2_256.to(self.device),
        )
        imgs = []
        for i in [0, 1]:
            change_ = img2vis(change, rescale=True)[..., i]
            change_ = cv2.cvtColor(change_, cv2.COLOR_GRAY2RGB)
            imgs.append(change_)
        flow_ = flow_to_image(img2vis(flow, rescale=False))
        imgs.append(flow_)
        return imgs  # change1, change2, flow


def img2vis(img, rescale=True):
    img = img.squeeze(0)
    img = einops.rearrange(img, "c h w -> h w c")
    img = img.detach().cpu().numpy()
    if rescale:
        img += np.max(img) - np.min(img)
        img *= 255 / np.max(img)
    return img.astype(np.float32)
