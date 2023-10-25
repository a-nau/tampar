import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())


import cv2
import lpips
import numpy as np
import pyiqa
import torch
from pytorch_msssim import ms_ssim, ssim
from skimage import feature

from src.tampering.utils import numpy2torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cw_ssim = pyiqa.create_metric("cw_ssim", device=device, as_loss=False)


class LpipsLossSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = lpips.LPIPS(net="alex").to(device)
        return cls._instance


def compute_lpips(im1: np.ndarray, im2: np.ndarray):
    lpips_loss = LpipsLossSingleton.get_instance()
    res = (
        lpips_loss(
            numpy2torch(im1).int(),
            numpy2torch(im2).int(),
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    return float(res)


def compute_msssim(im1: np.ndarray, im2: np.ndarray):
    res = ms_ssim(
        numpy2torch(im1), numpy2torch(im2), data_range=255, size_average=False
    )
    res = res.squeeze().detach().cpu().numpy()
    return float(res)


def compute_cwssim(im1: np.ndarray, im2: np.ndarray):
    res = cw_ssim(numpy2torch(im1 / 255), numpy2torch(im2 / 255))
    res = res.squeeze().detach().cpu().numpy()
    return float(res)


def compute_ssim(im1: np.ndarray, im2: np.ndarray):
    res = ssim(numpy2torch(im1), numpy2torch(im2), data_range=255, size_average=False)
    res = res.squeeze().detach().cpu().numpy()
    return float(res)


def compute_hog(im1: np.ndarray, im2: np.ndarray):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # Define the parameters for HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Compute HOG features
    hog_features1 = feature.hog(
        im1,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        transform_sqrt=True,
        block_norm="L1",
    )

    hog_features2 = feature.hog(
        im2,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        transform_sqrt=True,
        block_norm="L1",
    )

    res = np.linalg.norm(hog_features1 - hog_features2)
    return res


def compute_mse(im1: np.ndarray, im2: np.ndarray):
    return np.mean((im1 / 255 - im2 / 255) ** 2)


def compute_mae(im1: np.ndarray, im2: np.ndarray):
    return np.mean(np.abs(im1 / 255 - im2 / 255))


def compute_sift(im1: np.ndarray, im2: np.ndarray):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

    # Create a brute-force matcher
    bf = cv2.BFMatcher()

    try:
        # Match the descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Compute the similarity score as the average distance of the matches
        similarity_score = sum([match.distance for match in matches]) / len(matches)
    except:
        similarity_score = 0

    return similarity_score
