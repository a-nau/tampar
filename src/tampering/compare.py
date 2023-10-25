import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())


import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.filters import DexiNed, laplacian, sobel
from skimage import exposure

try:
    from src.simsac.inference import SimSaC
except ImportError as e:
    print(f"Could not import SimSaC: {repr(e)}")
from src.tampering.metrics import (  # noqa
    compute_cwssim,
    compute_hog,
    compute_lpips,
    compute_mae,
    compute_mse,
    compute_msssim,
    compute_sift,
    compute_ssim,
)
from src.tampering.parcel import PATCH_ORDER
from src.tampering.utils import get_side_surface_patches, numpy2torch

METRICS = [
    # "lpips",
    "msssim",
    "cwssim",
    "ssim",
    "hog",
    # "mse",
    "mae",
    # "sift"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CompareType:
    SIMSAC = "simsac"
    DEXINED = "dexined"
    CANNY = "canny"
    LAPLACIAN = "laplacian"
    SOBEL = "sobel"
    EXPOSURE = "exposure"
    PLAIN = "plain"
    MEAN_CHANNEL = "meanchannel"
    EQUALIZE_HIST = "equalize_hist"
    CLAHE = "clahe"

    @classmethod
    def KORNIA(cls):
        # return [cls.DEXINED, cls.LAPLACIAN, cls.SOBEL]
        return [cls.DEXINED]

    @classmethod
    def ALL(cls):
        return [
            cls.SIMSAC,
            cls.DEXINED,
            cls.CANNY,
            cls.LAPLACIAN,
            cls.SOBEL,
            cls.EXPOSURE,
            cls.PLAIN,
            cls.EQUALIZE_HIST,
            cls.MEAN_CHANNEL,
            cls.CLAHE,
        ]

    @classmethod
    def SELECTION(cls):
        return [
            cls.PLAIN,
            # cls.SIMSAC,
            # cls.DEXINED,
            cls.CANNY,
            cls.LAPLACIAN,
            # cls.SOBEL,
            # cls.EXPOSURE,
            # cls.EQUALIZE_HIST,
            cls.MEAN_CHANNEL,
            # cls.CLAHE,
        ]


class DexiNedInference:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DexiNed(pretrained=True).to(device)
        return cls._instance


def apply_homogenization(
    compare_type: str, input_patch1: np.ndarray, input_patch2: np.ndarray
):
    if compare_type == CompareType.SIMSAC:
        patch1, patch2 = compare_simsac(input_patch1, input_patch2)
    elif compare_type in CompareType.KORNIA():
        patch1, patch2 = compare_kornia(
            input_patch1, input_patch2, filter_name=compare_type
        )
    elif compare_type == CompareType.EXPOSURE:
        patch2 = exposure.match_histograms(input_patch2, input_patch1).astype(
            np.float32
        )
        patch1 = input_patch1
    elif compare_type == CompareType.PLAIN:
        patch1, patch2 = input_patch1, input_patch2
    elif compare_type == CompareType.MEAN_CHANNEL:
        patch1, patch2 = compare_mean_channel(input_patch1, input_patch2)
    elif compare_type == CompareType.EQUALIZE_HIST:
        patch1, patch2 = compare_equalize_histogram_bw(input_patch1, input_patch2)
    elif compare_type == CompareType.CLAHE:
        patch1, patch2 = compare_clahe(input_patch1, input_patch2)
    elif compare_type == CompareType.CANNY:
        patch1, patch2 = compare_canny(input_patch1, input_patch2)
    elif compare_type == CompareType.LAPLACIAN:
        patch1, patch2 = compare_laplacian(input_patch1, input_patch2)
    elif compare_type == CompareType.SOBEL:
        patch1, patch2 = compare_sobel(input_patch1, input_patch2)
    else:
        raise NotImplementedError
    return patch1, patch2


def compute_uvmap_similarity(
    uvmap1: np.ndarray,
    uvmap2: np.ndarray,
    output_path: Path,
    compare_type: str,
    visualize: bool = True,
):
    results = {}
    for i, (input_patch1, input_patch2) in enumerate(
        zip(get_side_surface_patches(uvmap1), get_side_surface_patches(uvmap2))
    ):
        metrics = {}
        if np.mean(input_patch1) >= 250 or np.mean(input_patch2) >= 250:
            continue  # skip
        patch1, patch2 = apply_homogenization(compare_type, input_patch1, input_patch2)

        for metric in METRICS:
            compute_metric = globals()[f"compute_{metric}"]
            if not compare_type == CompareType.SIMSAC:
                metrics[metric] = compute_metric(
                    patch1.astype(np.float32), patch2.astype(np.float32)
                )
            else:
                metrics[metric] = 0.5 * (
                    compute_metric(patch1, np.zeros_like(patch2))
                    + compute_metric(np.zeros_like(patch1), patch2)
                )
        results[PATCH_ORDER[i]] = metrics

        if visualize:
            pad_size = 6
            summary_imgs = []
            for img in [input_patch1, input_patch2, patch1, patch2]:
                img = np.pad(
                    img,
                    ((2, 2), (2, 2), (0, 0)),
                    constant_values=0,
                )  # black margin
                img = np.pad(
                    img,
                    ((0, 0), (pad_size, pad_size), (0, 0)),
                    constant_values=255,
                )  # white padding
                summary_imgs.append(img)
            summary = np.hstack(summary_imgs)
            plt.imshow(summary.astype(int))
            plt.show()
            cv2.imwrite(
                (output_path / f"compare_{str(i).zfill(2)}.jpg").as_posix(), summary
            )
    return results


def compare_simsac(im1, im2, threshod=200, ckpt_name=""):
    simsac = SimSaC.get_instance(ckpt_name)
    imgs = simsac.inference(im1.astype(np.uint8), im2.astype(np.uint8))
    for i, img in enumerate(imgs):
        img = cv2.resize(img, (im1.shape[1], im1.shape[0]))
        # Prepare
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.uint8)
        # Adjust Image
        new_img = (img > threshod).astype(np.float32) * 255
        # Save Image
        new_img = new_img.astype(np.float32)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        imgs[i] = new_img
    change1, change2, flow = imgs
    return 255 - change1, change2


def compare_kornia(im1: np.ndarray, im2: np.ndarray, filter_name="dexined"):
    # Prepare
    im1 = numpy2torch(im1).float()
    im2 = numpy2torch(im2).float()
    # Apply
    if filter_name == "dexined":
        dexined = DexiNedInference.get_instance()
        im1 = dexined(im1)[-1]
        im2 = dexined(im2)[-1]
        im1 *= 255
        im2 *= 255
    elif filter_name == "sobel":
        im1 = sobel(im1)
        im2 = sobel(im2)
        im1 = (1 - im1) * 255
        im2 = (1 - im2) * 255
    elif filter_name == "laplacian":
        kernel_size = 3
        im1 = laplacian(im1, kernel_size)
        im2 = laplacian(im2, kernel_size)
        im1 = (1 - im1) * 255
        im2 = (1 - im2) * 255
    else:
        raise NotImplementedError
    # Adjust
    im1 = im1.detach().cpu().squeeze(0)
    im2 = im2.detach().cpu().squeeze(0)
    im1 = einops.rearrange(im1, "c h w -> h w c").numpy()
    im2 = einops.rearrange(im2, "c h w -> h w c").numpy()
    if im1.shape[2] == 1 and im2.shape[2] == 1:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    return im1, im2


def compare_canny(
    im1: np.ndarray, im2: np.ndarray, threshold1=100, threshold2=200, adaptive=True
):
    imgs = []
    for img in [im1, im2]:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        if adaptive:
            threshold2, _ = cv2.threshold(
                imgray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            threshold1 = 0.4 * threshold2
        edges = cv2.Canny(imgray, threshold1, threshold2)
        edges = cv2.cvtColor(edges.astype(np.float32), cv2.COLOR_GRAY2RGB)
        imgs.append(edges)
    return imgs


def compare_laplacian(im1: np.ndarray, im2: np.ndarray):
    imgs = []
    for img in [im1, im2]:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        edges = cv2.Laplacian(imgray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        edges = cv2.cvtColor(edges.astype(np.float32), cv2.COLOR_GRAY2RGB)
        imgs.append(edges)
    return imgs


def compare_sobel(im1: np.ndarray, im2: np.ndarray):
    imgs = []
    for img in [im1, im2]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = np.uint8(np.absolute(sobel_x))
        sobel_y = np.uint8(np.absolute(sobel_y))
        edges = cv2.bitwise_or(sobel_x, sobel_y)
        edges = cv2.cvtColor(edges.astype(np.float32), cv2.COLOR_GRAY2RGB)
        imgs.append(edges)
    return imgs


def compare_mean_channel(im1: np.ndarray, im2: np.ndarray):
    base_mean = np.mean(im1, axis=(0, 1))
    adjust_mean = np.mean(im2, axis=(0, 1))
    diff_mean = base_mean - adjust_mean
    img = im2 + diff_mean
    return im1, img


def compare_equalize_histogram_bw(im1: np.ndarray, im2: np.ndarray):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)
    im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR).astype(np.float32)
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR).astype(np.float32)
    return im1, im2


def compare_clahe(im1: np.ndarray, im2: np.ndarray):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # Create a CLAHE object with a clip limit of 2.0 and tile size of 8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the grayscale images
    im1 = clahe.apply(im1)
    im2 = clahe.apply(im2)
    # to colori mage
    im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR).astype(np.float32)
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR).astype(np.float32)
    return im1, im2
