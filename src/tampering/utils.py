import cv2
import einops
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy2torch(img):
    return einops.rearrange(torch.tensor(img), "h w c -> c h w").unsqueeze(0).to(device)


def get_side_surface_patches(image, grid_size=3):
    patch_size = (image.shape[1] // grid_size, image.shape[0] // grid_size)
    n_channels = image.shape[2]

    strided_view = np.lib.stride_tricks.as_strided(
        image,
        shape=(
            grid_size,
            grid_size,
            patch_size[1],
            patch_size[0],
            n_channels,
        ),
        strides=(
            image.strides[0] * patch_size[1],
            image.strides[1] * patch_size[0],
            image.strides[0],
            image.strides[1],
            image.strides[2],
        ),
    )
    strided_view = strided_view.reshape(grid_size**2, patch_size[1], patch_size[0], 3)
    return [strided_view[i, ...] for i in range(strided_view.shape[0])]


def rescale(img):
    img += torch.max(img) - torch.min(img)
    img *= 255 / torch.max(img)
    return img


def compute_keypoint_mask(points: np.ndarray, image_size):
    image = np.zeros(image_size, dtype=np.uint8)
    cv2.drawContours(image, [points.astype(int)], 0, 255, -1)
    return image.astype(int)


def compute_keypoint_area(points: np.ndarray, image_size):
    image = compute_keypoint_mask(points, image_size)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])
    return area
