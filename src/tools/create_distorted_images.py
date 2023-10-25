import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np
import tqdm
from PIL import Image as PILImage
from skimage import measure
from wand.image import Image

IMAGE_ROOT = ROOT / "data" / "tampar_sample"
IMAGE_ROOT_DISTORTED = IMAGE_ROOT / "distortions"
IMAGE_ROOT_DISTORTED.mkdir(exist_ok=True)
DISTORTION_VALUES = [-0.08, -0.04, -0.02, 0.04, 0.08, 0.16]
KEYPOINT_COLORS = np.linspace(50, 255, 8).astype(np.uint8)


def compute_new_keypoint_annotations(img: np.ndarray):
    keypoints = []
    for color in KEYPOINT_COLORS:
        mask = img == color
        if np.sum(mask) <= 20:
            return None
        mask = mask.astype(np.uint8) * 255

        # Clean up noise
        kernel_size = 5  # Adjust this based on the size of the noise to remove
        eroded_image = cv2.erode(
            mask,
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1,
        )
        _, cleaned_mask = cv2.threshold(eroded_image, 100, 255, cv2.THRESH_BINARY)

        # Extract keypoints
        true_indices = np.argwhere(cleaned_mask)
        mean = np.mean(true_indices, axis=0)  # y, x
        keypoints.append([*mean.tolist()[::-1], 2])
    return sum(keypoints, [])


def compute_new_segmentation_annotations(img: np.ndarray):
    # BBox
    true_indices = np.argwhere(img > 0)
    if len(true_indices) > 0:
        y_min, x_min = np.min(true_indices, axis=0)
        y_max, x_max = np.max(true_indices, axis=0)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        bbox = [float(i) for i in bbox]
    else:
        bbox = []

    # Segmentation
    img[img > 0.1] == 1
    img[img <= 0.1] == 0
    polygons = binary_mask_to_polygon(img)
    return polygons, bbox


def binary_mask_to_polygon(binary_mask: np.ndarray, tolerance: float = 0):
    """Converts a binary mask to COCO polygon representation
    from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    Apache License 2.0, https://github.com/waspinator/pycococreator/blob/master/LICENSE


    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """

    def close_contour(contour: np.ndarray):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def create_keypoint_annotation_image(image_info: dict, keypoints: np.ndarray):
    white_img = np.zeros([image_info["height"], image_info["width"]])
    for idx, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        color = (int(KEYPOINT_COLORS[idx]), int(KEYPOINT_COLORS[idx]))
        cv2.circle(white_img, (x, y), 50, color, -1)
    white_img = white_img.astype(np.uint8)
    return white_img


def create_segm_annotation_image(image_info: dict, points: np.ndarray):
    img = np.zeros([image_info["height"], image_info["width"]])
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color=255)
    return img


def main(
    annotation_filename: str = "tampar_sample_validation.json",
    distortion_values: List = None,
):
    distortion_values = (
        DISTORTION_VALUES if distortion_values is None else distortion_values
    )
    for i, val in enumerate(distortion_values):
        print(f"### {i}: Distortion with {val}")
        info = json.loads((IMAGE_ROOT / annotation_filename).read_text())
        new_annotations = []
        new_image_infos = []
        for image_info in tqdm.tqdm(info["images"]):
            # Load data
            image_path = IMAGE_ROOT / image_info["file_name"]
            annotations = [
                a for a in info["annotations"] if a["image_id"] == image_info["id"]
            ][0]

            # Keypoints
            keypoints = np.array(annotations["keypoints"]).reshape(-1, 3)[:, :2]
            img_anno_kp = create_keypoint_annotation_image(image_info, keypoints)
            # Segmentation
            segm = np.array(annotations["segmentation"][0]).reshape(-1, 2)
            img_anno_seg = create_segm_annotation_image(image_info, segm)

            images_before = {
                "input": cv2.imread(image_path.as_posix()),
                "anno_kp": img_anno_kp,
                "anno_seg": img_anno_seg,
            }

            # Adjust
            images_after = {}
            new_image_path = (
                IMAGE_ROOT_DISTORTED / str(i).zfill(2) / image_info["file_name"]
            )
            new_image_path.parent.mkdir(exist_ok=True, parents=True)
            crop_area = None
            for im_type, img in images_before.items():
                # cv2.imwrite(f"before_{im_type}.jpg", img)
                with Image.from_array(img) as img_wand:
                    img_wand.virtual_pixel = "transparent"
                    img_wand.distort("barrel", (val, 0.0, 0.0, 1.0))
                    img_distorted = np.array(img_wand)
                    if im_type != "input":
                        img_distorted = img_distorted[:, :, 0]
                    else:
                        crop_area = PILImage.fromarray(img_distorted).getbbox()
                    images_after[im_type] = img_distorted[
                        crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]
                    ]

            cv2.imwrite(new_image_path.as_posix(), images_after["input"])
            keypoints = compute_new_keypoint_annotations(images_after["anno_kp"])
            segmentation, bbox = compute_new_segmentation_annotations(
                images_after["anno_seg"]
            )
            if keypoints is None or len(keypoints) != 8 * 3:
                continue
            annotations["segmentation"] = segmentation
            annotations["keypoints"] = keypoints
            annotations["bbox"] = bbox
            annotations["area"] = bbox[3] * bbox[2]
            new_annotations.append(annotations)

            image_info["file_name"] = new_image_path.relative_to(
                IMAGE_ROOT_DISTORTED
            ).as_posix()
            image_info["height"] = images_after["anno_kp"].shape[0]
            image_info["width"] = images_after["anno_kp"].shape[1]
            new_image_infos.append(image_info)
        info["annotations"] = new_annotations
        info["images"] = new_image_infos
        (
            IMAGE_ROOT_DISTORTED
            / annotation_filename.replace(
                "tampar", f"tampar_distorted_{str(i).zfill(2)}"
            )
        ).write_text(json.dumps(info))


if __name__ == "__main__":
    main()
