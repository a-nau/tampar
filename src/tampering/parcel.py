import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())


import cv2
import numpy as np
from cv2 import aruco
from skimage.restoration import estimate_sigma

from src.tampering.utils import compute_keypoint_mask, get_side_surface_patches
from src.utils.tampering_vis import (
    get_all_ordered_keypoints,
    get_perspective_transform,
    visualize_parcel_side_surfaces,
)

IMAGE_ROOT = ROOT / "imgs"
OUT_IMAGES = IMAGE_ROOT / "out"
OUT_IMAGES.mkdir(exist_ok=True, parents=True)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 5  # default: 3
parameters.adaptiveThreshWinSizeMax = 50  # default: 23
parameters.adaptiveThreshConstant = 25  # default: 7
parameters.cornerRefinementWinSize = 10  # default: 5

PATCH_SIZE = 400
PATCH_ORDER = [
    "",
    "top",
    "",
    "left",
    "center",
    "right",
    "",
    "bottom",
    "",
]
WHITE_PATCH = np.ones((PATCH_SIZE, PATCH_SIZE, 3)) * 255


@dataclass
class SideSurface:
    image: np.ndarray  # normalized
    name: str  # according to ArUco orientation
    mask: np.ndarray  # original image
    keypoints: np.ndarray

    @property
    def score(self):
        rel_area = self.area / (self.mask.shape[0] * self.mask.shape[1])
        if self.noise_level > 2.5:
            rel_area /= 2
        return rel_area

    @property
    def area(self):
        return np.sum(self.mask / 255)

    @property
    def convex_hull(self):
        mask = np.zeros_like(self.mask).astype(np.float32)
        hull = cv2.convexHull(self.keypoints.astype(np.float32))
        convex_hull = hull.astype(np.int32).reshape(-1, 2)  # fill needs int input
        mask = cv2.fillConvexPoly(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), convex_hull, tuple([255] * 3)
        )
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    @property
    def convexness(self):
        return self.area / np.sum(self.convex_hull > 0)

    @property
    def rectangleness(self):
        return self.area / np.sum(self.min_area_rectangle > 0)

    @property
    def angles(self):
        center = self.keypoints[1]
        kps = self.keypoints[[0, 2], ...] - center
        idxs = [0, 1]
        angles_ = []
        for i, axis in enumerate([[0, 1], [1, 0]]):
            for j, idx in enumerate(idxs):
                angles_.append(angle_between_vectors(kps[idx, ...], np.array(axis)))
        best = np.argmin(angles_)
        if best == 0 or best == 3:
            return [angles_[0], angles_[3]]
        elif best == 1 or best == 2:
            return [angles_[1], angles_[2]]

    @property
    def min_area_rectangle(self):
        # Compute the minimum area rectangle
        rect = cv2.minAreaRect(self.keypoints.astype(np.float32))

        # Draw the minimum area rectangle
        mask = np.zeros_like(self.mask).astype(np.float32)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    @property
    def noise_level(self):
        noise_level = estimate_sigma(self.image, average_sigmas=True)
        return noise_level


class ParcelView:
    output_size: int = PATCH_SIZE

    def __init__(self, image_path: Path, keypoints: np.ndarray) -> None:
        self.image_path = image_path
        self.image_id = "_".join(self.image_path.stem.split("_")[2:])
        self.parcel_id = int(self.image_path.stem.split("_")[1])
        image = cv2.imread(str(self.image_path))
        self.uv_map = None
        self.image_size = image.shape[:2]
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.keypoints = keypoints
        self.side_surface_mapping = {}
        self.side_surfaces: dict[SideSurface] = {}
        try:
            self.parcel_corners = get_all_ordered_keypoints(self.keypoints)
            success = self.find_aruco_plane()
        except:
            success = False
        if success:
            self.initialize_side_surfaces()

    def initialize_side_surfaces(self):
        self.front, self.top, self.side = visualize_parcel_side_surfaces(
            self.keypoints,
            self.image,
            pad_size=0,
            output_size=(self.output_size, self.output_size),
        )
        self.uv_map = self.get_uvmap_full()
        patches = get_side_surface_patches(self.uv_map)
        for name, patch in zip(
            PATCH_ORDER,
            patches,
        ):
            if name != "" and np.mean(patch) < 252:
                self.side_surfaces[name] = SideSurface(
                    image=patch,
                    name=name,
                    mask=compute_keypoint_mask(
                        self.parcel_corners[self.side_surface_mapping[name]],
                        self.image_size,
                    ),
                    keypoints=self.parcel_corners[self.side_surface_mapping[name]],
                )

    @property
    def aruco_plane(self):
        return self.__dict__[self.aruco_plane_name]

    def find_aruco_plane(self):
        corners, ids, rejected = aruco.detectMarkers(
            self.image, dictionary, parameters=parameters
        )
        if len(corners) != 1:
            print(
                f"WARNING: <>1 ArUco found for {self.image_path.name} (len: {len(corners)})"
            )
            self.aruco_corners = None
            return None
        self.image_aruco = aruco.drawDetectedMarkers(self.image.copy(), corners, ids)
        self.aruco_corners = corners[0].squeeze()
        self.id = ids[0]

        # find on which plane aruco is
        for name, corners in self.parcel_corners.items():
            hull = cv2.convexHull(corners.astype(np.float32).reshape(-1, 1, 2))
            matches = [
                cv2.pointPolygonTest(hull, p, False)
                for p in self.aruco_corners.tolist()
            ]
            if sum(matches) > 0:
                self.aruco_plane_name = name
                return self.compute_rotation_according_to_aruco()
        return False

    def compute_rotation_according_to_aruco(self):
        points = self.aruco_corners.astype(np.float32).reshape(-1, 1, 2)
        perspective_matrix = get_perspective_transform(
            self.parcel_corners[self.aruco_plane_name],
            output_size=(self.output_size, self.output_size),
        )
        self.aruco_corners_tranformed = cv2.perspectiveTransform(
            points, perspective_matrix
        ).squeeze()

        # Compute the angle between the marker and the X-axis of the image
        u = self.aruco_corners_tranformed[0, :] - self.aruco_corners_tranformed[3, :]
        u = u / np.linalg.norm(u)
        angle = np.arctan2(u[0], u[1])
        if np.isnan(angle):
            return False
        angle = angle * 180 / np.pi
        angle = round(angle / 90) * 90
        self.angle = -angle + 180
        return True

    def get_uvmap_full(self):
        image = (
            np.ones((self.output_size * 3, self.output_size * 3, 3), dtype=float) * 255
        )
        image[
            self.output_size : 2 * self.output_size,
            self.output_size : 2 * self.output_size,
        ] = self.aruco_plane
        self.side_surface_mapping["center"] = self.aruco_plane_name
        if self.aruco_plane_name == "top":
            self.side_surface_mapping["bottom"] = "front"
            image[
                self.output_size * 2 :, self.output_size : 2 * self.output_size
            ] = self.front
            self.side_surface_mapping["right"] = "side"
            image[
                self.output_size : 2 * self.output_size,
                self.output_size * 2 :,
            ] = cv2.rotate(self.side, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.aruco_plane_name == "front":
            self.side_surface_mapping["right"] = "side"
            image[
                self.output_size : 2 * self.output_size, self.output_size * 2 :
            ] = self.side
            self.side_surface_mapping["top"] = "top"
            image[
                : self.output_size, self.output_size : 2 * self.output_size
            ] = self.top
        elif self.aruco_plane_name == "side":
            self.side_surface_mapping["left"] = "front"
            image[
                self.output_size : 2 * self.output_size, : self.output_size
            ] = self.front
            self.side_surface_mapping["top"] = "top"
            image[
                : self.output_size, self.output_size : 2 * self.output_size
            ] = cv2.rotate(self.top, cv2.ROTATE_90_CLOCKWISE)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
        rotated_img = cv2.warpAffine(image, M, (cols, rows))
        self.rotate_mapping()
        return rotated_img.astype(np.float32)

    def rotate_mapping(self):
        num_rotations = self.angle // 90
        direction = 1 if self.angle < 0 else 0
        if direction == 1:  # clockwise
            rot = {
                "top": "right",
                "right": "bottom",
                "bottom": "left",
                "left": "top",
                "center": "center",
            }
        else:
            rot = {
                "top": "left",
                "left": "bottom",
                "bottom": "right",
                "right": "top",
                "center": "center",
            }
        for _ in range(num_rotations):
            new_dict = {}
            for key, value in self.side_surface_mapping.items():
                new_dict[rot[key]] = value
            self.side_surface_mapping = new_dict


class Parcel:
    def __init__(self, views: List[ParcelView]):
        self.views = views
        self.uvmap = self.create_uvmap_from_views()

    def create_uvmap_from_views(self):
        if len(self.views) <= 1:
            self.uvmap = None
        else:
            mapping = {key: [] for key in PATCH_ORDER}
            for view in self.views:
                for name in PATCH_ORDER:
                    pss = view.side_surfaces.get(name, None)
                    if pss is not None:
                        mapping[name].append(view.side_surfaces[name])
            # Compose image
            uvmap = WHITE_PATCH
            for name in PATCH_ORDER[1:]:
                candidates = mapping[name]
                if len(candidates) > 0:
                    i_max = np.argmax([c.score for c in candidates])
                    patch = candidates[i_max].image
                else:
                    patch = WHITE_PATCH
                uvmap = np.hstack((uvmap, patch))
            uvmap = hstacked_to_grid(uvmap)
            return uvmap.astype(np.float32)

    def get_uvmap_full(self):
        return self.uvmap


def hstacked_to_grid(image: np.ndarray):
    grid = np.vstack(
        (
            [
                image[
                    :,
                    i * 3 * PATCH_SIZE : (i + 1) * 3 * PATCH_SIZE,
                    ...,
                ]
                for i in range(3)
            ]
        )
    )
    return grid.astype(int)


def angle_between_vectors(vector1, vector2):
    cosine_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees
