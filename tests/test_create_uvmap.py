import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np

from src.tampering.parcel import Parcel
from src.tools.create_uvmaps import save_uvmap

logger = logging.getLogger(__name__)

annotations = [
    {
        "image_id": 30,
        "keypoints": np.array(
            [
                2213.47,
                1307.37,
                1163.05,
                1551.43,
                851.11,
                299.97,
                2132.67,
                2746.9,
                2831.63,
                990.05,
                1533.54,
                1425.96,
                1351.84,
                176.74,
                2618.89,
                2394.88,
            ]
        ).reshape(-1, 2),
    },
    {
        "image_id": 25,
        "keypoints": np.array(
            [
                1776.94,
                2161.58,
                613.58,
                1541.57,
                419.76,
                1040.32,
                1832.77,
                2642.73,
                3397.09,
                839.63,
                2132.44,
                775.52,
                2136.21,
                353.37,
                3249.72,
                1281.71,
            ]
        ).reshape(-1, 2),
        "num_keypoints": 8,
    },
]


class TestCreateUVMap(unittest.TestCase):
    def test_create_uvmap(self):
        print("Testing UV map creation from single image")
        save_uvmap(
            image_path=ROOT / "data/tampar_sample/validation/id_01_20230516_142710.jpg",
            output_path=ROOT / "test.png",
            keypoints=annotations[0]["keypoints"],
            predictor=None,
        )

        uv_map = cv2.imread(str(ROOT / "test.png"))
        compare_image = cv2.imread(
            str(ROOT / "data/misc/id_01_20230516_142710_uvmap_gt.png")
        )
        assert np.allclose(
            uv_map, compare_image
        ), f"Failed average pixel distance is: {np.mean(np.abs(uv_map-compare_image))}"

    def test_create_parcel_uvmap(self):
        print("Testing UV map creation from multiple images")
        img1 = ROOT / "data/tampar_sample/validation/id_01_20230516_142710.jpg"
        view1 = save_uvmap(
            (img1),
            (ROOT / "test1.png"),
            keypoints=annotations[0]["keypoints"],
        )
        img2 = ROOT / "data/misc/id_01_20230516_142642.jpg"
        view2 = save_uvmap(
            (img2),
            (ROOT / "test2.png"),
            keypoints=annotations[1]["keypoints"],
        )
        parcel = Parcel([view1, view2])
        uvmap = parcel.uvmap
        if uvmap is not None:
            cv2.imwrite(
                str(ROOT / "test1+2.png"), cv2.cvtColor(uvmap, cv2.COLOR_RGB2BGR)
            )

        compare_image = cv2.imread(str(ROOT / "data/misc/test1+2.png"))
        uv_map = cv2.imread(str(ROOT / "test1+2.png"))
        assert np.allclose(
            uv_map, compare_image
        ), f"Failed average pixel distance is: {np.mean(np.abs(uv_map-compare_image))}"
