import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np
import tqdm
from detectron2.engine import DefaultPredictor

from src.maskrcnn.data import register_datasets  # noqa
from src.tampering.parcel import Parcel, ParcelView

IMAGE_ROOT = ROOT / "data" / "tampar_sample"
UVMAP_DIR = IMAGE_ROOT / "uvmaps"
UVMAP_DIR.mkdir(exist_ok=True)


def save_uvmap(
    image_path: Path,
    output_path: Path,
    keypoints: np.ndarray = None,
    predictor: DefaultPredictor = None,
):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if keypoints is None and predictor is not None:
        outputs = predictor(img_rgb)
        if len(outputs["instances"].pred_keypoints) == 0:
            return None
        keypoints = outputs["instances"].pred_keypoints[0, :, :2].cpu().numpy()
    view = ParcelView(
        image_path,
        np.array(keypoints),
    )
    if view.uv_map is not None:
        cv2.imwrite(
            output_path.as_posix(), cv2.cvtColor(view.uv_map, cv2.COLOR_RGB2BGR)
        )
    return view


def create_pred_uvmaps(predictor: DefaultPredictor, image_paths: List[Path]):
    for i, img_path in tqdm.tqdm(enumerate(image_paths)):
        new_path = img_path.parent / f"{img_path.stem}_uvmap_pred.png"
        save_uvmap(img_path, new_path, keypoints=None, predictor=predictor)


def create_gt_uvmaps(coco_annotations, groundtruth=True):
    base_views = {i: [] for i in range(30)}
    view_infos = []
    identification_string = "gt" if groundtruth else "pred"
    for image_info in tqdm.tqdm(coco_annotations["images"]):
        rel_image_path = Path(image_info["file_name"])
        image_path = IMAGE_ROOT / rel_image_path
        image_id = image_info["id"]
        annotations = [
            a for a in coco_annotations["annotations"] if a["image_id"] == image_id
        ]
        if len(annotations) == 1:
            keypoints = np.array(annotations[0]["keypoints"]).reshape(-1, 3)[..., :2]
            new_path = (
                image_path.parent
                / f"{image_path.stem}_uvmap_{identification_string}.png"
            )
            view = save_uvmap(image_path, new_path, keypoints=keypoints, predictor=None)
            if view is not None and image_path.parent.name == "base":
                base_views[view.parcel_id].append(view)
            for name, sidesurface in view.side_surfaces.items():
                view_infos.append(
                    {
                        "image_path": view.image_path.relative_to(
                            IMAGE_ROOT
                        ).as_posix(),
                        "image_id": view.image_id,
                        "name_uvmap": name,
                        "convexness": sidesurface.convexness,
                        "rectangleness": sidesurface.rectangleness,
                        "area": sidesurface.area,
                        "score": sidesurface.score,
                        # "mask": sidesurface.mask.tolist(),
                        "keypoints_side": sidesurface.keypoints.tolist(),
                        "name_keypoints": view.side_surface_mapping[name],
                        "keypoints_parcel": view.keypoints.tolist(),
                        "angles": sidesurface.angles,
                    }
                )

    if groundtruth:
        for parcel_id, views in base_views.items():
            if len(views) == 0:
                continue
            parcel = Parcel(views)
            uvmap = parcel.uvmap
            if uvmap is not None:
                new_path = UVMAP_DIR / f"id_{str(parcel_id).zfill(2)}_uvmap.png"
                cv2.imwrite(new_path.as_posix(), cv2.cvtColor(uvmap, cv2.COLOR_RGB2BGR))
    return view_infos
