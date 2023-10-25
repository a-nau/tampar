from functools import partial
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

meta = [
    {"name": "normal box", "color": [255, 255, 25], "id": 0},  # noqa
    {"name": "damaged box", "color": [230, 25, 75], "id": 1},  # noqa
]


def register_dataset(dataset_name: str, json_file: Path, image_root: Path):
    DatasetCatalog.register(
        dataset_name, partial(load_coco_json, json_file, image_root)
    )
    # Set meta data
    things_ids = [k["id"] for k in meta]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in meta]
    thing_colors = [k["color"] for k in meta]
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "keypoint_names": [
            "front_intersect3_inside",
            "front_intersect2",
            "front_intersect3_left",
            "front_intersect3_right",
            "back_intersect3_outside",
            "back_hidden",
            "back_intersect2_left",
            "back_intersect2_right",
        ],
        "keypoint_connection_rules": [  # using BGR
            # Back
            ("back_intersect3_outside", "back_intersect2_left", (255, 255, 0)),
            ("back_intersect3_outside", "back_intersect2_right", (255, 255, 0)),
            # Sides
            ("front_intersect3_inside", "back_intersect3_outside", (0, 0, 255)),
            ("front_intersect3_left", "back_intersect2_left", (0, 0, 255)),
            ("front_intersect3_right", "back_intersect2_right", (0, 0, 255)),
            # Front
            ("front_intersect3_inside", "front_intersect3_left", (0, 255, 0)),
            ("front_intersect3_inside", "front_intersect3_right", (0, 255, 0)),
            ("front_intersect2", "front_intersect3_left", (0, 255, 0)),
            ("front_intersect2", "front_intersect3_right", (0, 255, 0)),
            # Hidden
            ("back_hidden", "back_intersect2_left", (255, 255, 200)),  # back
            ("back_hidden", "back_intersect2_right", (255, 255, 200)),  # back
            ("front_intersect2", "back_hidden", (200, 200, 255)),  # side
        ],
        "keypoint_flip_map": [],
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=str(json_file.resolve()), image_root=image_root.as_posix(), **metadata
    )


dataset_paths = {
    "tampar_sample": ROOT / "data" / "tampar_sample",
}

for dataset_name, dataset_path in dataset_paths.items():
    # We register all encountered json files from the folders
    for json_file in dataset_path.glob("*.json"):
        dataset_split_name = f"{json_file.stem}"
        print(f"Registering: {dataset_split_name}")
        register_dataset(
            dataset_name=dataset_split_name,
            json_file=json_file,
            image_root=dataset_path,
        )
print("Registered all datasets")
