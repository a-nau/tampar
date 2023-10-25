import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely import LinearRing, is_ccw


def get_ordered_corners(corners):
    poly_points = np.vstack((corners, corners[0]))
    polygon = LinearRing(poly_points)
    if is_ccw(polygon):
        sorted_corners = corners
    else:
        sorted_corners = corners[[2, 1, 0, 3], :]
    return sorted_corners


def draw_corners(corners, img_):
    p_old = None
    for i, p in enumerate(corners.astype(int).tolist()):
        cv2.circle(img_, p, (i + 1) * 3, (255, 0, 0), -1)
        if p_old is not None:
            cv2.line(img_, p, p_old, (0, 255, 0), 2)
        p_old = p
    plt.imshow(img_)


def get_perspective_transform(corners, output_size=(300, 300)):
    target_corners = np.array(
        [
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ]
    )
    M = cv2.getPerspectiveTransform(
        corners.astype(np.float32), target_corners.astype(np.float32)
    )
    return M


def apply_perspective_transform_to_image(corners, image, output_size=(300, 300)):
    M = get_perspective_transform(corners, output_size)
    output_image = cv2.warpPerspective(image, M, output_size)
    return output_image


def get_all_ordered_keypoints(keypoints):
    planes = {
        "front": keypoints[[2, 0, 3, 1], :],
        "top": keypoints[[6, 4, 0, 2], :],
        "side": keypoints[[0, 4, 7, 3], :],
    }
    keypoint_overview = {}
    for plane_name, plane_keypoints in planes.items():
        corners = get_ordered_corners(plane_keypoints)
        keypoint_overview[plane_name] = corners
    return keypoint_overview


def visualize_parcel_side_surfaces(keypoints, img_, pad_size=6, output_size=(300, 300)):
    imgs = []
    parcel_corners = get_all_ordered_keypoints(keypoints)
    for name, corners in parcel_corners.items():
        try:
            output_image = apply_perspective_transform_to_image(
                corners, img_, output_size
            )
            if pad_size > 0:
                output_image = np.pad(
                    output_image,
                    ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    constant_values=255,
                )
                green, yellow, blue = (0, 255, 0), (255, 255, 0), (0, 0, 255)
                if name == "front":
                    colors = [green] * 4
                elif name == "top":
                    colors = [yellow, blue, green, blue]
                elif name == "side":
                    colors = [blue, yellow, blue, green]
                else:
                    raise
                output_image[0:, -pad_size:, :] = colors[1]
                output_image[0:, 0:pad_size, :] = colors[3]
                output_image[0:pad_size, 0:, :] = colors[0]
                output_image[-pad_size:, 0:, :] = colors[2]

            imgs.append(output_image)
        except Exception as e:
            raise e
    return imgs
