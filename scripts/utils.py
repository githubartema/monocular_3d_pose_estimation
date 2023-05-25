import copy
import glob
import json
import os
from typing import Any, Deque, List, Tuple

import cv2
import numpy as np
from numpy.linalg import norm

from pose_estimation.core.camera import *


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj: object) -> object:
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def detect_clap(
    l_wrist: np.ndarray,
    r_wrist: np.ndarray,
    l_elbow: np.ndarray,
    r_elbow: np.ndarray,
    neck: np.ndarray,
    last_clap_idx: int,
    idx: int,
    buffer: list,
) -> Tuple[Deque[int], int, bool]:

    """
    Clap detection method.

    Used heuristic idea is simple enough: to check if both left and right wrists are above neck (head)
    and whether distance between them (L2 norm) is less than some threshold
    (in this case averaged distance between wrists and elbows is considered).

    Additional conditions SHOULD BE MET in order not to detect
    multiple claps for sequential frames.

    :type l_wrist:np.ndarray:
    :param l_wrist:np.ndarray: 3D position of left wrist joint.

    :type r_wrist:np.ndarray:
    :param r_wrist:np.ndarray: 3D position of right wrist joint.

    :type l_elbow:np.ndarray:
    :param l_elbow:np.ndarray: 3D position of left elbow joint.

    :type r_elbow:np.ndarray:
    :param r_elbow:np.ndarray: 3D position of right elbow joint.

    :type last_clap_idx:int:
    :param last_clap_idx:int: index of the last frame with detected clap.

    :type idx:int:
    :param idx:int: index of the current frame.

    :type buffer:Deque:
    :param buffer:Deque: deque that contains info about claps.

    :raises:

    :rtype: None
    """

    buffer_length = len(buffer)

    wrist_dist_less_than_avg_elbow_dist = (
        norm(l_wrist - r_wrist)
        < (norm(l_wrist - l_elbow) + norm(r_wrist - r_elbow)) / 2
    )
    wrists_above_neck = l_wrist[2] > neck[2] and r_wrist[2] > neck[2]

    if wrist_dist_less_than_avg_elbow_dist and wrists_above_neck:
        if (idx - last_clap_idx) < buffer_length:
            buffer.append(1)
            buffer.popleft()

            return buffer, last_clap_idx, False
        else:
            if sum(buffer) < buffer_length / 2:
                buffer.append(1)
                buffer.popleft()
                last_clap_idx = idx

                return buffer, last_clap_idx, True
            else:
                buffer.append(1)
                buffer.popleft()

                return buffer, last_clap_idx, False
    else:
        buffer.append(0)
        buffer.popleft()

        return buffer, last_clap_idx, False


def show2Dpose(kps: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Runs plotting of predicted 2D joints and skeleton.
    """
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
    ]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor, rcolor = (255, 0, 0), (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = list(map(int, kps[c[0]]))
        end = list(map(int, kps[c[1]]))

        cv2.line(
            img,
            (start[0], start[1]),
            (end[0], end[1]),
            lcolor if LR[j] else rcolor,
            thickness,
        )

        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax) -> None:
    """
    Runs predicted 3D joints and skeleton plotting.
    """
    ax.view_init(elev=15.0, azim=70)

    lcolor, rcolor = (0, 0, 1), (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS, RADIUS_Z = 0.72, 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect("equal")  # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params("x", labelbottom=False)
    ax.tick_params("y", labelleft=False)
    ax.tick_params("z", labelleft=False)


def preprocess_2d_joints(
    joints_left: List[int],
    joints_right: List[int],
    input_2D_no: np.ndarray,
    img_size: Tuple[Any],
) -> np.ndarray:
    """
    Runs predicted 2D skeleton joints preprocessing.
    """
    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[:, :, 0] *= -1
    input_2D_aug[:, joints_left + joints_right] = input_2D_aug[
        :, joints_right + joints_left
    ]
    input_2D = np.concatenate(
        (np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0
    )
    input_2D = input_2D[np.newaxis, :, :, :, :]

    return input_2D


def postprocess_3d_joints(
    joints_left: List[int],
    joints_right: List[int],
    output_3D_non_flip: torch.Tensor,
    output_3D_flip: torch.Tensor,
    pad: int,
) -> torch.Tensor:
    """
    Runs predicted 3D skeleton joints postprocessing.
    """
    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[
        :, :, joints_right + joints_left, :
    ]
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    output_3D = output_3D[0:, pad].unsqueeze(1)
    output_3D[:, :, 0, :] = 0

    return output_3D


def img2video(video_path: str, output_path: str) -> None:
    """
    Turns sequence of images into video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # + 5

    print(f"FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    names = sorted(glob.glob(os.path.join(output_path, "*.png")))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_path + ".avi", fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()
