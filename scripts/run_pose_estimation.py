import argparse
import copy
import glob
import json
import multiprocessing
import os
import sys
import time
from collections import deque
from multiprocessing import Event, Process, Queue
from queue import Empty
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import *

from pose_estimation.core import Model
from pose_estimation.core.camera import *
from pose_estimation.utils import (get_keypoints_hrnet, h36m_coco_format,
                                   revise_kpts)

plt.switch_backend("agg")


class ResultsProcessor(Process):
    def __init__(self, queue: Queue):
        super(ResultsProcessor, self).__init__()

        self.queue = queue
        self.exit = Event()

    def run(self):
        while not self.exit.is_set():
            try:
                image_2D, post_out, clap_counter, output_path, indx = self.queue.get(
                    False
                )
                if image_2D is None:
                    break

                self.processing(image_2D, post_out, clap_counter, output_path, indx)
            except Empty:
                time.sleep(1)
        return

    def stop(self):
        self.exit.set()

    def processing(self, image_2D, post_out, clap_counter, output_path, indx) -> None:
        """
        Saves plot with 2D and 3D results to image
        """
        fig = plt.figure(figsize=(9.6, 5.4))

        ax_2D = fig.add_subplot(1, 2, 1)
        ax_2D.imshow(np.flip(image_2D, axis=2))
        ax_2D.set_title("Detected 2D keypoints + skeleton")

        ax_3D = fig.add_subplot(1, 2, 2, projection="3d")
        ax_3D.set_title(
            f"Reconstructed 3D skeleton, \n Amount of claps above neck point: {clap_counter}"
        )
        show3Dpose(post_out, ax_3D)

        output_path_2d_3d = os.path.join(output_path, "2d+3d")
        os.makedirs(output_path_2d_3d, exist_ok=True)
        plt.savefig(
            os.path.join(output_path_2d_3d, str(("%04d" % indx))),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def get_pose2D(video_path: str, output_path: str) -> None:
    """
    Runs landmark detector (HRNet + YOLOv3) to predict landmarks.

    :type video_path:str:
    :param video_path:str: path to video to process.

    :type video_path:str:
    :param video_path:str: path to output directory.

    :raises:

    :rtype: None
    """
    cap = cv2.VideoCapture(video_path)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    print("\n Running landmarks detection...\n")
    with torch.no_grad():
        # the first frame of the video should be detected a person
        joints_2D, scores = get_keypoints_hrnet(
            video=video_path, det_dim=416, num_peroson=1, gen_output=True
        )

    joints_2D, scores, valid_frames = h36m_coco_format(joints_2D, scores)
    revised_joints_2D = revise_kpts(joints_2D, scores, valid_frames).tolist()

    print("\n Landmarks have been detected successfully!\n")

    with open(
        os.path.join(output_path, "body_2d_landmarks.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"landmarks": revised_joints_2D},
            f,
            ensure_ascii=False,
            indent=4,
            cls=NumpyEncoder,
        )


def get_pose3D(video_path: str, output_path: str, checkpoint_path: str) -> None:
    """
    Inferences pretrained MHFormer to predict 3D skeleton poses for the specified video frames.
    Default MHFormer checkpoint is with receptive field of 81 frames.

    :type video_path:str:
    :param video_path:str: path to video to process.

    :type video_path:str:
    :param video_path:str: path to output directory.

    :type checkpoint_path:str:
    :param checkpoint_path:str: path to MHFormer checkpoint.

    :raises:

    :rtype: None
    """
    args, _ = argparse.ArgumentParser().parse_known_args()

    processors, queues = [], []

    # to speed up plots saving
    for _ in range(multiprocessing.cpu_count()):
        queue = Queue()
        processor = ResultsProcessor(queue)
        processor.start()
        processors.append(processor)
        queues.append(queue)

    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 81
    args.pad = (args.frames - 1) // 2
    args.n_joints, args.out_joints = 17, 17

    model = Model(args).cuda()
    model_path = sorted(glob.glob(os.path.join(checkpoint_path, "*.pth")))[0]
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model)
    model.eval()

    # loading predicted 2d body landmarks
    with open(os.path.join(output_path, "body_2d_landmarks.json")) as f_in:
        joints_2D = np.array(json.load(f_in)["landmarks"])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    action_buffer_length = int(fps / 3)
    action_buffer = deque([0 for _ in range(action_buffer_length)])
    last_clap_idx, clap_counter = -1, 0

    R = np.array(
        [
            0.1407056450843811,
            -0.1500701755285263,
            -0.755240797996521,
            0.6223280429840088,
        ],
        dtype="float32",
    )

    print("\n Reconstructing 3D skeleton poses...\n")

    joints_3D = []

    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        img_size = img.shape

        start = max(0, i - args.pad)
        end = min(i + args.pad, len(joints_2D[0]) - 1)

        input_2D_no = joints_2D[0][start : end + 1]

        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(joints_2D[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(joints_2D[0]) - 1)

            input_2D_no = np.pad(
                input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), "edge"
            )

        joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

        input_2D = preprocess_2d_joints(
            joints_left, joints_right, input_2D_no, img_size
        )
        input_2D = torch.from_numpy(input_2D.astype("float32")).cuda()

        ## 3D skeleton pose estimation
        output_3D_non_flip, output_3D_flip = model(input_2D[:, 0]), model(
            input_2D[:, 1]
        )
        output_3D = postprocess_3d_joints(
            joints_left, joints_right, output_3D_non_flip, output_3D_flip, args.pad
        )
        post_out = output_3D[0, 0].cpu().detach().numpy()
        joints_3D.append(post_out)

        post_out = camera_to_world(post_out, R=R, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        if i >= action_buffer_length:
            action_buffer, last_clap_idx, is_clapped = detect_clap(
                np.array(post_out[13]),  # l_wrist
                np.array(post_out[16]),  # r_wrist
                np.array(post_out[12]),  # l_elbow
                np.array(post_out[15]),  # r_elbow
                np.array(post_out[8]),  # neck
                last_clap_idx,
                i,
                action_buffer,
            )

            if is_clapped:
                clap_counter += 1

        input_2D_no = input_2D_no[args.pad]
        image_2D = show2Dpose(input_2D_no, copy.deepcopy(img))

        # do visual results saving in separated processes to speed that up
        queues[i % len(queues)].put([image_2D, post_out, clap_counter, output_path, i])

    print(
        f"\n Creating output video with visualization ({multiprocessing.cpu_count()} processes), this may take a few minutes...\n"
    )

    for i, queue in enumerate(queues):
        queue.put([None for _ in range(5)])
        processors[i].join()
        queue.close()
        queue.join_thread()

    joints_3D = np.stack(joints_3D, axis=0).tolist()

    # dump 3D body landmarks
    with open(
        os.path.join(output_path, "body_3d_landmarks.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"landmarks": joints_3D}, f, ensure_ascii=False, indent=4, cls=NumpyEncoder
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="path to input video")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../mhformer_checkpoints/receptive_field_81",
        help="path to MHFormer checkpoints",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_path = os.path.join(
        "../data/output", os.path.basename(args.video).split(".")[0]
    )
    os.makedirs(output_path, exist_ok=True)

    get_pose2D(args.video, output_path)
    get_pose3D(args.video, output_path, args.checkpoint_path)
    img2video(args.video, os.path.join(output_path, "2d+3d"))

    print("All done!")
