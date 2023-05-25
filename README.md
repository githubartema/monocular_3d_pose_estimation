# Monocular 3D Human Pose Estimation and Clapping Action Recognition

Upon reviewing the papers available on [paperswithcode.com](https://paperswithcode.com/sota/3d-human-pose-estimation-on-human36m), it becomes evident that the majority of SOTA Monocular 3D Human Pose Estimation pipelines are increasingly leveraging Transformer-based models. Despite significant strides in 3D human pose estimation with monocular systems, a considerable performance gap with multi-view systems still persists.

This repository is dedicated to the implementation of the Monocular 3D Human Pose Estimation Pipeline based on the paper [**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation**](https://arxiv.org/pdf/2111.12707) (Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, and Luc Van Gool, IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in 2022).

While the authors' original research code and pre-trained models form the foundation for this repository, some refactoring and optimizations have been carried out to transform "research-like" code into a more "inference-ready" format.

Furthermore, a simple yet powerful approach for the action recognition of **clapping above the head** has been proposed. This heuristic algorithm is based on tracking the predicted 3D positions of wrist joints. A **clapping above the head** action is detected when the distance between these joints is less than a certain threshold. Several additional conditions must be met to avoid detecting multiple clapping actions in sequential frames. However, it is important to note that there are definitely more powerful deep learning-based approaches available, as detailed here: [paper](https://www.nature.com/articles/s41598-022-09293-8), [paper](https://link.springer.com/article/10.1007/s11042-022-14214-y) and [paper](https://arxiv.org/pdf/1807.02131.pdf).

For more detailed information on the original research and model, please refer to the [MHFormer paper](https://arxiv.org/pdf/2111.12707) and the authors' [original code](https://github.com/Vegetebird/MHFormer).

It's also worth noting that there are many potential enhancements to be made, such as the usage of different filters (i.e. Extended Kalman Filter), better and faster landmark detectors like OpenPose or MoveNet (or custom ones), and further optimization of the transformer model for inference on low-end devices. And, generally speaking, just usage of Multi-view system :)

| ![example_1](figure/example_1.gif)  | ![example_2](figure/example_2.gif) |
| ------------- | ------------- |
i
## Install using Docker
1. Put pretrained **MHFormer** model to the **./mhformer_checkpoints/81** directory.
Put pretrained **HRNet** and **YOLOv3** models to **./pose_estimation/utils/data** directory.
As a result the structure should look like this:
```bash
.
├── mhformer_checkpoints
│   └── 81
│       └── mhformer_pretrained.pth
├── pose_estimation
│   └── utils
│       └── data
│           ├── yolov3.weights
│           ├── pose_hrnet_w48_384x288.pth
│   ...
```
2. Put in-the-wild video to the **./data/input** directory.
3. Build image
   ```shell
   docker build -t <tag name> .
   ```
4. Run container
   ```shell
   docker run -v $(pwd)/data/input:/app/data/input -v $(pwd)/data/output:/app/data/output -it <tag name> /bin/bash
   ```

## Run 3D skeleton pose estimation and clapping detection
Inside container:
   ```shell
   cd ./scripts && python run_pose_estimation.py --video '../data/input/<video_name>.mp4'
   ```
#### Arguments:
```
- video    : Path to the video.
```
Results will be stored in **./data/output/<video_name>/** in the following format:
```bash
.
├── data
│   ├── output
│       ├── <video_name>
│           ├── 2d+3d.mp4 (predicted 2d and 3d skeletons , clapping counter showed)
│           ├── body_2d_landmarks.json
│           ├── body_3d_landmarks.json
│           ├── 2d+3d
│               ├── 0000.png
│               ├── 00001.png
│               ...
```

## Licence

This project is licensed under the terms of the MIT license.
