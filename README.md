<div align="center">

# 🐫 CAMELTrack 🐫
## Context-Aware Multi-cue ExpLoitation for Online Multi-Object Tracking

[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-<COLOR>.svg)]() 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sam2mot-a-novel-paradigm-of-multi-object/multi-object-tracking-on-dancetrack)]()
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ettrack-enhanced-temporal-motion-predictor/multi-object-tracking-on-sportsmot)]()
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot17)]()
<!---
Add PoseTrack21 & BEE24
--->

</div>

>**[CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online Multi-Object Tracking]()**
>
>Vladimir Somers, Baptiste Standaert, Victor Joos, Alexandre Alahi, Christophe De Vleeschouwer
>
>[*arxiv 0000.00000*]()

<p align="center">
  <img src="media/dancetrack.gif" width="33%" style="margin:1%;" alt="DanceTrack">
  <img src="media/sportsmot.gif" width="33%" style="margin:1%;" alt="SportsMOT">
</p>
<p align="center">
  <img src="media/mot17.gif" width="33%" style="margin:1%;" alt="MOT17">
  <img src="media/bee24.gif" width="33%" style="margin:1%;" alt="BEE24">
</p>

**CAMELTrack** is an **Online Multi-Object Tracker** that learns to associate detections without hand-crafted heuristics. 
It combines multiple cues through a lightweight, fully trainable module and achieves state-of-the-art performance while 
staying modular and fast.

## 📄 Abstract
**Online Multi-Object Tracking** has been recently dominated by **Tracking-by-Detection** (TbD) methods, where recent advances 
rely on increasingly sophisticated heuristics for tracklet representation, feature fusion, and multi-stage matching. 
The key strength of TbD lies in its modular design, enabling the integration of specialized off-the-shelf models like 
motion predictors and re-identification. However, the extensive usage of human-crafted rules for temporal associations 
makes these methods inherently limited in their ability to capture the complex interplay between various tracking cues. 
In this work, we introduce **CAMEL**, a novel association module for Context-Aware Multi-Cue ExpLoitation, that learns 
resilient association strategies directly from data, breaking free from hand-crafted heuristics while maintaining TbD's 
valuable modularity.

<p align="center">
  <img src="media/pull_figure.jpg" width="50%" alt="Pull Figure of CAMEL">
</p>

At its core, CAMEL employs two transformer-based modules and relies on a novel **Association-Centric 
Training** scheme to effectively model the complex interactions between tracked targets and their various association cues. 
Unlike End-to-End Detection-by-Tracking approaches, our method remains lightweight and fast to train while being able 
to leverage external off-the-shelf models. Our proposed online tracking pipeline, CAMELTrack, achieves state-of-the-art 
performance on multiple tracking benchmarks.

## 🚀 Upcoming

- [x] Cleaning of the code
- [x] Simplified installation and integration into TrackLab
- [x] Public release of the repository
- [ ] Release of the SOTA weights
- [ ] Release of the pickles for the training
- [ ] Cleaning of the code for the training

## ⚙️ Quick Installation Guide
CAMELTrack is built on top of [TrackLab](https://github.com/TrackingLaboratory/tracklab), a research framework for Multi-Object Tracking.

### Clone the repositories
First git clone this repository, and the [TrackLab framework](https://github.com/TrackingLaboratory/tracklab) *in adjacent directories* : 
```bash
mkdir mot
cd mot
git clone XXX
git clone https://github.com/TrackingLaboratory/tracklab.git
```

> [!NOTE]
> If you are using an IDE (like PyCharm or VS Code), we suggest creating a single project with `soccernet` as root directory.
> Instructions : [PyCharm](https://www.jetbrains.com/help/pycharm/configuring-project-structure.html) and [VS Code](https://code.visualstudio.com/docs/editor/multi-root-workspaces)

#### Install using Poetry
1. Install poetry : https://python-poetry.org/docs/#installing-with-the-official-installer
2. Install the dependencies : 
```bash
cd cameltrack
poetry install
poetry run mim install mmcv==2.0.1
poetry shell
```

To enter the virtual environment created by Poetry, you can either use `poetry shell`,
or prefix all commands by `poetry run`.

### Updating
Please make sure to check the official GitHub regularly for updates.
To update this repository to its latest version, run `git pull` on both repositories:
```bash
git pull
git -C ../tracklab pull
```

After updating, you should rerun the installation of the dependencies in case they are updated 
(either running `poetry install` or *both* `pip install`'s).

### Data preparation

Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), 
[DanceTrack](https://drive.google.com/drive/folders/1ASZCFpPEfSOJRktR8qQ_ZoT9nZR0hOea), 
[BEE24](https://holmescao.github.io/datasets/BEE24), [PoseTrack21](https://github.com/anDoer/PoseTrack21) and put them under cameltrack/data.

### Off-the-shelf Model weights
The model weights for the detectors, reID models, and pose estimation models are available here : XXX. The input cues
from each model (if you don't want to run the off-the-shelf models) are available here : XXX.

Alternatively, you can use the detections from [DiffMOT](https://github.com/Kroery/DiffMOT) directly, by placing them in the appropriate directories.

### CAMELTrack model weights
The weights that have been used for the paper can be found here: XXX. They are downloaded
automatically when running with the default config.

| Dataset     |     Appearance     |      Keypoints      |  HOTA  | Weights                                 |
|:------------|:------------------:|:-------------------:|:------:|:----------------------------------------|
| DanceTrack  | :white_check_mark: | :white_check_mark:  |  69.3  | [camel_bbox_app_kps_dancetrack.ckpt]()  |
| DanceTrack  | :white_check_mark: |                     |  66.1  | [camel_bbox_app_dancetrack.ckpt]()      |
| SportsMOT   | :white_check_mark: | :white_check_mark:  |  80.3  | [camel_bbox_app_kps_sportsmot.ckpt]()   |
| MOT17       | :white_check_mark: | :white_check_mark:  |  62.4  | [camel_bbox_app_kps_mot17.ckpt]()       |
| PoseTrack21 | :white_check_mark: | :white_check_mark:  |  66.0  | [camel_bbox_app_kps_posetrack24.ckpt]() |
| BEE24       |                    |                     |  50.3  | [camel_bbox_bee24.ckpt]()               |


## 🎯 Tracking

Run the following command to track, for example, on DanceTrack, with the checkpoint obtained from training, or the provided
model weights :

```
tracklab -cn cameltrack dataset=dancetrack dataset.eval_set=test modules.track=camel_dancetrack modules.track.checkpoint_path=/absolute/path/to/cameltrack_checkpoint.ckpt
```

By default this will create a new directory inside `outputs/cameltrack` which will contain a visualization of the
output for each sequence, in addition to the tracking output in MOT format.

## 💪 Training

### Training on a default dataset
Run the following command to train on a specific dataset (for example, DanceTrack) : 
```
tracklab -cn cameltrack_train dataset=dancetrack modules.track=camel_dancetrack
```
> [!NOTE]
> You can always modify the configuration in [cameltrack.yaml](cameltrack/configs/cameltrack.yaml), and in the
> other files inside this directory, instead of passing these values in the command line.

By default this will create a new directory inside `outputs/cameltrack_train`, which will contain the checkpoints
to the created models, which can then be used for tracking and evaluation.

### Training on a custom dataset
To train on a custom dataset, you'll have to integrate it in tracklab, either by using the MOT format, or by implementing
a new dataset class. Once that's done, you can modify [cameltrack.yaml](cameltrack/configs/cameltrack.yaml), to point to
the new dataset.

## 🖋 Citation

If you use this repository for your research or wish to refer to our contributions, please use the following BibTeX entries:

[CAMELTrack]():
```
@inproceedings{,
        title = ,
        author = ,
        booktitle = ,
}
```

[TrackLab](https://github.com/TrackingLaboratory/tracklab):
```
@misc{Joos2024Tracklab,
	title = {{TrackLab}},
	author = {Joos, Victor and Somers, Vladimir and Standaert, Baptiste},
	journal = {GitHub repository},
	year = {2024},
	howpublished = {\url{https://github.com/TrackingLaboratory/tracklab}}
}
```