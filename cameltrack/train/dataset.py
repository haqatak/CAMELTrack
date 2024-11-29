import cmd
import collections.abc
import io
import json
import logging
import multiprocessing
import pickle
import shutil
import zipfile
from collections import defaultdict, Counter
from contextlib import redirect_stdout
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, List, Optional, NamedTuple, Dict
from filelock import FileLock

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

from cameltrack.train.sampler import samplers, ConcatSampler
from cameltrack.train.transforms import Transform, NoOp, OfflineTransforms
from tracklab.datastruct import TrackingDataset, TrackingSet, TrackerState
from tracklab.pipeline import Pipeline

log = logging.getLogger(__name__)

from functools import lru_cache


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")


class SimFormerDataset(Dataset):
    feature_names = ["age",
                     "bbox_ltwh",
                     "bbox_conf",
                     "keypoints_xyc",
                     "visibility_scores",
                     "embeddings",
                     "image_id",
                     "iou_occlusion",
                     "im_width",
                     "im_height",
                     "drop_app",
                     "drop_bbox",
                     "drop_kps",]

    def __init__(
        self,
        gallery_path,
        config_file,
        video_ids: List[str],
        tracklet_transforms: Optional[Transform] = None,
        max_length: int = 50,
    ):
        self.gallery_path = Path(gallery_path)
        self.config_file = Path(config_file)
        assert self.gallery_path.exists(), f"Gallery path {self.gallery_path} does not exist"
        assert self.config_file.exists(), f"Config file {self.config_file} does not exist"
        self.tracklet_transforms = tracklet_transforms or NoOp()
        self.max_length = max_length
        self._zf = None
        log.debug(f"gallery_path {self.gallery_path} cf {self.config_file}")
        with self.config_file.open() as fp:
            self.samples = json.load(fp)
            self.samples = [s for s in self.samples if s["video_id"] in video_ids]

        self.count_ids = 0.0
        self.global_track_id = 0
        self._feature_columns = None

    def __len__(self):
        return len(self.samples)

    @property
    def zf(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.gallery_path, mode="r")
        return self._zf

    @lru_cache(maxsize=1)
    def _load_pickle(self, sample_video_id):
        with self.zf.open(sample_video_id, "r") as fp:
            df = pickle.load(fp)
        return df

    def print_info(self, training=True):
        videos = Counter(s["video_id"] for s in self.samples)
        video_lengths = defaultdict(int)
        for sample in self.samples:
            video_lengths[sample["video_id"]] = max(video_lengths[sample["video_id"]], len(sample["image_id"]))
        with redirect_stdout(io.StringIO()) as stream:
            cli = cmd.Cmd()
            cli.columnize([f"{name}: {num_tracks} tracks,{video_lengths[name]} images" for name, num_tracks in videos.items()],
                          displaywidth=shutil.get_terminal_size().columns)
        if training:
            log.info(f"Training on {self.gallery_path.name} with the following videos : \n{stream.getvalue()}")
        else:
            log.info(
                f"Validating on {self.gallery_path.name} with the following videos : \n{stream.getvalue()}")


    def feature_columns(self, df: Optional[pd.DataFrame]):
        if self._feature_columns is not None:
            return self._feature_columns
        assert df is not None, "df was None, but the feature sizes haven't been computed yet."
        feature_columns = []
        for name in self.feature_names:
            if name not in df.columns:
                feature_columns.append((name, (1,)))
                continue
            shape = df.iloc[0][name].shape
            shape = (1,) if shape == () else shape
            feature_columns.append((name, shape))
        self._feature_columns = feature_columns
        return self._feature_columns

    def __getitem__(self, idxs):
        true_idxs = [idx for idx in idxs if idx[1] != -1]
        if len(true_idxs) == 0:
            return [self.create_empty_input() for _ in idxs]
        video_img_ids = set([(vid_id, img_id) for vid_id,_,img_id in true_idxs])
        # image_ids = np.unique([img_id for _,_,img_id in true_idxs])
        # idx, image_id = idx
        # if idx == -1:
        #     return self.create_empty_input()
        # else:
        #     sample = self.samples[idx]
        video_dfs = []
        detections = defaultdict(list)
        global_track_ids = {}
        track_ids_dict = {}
        all_global_track_ids = []
        for video_id, idx, img_id in true_idxs:
            detections[(video_id, img_id)].extend(self.samples[idx]["detections"])
            global_track_ids[(video_id, self.samples[idx]["track_id"], img_id)] = self.samples[idx]["global_track_id"]
            all_global_track_ids.append(self.samples[idx]["global_track_id"])
            track_ids_dict[(video_id, idx, img_id)] = self.samples[idx]["track_id"]

        assert len(all_global_track_ids) == len(set(all_global_track_ids)), "Multiple tracklets found with same track_id"
        dets = []
        tracks = []
        track_ids = []
        for video_id, img_id in video_img_ids:
            video_df = self._load_pickle(video_id)
            if self._feature_columns is None:
                self.feature_columns(video_df)
            video_df = video_df.loc[detections[(video_id, img_id)]]
            video_df = video_df[video_df["image_id"] <= img_id].copy()
            video_df.loc[:, "age"] = img_id - video_df["image_id"]
            vid_track_ids = video_df.track_id.unique()
            for track_id in vid_track_ids:
                video_df.track_id[video_df.track_id == track_id] = global_track_ids[(video_id, track_id, img_id)]
            dets.append(video_df[video_df.image_id == img_id])
            tracks.append(video_df[video_df.image_id != img_id])
            # assert video_df.image_id.is_monotonic_increasing
            # dets.append(video_df.tail(1))
            # tracks.append(video_df.head(-1))

        dets = pd.concat(dets, axis=0)
        tracks = pd.concat(tracks, axis=0)
        tracks = tracks.sort_values(by="age", ascending=True)

        tracks["drop_app"], dets["drop_app"],  = False, False
        tracks["drop_bbox"], dets["drop_bbox"] = False, False
        tracks["drop_kps"], dets["drop_kps"] = False, False
        tracks, dets = self.tracklet_transforms(tracks=tracks, dets=dets)
        output_dicts = []
        for video_id, idx, image_id in idxs:
            if idx == -1:
                output_dicts.append(self.create_empty_input())
                continue
            track_id = track_ids_dict[(video_id, idx, image_id)]
            global_track_id = global_track_ids[(video_id, track_id, image_id)]
            track_det = dets[dets.track_id == global_track_id]
            track_track = tracks[tracks.track_id == global_track_id]
            track_track = track_track.sort_values(by="age", ascending=True)
            track_track = track_track.head(self.max_length)
            det_features, det_targets = self.features_targets(track_det)
            track_features, track_targets = self.features_targets(track_track)
            video_id = video_id.split("_")[-1].split(".")[0]
            video_id = np.array(int(video_id)).reshape(1)
            image_id = np.array(image_id).reshape(1)
            output_dicts.append({
                "track_feats": track_features,
                "track_targets": track_targets,
                "det_feats": det_features,
                "det_targets": det_targets,
                "video_id": video_id,
                "image_id": image_id,
            })

        return output_dicts

    def features_targets(self, df):
        features = {}
        for feature_name, feature_dim in self.feature_columns(df):
            if len(df) > 0:
                feature = np.stack(df[feature_name]).astype(np.float32)
                if feature_name in ["keypoints_xyc", "embeddings"]:  # TODO nicer solution?
                    features[feature_name] = feature
                else:
                    features[feature_name] = feature.reshape(len(df), -1)
            else:
                dim = (0, *feature_dim)
                features[feature_name] = np.empty(dim, dtype=np.float32)
        features["index"] = df.index.to_numpy()
        targets = np.array(df["track_id"].to_numpy().astype(np.float32))
        return features, targets

    def create_empty_input(self):
        targets = np.empty((0,), dtype=np.float32)

        return {
            "track_feats": empty_features(self.feature_columns(None)),
            "track_targets": targets,
            "det_feats": empty_features(self.feature_columns(None)),
            "det_targets": targets.copy(),
            "video_id": np.empty((0,), dtype=int),
            "image_id": np.empty((0,), dtype=int),
        }


def empty_features(feature_columns):
    features = {}
    for feature_name, feature_dim in feature_columns:
        dim = (0, *feature_dim)
        features[feature_name] = np.empty(dim, dtype=np.float32)
    features["index"] = np.empty((0,), dtype=np.int64)
    return features


def pad_dict(batch):
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        new_batch = batch
        for key in elem:
            batch_part = pad_dict([d[key] for d in batch])
            for i, _ in enumerate(batch_part):
                new_batch[i][key] = batch_part[i]
        return new_batch
    else:
        max_len = max(len(s) for s in batch)
        new_batch = []
        for i, sample in enumerate(batch):
            length = len(sample)
            if length == max_len:
                new_batch.append(sample)
            else:
                pad_values = ((0, max_len - length),) + ((0, 0),) * (sample.ndim - 1)
                constant_values = -1 if sample.dtype == int else np.nan
                padded_sample = np.pad(sample, pad_width=pad_values, constant_values=constant_values)
                new_batch.append(padded_sample)
        return new_batch


def collate_fn(batch, batch_size=np.nan):
    """Turn pandas dataframe into dict of variable size"""
    batch = list(chain.from_iterable(batch))
    pad_dict(batch)
    batch = default_collate(batch)
    for k, d in batch.items():
        if isinstance(d, collections.abc.Mapping):
            for kk, dd in d.items():
                num_samples = dd.shape[0] // batch_size
                batch[k][kk] = dd.reshape(batch_size, num_samples, *dd.shape[1:])
        else:
            num_samples = d.shape[0] // batch_size
            batch[k] = d.reshape(batch_size, num_samples, *d.shape[1:])
    return batch


class SimFormerDataModule(pl.LightningDataModule):
    """DataModule to create and manage tracking-specific datasets.

    Args:
        detections: ...
        metadatas: ...
        dataset: "train", "val" or "test"
        path: where to save the data to
        samples_per_video: number of samples that will be created per video
        std_age: standard deviation of distribution of the ages of a generated tracklet
        max_length: maximum length of a generated tracklet
        dataset_transforms: list of functions that will transform the given tracklets and det.
    """

    def __init__(
            self,
            tracking_dataset: TrackingDataset,
            dataset_splits: List[str],
            path: Optional[str] = None,
            name: Optional[str] = None,
            num_videos: Optional[int] = None,
            samples_per_video: int = 10,
            std_age: float = 3,
            max_length: int = 50,
            dataset_transforms: List[Callable] = ("normalize2image",),
            tracklet_transforms: Optional[Dict[str, Transform]] = None,
            batch_transforms: Optional[Dict[str, Transform]] = None,
            pipeline: Pipeline = None,
            tracker_states: Dict[str, Path] = None,
            batch_size: int = 128,
            num_samples: int = 8,
            num_workers: int = 8,
            sampler: str = "simple",  # "simple", "occlusion", "gap"
            sampler_args: dict = None,
            train_add_val: bool = False,
            multi_dataset_training: Optional[List[str]] = None,
            datasets = None,
            dataset_paths: Optional[Any] = None,
    ):
        super().__init__()
        self.cfg_detections = dict(
            dataset_transforms=[t for t in dataset_transforms],
            pipeline=str(pipeline),
        )
        if num_videos is None:
            self.cfg = dict(spv=samples_per_video, age=std_age, ml=max_length)
        else:
            self.cfg = dict(
                spv=samples_per_video, age=std_age, ml=max_length, num_videos=num_videos
            )
        self.num_videos = num_videos
        self.samples_per_video = samples_per_video
        self.std_age = std_age
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.train_add_val = train_add_val
        self.sampler = partial(samplers[sampler], **(sampler_args or {}))
        self.val_sampler = samplers["val"]

        self.path = Path(path)
        self.name = name
        self.metadatas = {}
        self.detections = {}
        self.dataset_splits = dataset_splits
        self.detections_paths = {}
        self.dataset_configs = {}
        self.config_locks = {}
        self.tracking_sets = {}
        self.tracker_states = {}
        self.enabled_datasets = multi_dataset_training or []

        for dataset_name in self.enabled_datasets:
            dataset = datasets[dataset_name]
            ds_paths = dataset_paths[dataset_name]
            for ds_split in dataset_splits:
                tracking_set: TrackingSet = dataset.training_sets[ds_split]
                if tracking_set is None and ds_split == "val" and self.train_add_val:
                    raise ValueError(f"{dataset_name} doesn't have a {ds_split} set.")
                elif ds_split == "val" and not self.train_add_val:
                    continue
                self.tracking_sets[(dataset_name, ds_split)] = tracking_set
                self.metadatas[(dataset_name, ds_split)] = tracking_set.image_metadatas
                self.detections[(dataset_name, ds_split)] = tracking_set.detections_gt
                self.detections_paths[(dataset_name, ds_split)] = Path(ds_paths.path) / f"{ds_paths.name}_{ds_split}.pklz"
                self.dataset_configs[(dataset_name, ds_split)] = Path(ds_paths.path) / f"{ds_paths.name}_{ds_split}_tracklets.json"
                self.config_locks[(dataset_name, ds_split)] = FileLock(
                    Path(ds_paths.path) / f"{self.name}_{ds_split}_tracklets.json.lock"
                )
                self.tracker_states[(dataset_name, ds_split)] = ds_paths.tracker_states[ds_split]

        # cfg_name = "_".join([f"{k}{v}" for k, v in self.cfg.items()])
        if path is not None:
            for ds_split in dataset_splits:
                tracking_set: TrackingSet = tracking_dataset.training_sets[ds_split]
                if tracking_set is None:
                    raise ValueError(f"This dataset doesn't have a {ds_split}.")
                self.tracking_sets[ds_split] = tracking_set
                self.metadatas[ds_split] = tracking_set.image_metadatas
                self.detections[ds_split] = tracking_set.detections_gt
                self.detections_paths[ds_split] = self.path / f"{self.name}_{ds_split}.pklz"
                self.dataset_configs[ds_split] = self.path / f"{self.name}_{ds_split}_tracklets.json"
                self.config_locks[ds_split] = FileLock(
                    self.path / f"{self.name}_{ds_split}_tracklets.json.lock"
                )
                self.tracker_states[ds_split] = tracker_states[ds_split]
        self.dataset_transforms = OfflineTransforms.get_transforms(dataset_transforms) if dataset_transforms else []
        self.tracklet_transforms = tracklet_transforms
        self.batch_transforms = batch_transforms
        self.datasets = {}
        self.pipeline = pipeline

    def prepare_data(self) -> None:
        """Generate data on one single node if it doesn't exist yet.

        You shouldn't add things to self in this function as it might be called
        in only one process (for example during multi-gpu training)
        """
        for ds_split in self.dataset_splits:
            log.info(f"Loading detections from {self.detections_paths[ds_split]}")
            if not self.detections_paths[ds_split].exists():
                self.generate_detections(self.detections, self.metadatas, ds_split)

            # Check if the detections have been generated with the same parameters as before
            with zipfile.ZipFile(self.detections_paths[ds_split], mode="r") as zf:
                with zf.open("configuration.json") as fp:
                    cfg = json.load(fp)
                    if cfg != self.cfg_detections:
                        from deepdiff import DeepDiff

                        # FIXME: this is not working
                        # raise ValueError(
                        #     "An existing dataset "
                        #     "with different parameters exists "
                        #     f"{DeepDiff(cfg, self.cfg)}"
                        # )
                        log.warning(f"An existing dataset with different parameters exists {DeepDiff(cfg, self.cfg)}")

            if not self.dataset_configs[ds_split].exists():
                with self.config_locks[ds_split]:
                    self.generate_dataset(ds_split)

    def setup(self, stage: str) -> None:
        ds_names = [(dn, ds) for dn in self.enabled_datasets for ds in self.dataset_splits]
        ds_names += [ds for ds in self.dataset_splits]
        for set_name in ds_names:
            if set_name not in self.detections:
                continue
            if isinstance(set_name, tuple):
                dataset_name, dataset_split = set_name
            else:
                dataset_name, dataset_split = None, set_name
            kwargs = {}
            if self.tracklet_transforms is not None and dataset_split in self.tracklet_transforms:
                kwargs = dict(tracklet_transforms=self.tracklet_transforms[dataset_split])
            kwargs = {**kwargs, "max_length": self.max_length}
            video_ids = [f"sample_{video_id}.pkl" for video_id in self.detections[set_name].video_id.unique()]
            self.datasets[set_name] = SimFormerDataset(
                self.detections_paths[set_name],
                self.dataset_configs[set_name],
                video_ids=video_ids,
                **kwargs
            )

    def train_dataloader(self):
        enabled_sets = ["train", "val"] if self.train_add_val else ["train"]
        if self.train_add_val:
            datasets = [self.datasets["train"], self.datasets["val"]]
        else:
            datasets = [self.datasets["train"]]
        if self.enabled_datasets is not None:
            datasets = [self.datasets[(dn, ds)] for dn in self.enabled_datasets for ds in enabled_sets]
        if self.path is not None:
            datasets += [self.datasets[ds] for ds in enabled_sets]
        samplers = []
        for dataset in datasets:
            dataset.print_info(training=True)
            sampler = self.sampler(dataset=dataset,
                                   batch_size=1, # self.batch_size,
                                   num_samples=self.num_samples,
                                   samples_per_video=self.samples_per_video)
            samplers.append(sampler)
        sampler = ConcatSampler(samplers, batch_size=self.batch_size, num_samples=self.num_samples)
        return DataLoader(
            ConcatDataset(datasets),
            num_workers=self.num_workers,
            collate_fn=partial(collate_fn, batch_size=self.batch_size),
            batch_sampler=sampler,
            worker_init_fn=set_worker_sharing_strategy,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = self.val_sampler(dataset=self.datasets["val"],
                               batch_size=self.batch_size,
                               num_samples=self.num_samples,
                               samples_per_video=self.samples_per_video)

        self.datasets["val"].print_info(training=False)
        return DataLoader(
            self.datasets["val"],
            collate_fn=partial(collate_fn, batch_size=self.batch_size),
            num_workers=self.num_workers,
            batch_sampler=sampler,
            worker_init_fn=set_worker_sharing_strategy,
            pin_memory=True,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        split = "val"
        if self.trainer.validating:
            split = "val"
        elif self.trainer.testing:
            split = "test"
        elif self.trainer.training:
            split = "train"
        if split in self.batch_transforms:
            batch = self.batch_transforms[split](batch)
        return batch

    def generate_detections(self, detections, metadatas, ds_split):
        detections = detections[ds_split]
        metadatas = metadatas[ds_split]
        detections_path = self.detections_paths[ds_split]
        save_zf = zipfile.ZipFile(
            detections_path,
            mode="x",
            allowZip64=True,
            compression=zipfile.ZIP_DEFLATED,
        )
        with save_zf.open("configuration.json", "w") as fp:
            cfg_bytes = json.dumps(self.cfg_detections).encode("utf-8")
            fp.write(cfg_bytes)
        save_zf.close()

        for video_id, video_detections in tqdm(
                detections.groupby("video_id"),
                desc=f"Generate detections for {ds_split} set",
                unit="video",
        ):
            save_zf = zipfile.ZipFile(
                detections_path,
                mode="a",
                allowZip64=True,
            )
            tracker_state = TrackerState(
                self.tracking_sets[ds_split], pipeline=Pipeline([]), load_file=self.tracker_states[ds_split]
            )
            with tracker_state(video_id) as ts:
                detections = video_detections
                for data_trans in self.dataset_transforms:
                    video_detections, video_image_preds = ts.load()
                    detections = data_trans(detections, metadatas, preds=video_detections, tracker_state=ts,
                                            pipeline=self.pipeline)
            with save_zf.open(f"sample_{video_id}.pkl", "w", force_zip64=True) as fp:
                if "body_masks" in detections:  # FIXME this shouldn't be done like this
                    detections = detections.drop(columns=["body_masks"])
                pickle.dump(detections, fp, protocol=pickle.DEFAULT_PROTOCOL)

            save_zf.close()

    def generate_dataset(self, ds_split):
        metadatas = self.metadatas[ds_split]
        detections_path = self.detections_paths[ds_split]
        detections_zf = zipfile.ZipFile(detections_path, mode="a")
        params = []
        video_list = [vid for vid in detections_zf.namelist() if vid.endswith(".pkl")]
        video_list = islice(video_list, self.num_videos) if self.num_videos else video_list
        for video_id in video_list:
            if video_id.endswith(".pkl"):
                params.append(
                    CreateTrackletParams(
                        detections_path=detections_path,
                        video_id=video_id,
                        metadatas=metadatas,
                    )
                )
        with multiprocessing.Pool(max(self.num_workers, 1)) as pool:
            samples = list(
                chain.from_iterable(
                    tqdm(
                        pool.imap(create_tracklets_from_video, params, chunksize=1),
                        desc=f"Building track association {ds_split} set",
                        unit="video",
                        total=len(params),
                    )
                )
            )

        with self.dataset_configs[ds_split].open("w") as fp:
            json.dump(samples, fp)

    # def __len__(self):
    #     return len(self.df_samples)
    #
    # def __getitem__(self, idx):
    #     df_sample = self.df_samples.loc[self.df_samples.sample_id == idx]
    #     df_sample = self.transforms
    #     for transform in self.transforms:
    #         df_sample = transform(df_sample, self.metadatas)
    #     return df_sample


def create_tracklets_from_video(params):
    detections_path, video_id, metadatas = params
    video_id_num = int(video_id.split("_")[-1].split(".")[0])  # remove string stuff
    samples = []
    with zipfile.ZipFile(detections_path, mode="r") as detections_zf:
        with detections_zf.open(video_id) as pkl:
            video_detections = pickle.load(pkl)

        for track_id, track_detections in video_detections.groupby("track_id"):
            if track_id >= 0:
                samples.append(
                    {
                        "video_id": str(video_id),
                        "track_id": int(track_id),
                        "global_track_id": int(video_id_num * 1000) + int(track_id),
                        "image_id": [int(x) for x in track_detections["image_id"]],
                        "detections": [x for x in track_detections.index],
                    }
                )

    return samples


class CreateTrackletParams(NamedTuple):
    detections_path: Path
    video_id: str
    metadatas: Any


class CreateSamplesParams(NamedTuple):
    detections_path: Path
    video_id: str
    samples_per_video: int
    std_age: float
    max_length: int
    metadatas: Any


def create_samples_from_video(params):
    detections_path, video_id, samples_per_video, std_age, max_length, metadatas = params
    samples = []
    with zipfile.ZipFile(detections_path, mode="r") as detections_zf:
        rng = np.random.default_rng()
        with detections_zf.open(video_id) as pkl:
            video_detections = pickle.load(pkl)

        image_ids = sorted(video_detections.image_id.unique())
        samples_per_video = len(
            image_ids[2:]) if samples_per_video == -1 else samples_per_video
        hard_chosen_image_ids = []
        if "id_switch" in video_detections.columns:
            switches_per_image = video_detections.groupby("image_id").id_switch.sum()[2:]
            if switches_per_image.sum() > 0:
                prob_per_image = switches_per_image.astype(np.float64) / switches_per_image.astype(np.float64).sum()
                hard_chosen_image_ids = sorted(
                    rng.choice(image_ids[2:],
                               size=min(samples_per_video, len(prob_per_image.to_numpy().nonzero()[0])),
                               p=prob_per_image,
                               replace=False)
                )
        image_ids_left = list(set(image_ids[2:]) - set(hard_chosen_image_ids))
        chosen_image_ids = sorted(
            list(rng.choice(image_ids_left,
                            size=min(samples_per_video - len(hard_chosen_image_ids), len(image_ids_left)),
                            replace=False
                            )
                 ) + hard_chosen_image_ids
        )
        for img_id in chosen_image_ids:
            sample_idx = img_id * 1000
            idx = image_ids.index(img_id)
            detections = video_detections[video_detections.image_id == img_id].copy()
            if len(detections) == 0:
                continue
            detections["detection"] = True
            detections["sample_id"] = sample_idx
            for track_id, track_detections in video_detections.groupby("track_id"):
                if track_id >= 0:
                    # Only create a tracklet when it is valid
                    tracklet_age = 0
                    tracklet_length = max_length
                    tracklet_image_ids = image_ids[max(0, idx - tracklet_length):idx]
                    tracklet = track_detections.loc[track_detections.image_id.isin(tracklet_image_ids)]
                    tracklet = tracklet.copy()
                    tracklet["sample_id"] = sample_idx
                    tracklet["detection"] = False
                else:
                    tracklet = pd.DataFrame(columns=detections.columns)
                detection = detections[detections.track_id == track_id]
                assert len(detection) <= 1
                if len(tracklet) == 0 or len(detection) == 0:
                    continue
                if len(detection) > 0 and hasattr(detection, "id_switch"):
                    id_switch = int(detection.id_switch.item())
                else:
                    id_switch = 0
                samples.append(
                    {
                        "video_id": str(video_id),
                        "image_id": int(img_id),
                        "file_path": metadatas.loc[img_id].file_path,
                        "detections": [x for x in tracklet.index] + [x for x in detection.index],
                        "id_switch": id_switch,
                    }
                )
    return samples

class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        idxs = [index[1:] for index in idx]
        dataset_idx = idx[0][0]
        assert all(di[0]==dataset_idx for di in idx), "tracklets should all come from the same dataset"
        return self.datasets[dataset_idx][idxs]