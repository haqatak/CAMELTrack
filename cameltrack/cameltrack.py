import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from hydra.utils import instantiate
from pathlib import Path

from omegaconf import DictConfig

from cameltrack.train.callbacks import PairsStatistics, SimMetrics

from tracklab.pipeline import ImageLevelModule
from tracklab.datastruct import TrackingDataset

log = logging.getLogger(__name__)


class CAMELTrack(ImageLevelModule):
    input_columns = ["bbox_ltwh", "bbox_conf", "keypoints_xyc", "visibility_scores", "embeddings"]
    output_columns = ["track_id"]

    def __init__(
            self,
            CAMEL,
            device,
            min_det_conf: float = 0.4,
            min_init_det_conf: float = 0.6,
            min_num_hits: int = 0,
            max_wo_hits: int = 150,
            max_track_gallery_size: int = 50,
            override_camel_cfg: DictConfig = None,
            checkpoint_path: str = None,
            training_enabled: bool = False,
            train_cfg: DictConfig = None,
            datamodule_cfg: DictConfig = None,
            tracking_dataset: TrackingDataset = None,
            **kwargs,
    ):
        super().__init__(batch_size=1)

        self.CAMEL = instantiate(CAMEL, _recursive_=False).to(device)
        self.device = device

        self.min_det_conf = min_det_conf
        self.min_init_det_conf = min_init_det_conf
        self.min_num_hits = min_num_hits
        self.max_wo_hits = max_wo_hits
        self.max_track_gallery_size = max_track_gallery_size

        self.override_camel_cfg = override_camel_cfg
        if checkpoint_path:
            self.CAMEL = type(self.CAMEL).load_from_checkpoint(checkpoint_path, map_location=self.device, **override_camel_cfg)
            log.info(f"Loading CAMEL checkpoint from `{Path(checkpoint_path).resolve()}`.")

        self.training_enabled = training_enabled
        self.train_cfg = train_cfg
        self.datamodule_cfg = datamodule_cfg
        self.tracking_dataset = tracking_dataset

        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.CAMEL.to(self.device).eval()
        self.tracklets = []
        self.frame_count = 0

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        # keep_flags = []
        # for det_id, detection in detections.iterrows():
        #    if 'keypoints_xyc' in detection and self.vis_keypoint_threshold is not None:
        #        detection["keypoints_xyc"][
        #            detection["keypoints_xyc"][:, 2] < self.vis_keypoint_threshold,
        #            2,
        #        ] = 0
        #        keep = (detection["bbox_conf"] >= self.min_det_conf) and (
        #            (detection["keypoints_xyc"][:, 2] != 0).sum() >= self.min_vis_keypoints)
        #    elif self.min_det_conf is not None:
        #        keep = (detection["bbox_conf"] >= self.min_det_conf)
        #    else:
        #        keep = True
        #    keep_flags.append(keep)
        # keep_flags = np.array(keep_flags)
        # return {"keep_flags": keep_flags}
        return image

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # keep = batch["keep_flags"][0]
        # if not len(keep):
        #    return []
        tracklab_ids = torch.tensor(detections.index, dtype=torch.int32)  # [keep]
        features = {}
        # image = cv2_load_image(metadatas['file_path'].values[0])
        detections["im_width"] = batch.shape[2]
        detections["im_height"] = batch.shape[1]
        for feature_name in self.input_columns + ["im_width", "im_height"]:
            features[feature_name] = torch.tensor(np.stack(detections[feature_name]),  # [list(keep)],
                                                  dtype=torch.float32).unsqueeze(0)
        image_id = int(metadatas.index[0])
        results = self.update(features, tracklab_ids, image_id)
        if results:
            results = pd.DataFrame(results)
            results.set_index("tracklab_id", inplace=True, drop=True)
            assert set(results.index).issubset(detections.index), "Mismatch of indexes during the tracking. The results should match the detections."
            return results
        else:
            return []

    @torch.no_grad()
    def update(self, features, tracklab_ids, image_id):
        """
        Rough overview of the update method:
        1. forward each tracklet, update {age, time_wo_hits, hit_streak}
        2. perform data association between existing tracklets and new detections
        3. update all track with matched det: update detections list, last_detection, hits, hit_streak, time_wo_hits
        4. init track with unmatched dets
        5. update state of all tracklets: init ( hit_streak < min_num_hits) -> active -> lost (time_wo_hits < max_wo_hits) -> dead
        6. update track list by removing dead tracklets
        7. return all active tracks
        """
        self.frame_count += 1  # FIXME should be send in the end

        # build detections from features
        detections = []
        for i in range(len(tracklab_ids)):
            features_i = {k: v[0, i] for k, v in features.items()}
            if features_i["bbox_conf"] >= self.min_det_conf:
                detections.append(
                    Detection(
                        image_id,
                        features_i,
                        tracklab_ids[i],
                        frame_idx=self.frame_count - 1
                    )
                )

        # Update the states of the tracklets
        for track in self.tracklets:
            track.forward()

        # Associate detections to tracklets
        (
            matched,
            unmatched_trks,
            unmatched_dets,
            td_sim_matrix,
        ) = self.associate_dets_to_trks(self.tracklets, detections)

        # Assert each track and det index is present exactly once in matched or unmatched_trks or unmatched_dets
        # Ensure that the track indices are all accounted for, and no duplicates in matched
        all_trk_indices = [m[0] for m in matched.tolist()] + unmatched_trks
        assert len(all_trk_indices) == len(self.tracklets)  # Ensure all indices are present
        assert len(set(all_trk_indices)) == len(all_trk_indices)  # Ensure no duplicates

        # Ensure that the detection indices are all accounted for, and no duplicates in matched
        all_det_indices = [m[1] for m in matched.tolist()] + unmatched_dets
        assert len(all_det_indices) == len(detections)  # Ensure all indices are present
        assert len(set(all_det_indices)) == len(all_det_indices)  # Ensure no duplicates

        # Update matched tracklets with assigned detections
        for m in matched:
            tracklet = self.tracklets[m[0]]
            detection = detections[m[1]]
            tracklet.update(detection)
            similarity = td_sim_matrix[m[0], m[1]]
            detection.similarity_with_tracklet = similarity
            detection.similarities = td_sim_matrix[:len(self.tracklets), m[1]]

        # Create and initialise new tracklets for unmatched detections
        first_tracks = len(self.tracklets) == 0
        for i in unmatched_dets:
            # Check that confidence is high enough
            if detections[i].bbox_conf < self.min_init_det_conf:
                continue
            trk = Tracklet(detections[i], self.max_track_gallery_size, first_tracks=first_tracks)
            self.tracklets.append(trk)

        # Handle tracklets outputs and cleaning
        actives = []
        for trk in self.tracklets:
            # Get active tracklets
            self.update_state(trk)
            if trk.state == "active":
                actives.append(
                    {
                        "tracklab_id": trk.last_detection.tracklab_id.item(),
                        "track_id": trk.id,  # id is computed from a counter
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().numpy()) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in enumerate(trk.last_detection.similarities.cpu().numpy())} if trk.last_detection.similarities is not None else None,
                            "St": self.CAMEL.sim_threshold,
                        }
                    }
                )
            # for first frames, also return tracklets in init state but not confirm them as active
            elif trk.state == "init" and self.frame_count <= self.min_num_hits:
                actives.append(
                    {
                        "tracklab_id": trk.last_detection.tracklab_id.item(),
                        "track_id": trk.id,  # id is computed from a counter
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().numpy()) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in enumerate(trk.last_detection.similarities.cpu().numpy())} if trk.last_detection.similarities is not None else None,
                            "St": self.CAMEL.sim_threshold,
                        }
                    }
                )
        self.tracklets = [trk for trk in self.tracklets if trk.state != "dead"]
        return actives

    @torch.no_grad()
    def associate_dets_to_trks(self, tracklets, detections):
        if len(tracklets) == 0:
            return (
                np.empty((0, 2)),
                np.empty((0,)).tolist(),
                np.arange(len(detections)).tolist(),
                np.empty((0,)),
            )
        if len(detections) == 0:
            return (
                np.empty((0, 2)),
                np.arange(len(self.tracklets)).tolist(),
                np.empty((0,)).tolist(),
                np.empty((0,)),
            )

        batch = self.build_camel_batch(tracklets, detections)
        association_matrix, association_result, td_sim_matrix = self.CAMEL.predict_step(batch, self.frame_count)
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)

    @torch.no_grad()
    def build_camel_batch(self, tracklets, detections):  # TODO UGLY - refactor
        T_max = np.array([len(t.detections) for t in tracklets]).max()
        batch = {
            'image_id': detections[0].image_id,  # int
            'det_feats': {
                'visibility_scores': torch.stack([det.visibility_scores for det in detections]).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 7]
                'embeddings': torch.stack([det.embeddings for det in detections]).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 7*D]
                'index': torch.IntTensor([det.tracklab_id for det in detections]).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1]
                'bbox_conf': torch.stack([det.bbox_conf for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 1]
                'bbox_ltwh': torch.stack([det.bbox_ltwh for det in detections]).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 4]
                'keypoints_xyc': torch.stack([det.keypoints_xyc for det in detections]).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 17, 3]
                'age': torch.zeros(len(detections)).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device),  # [B, N, 1, 1]
                'im_width': torch.stack([det.im_width for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 1]
                'im_height': torch.stack([det.im_height for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device),  # [1, N, 1, 1]
            },
            'det_masks': torch.ones((1, len(detections), 1), dtype=torch.bool).to(self.device),  # [1, N, 1]
            'track_feats': {
                'visibility_scores': torch.stack([t.padded_features("visibility_scores", T_max) for t in tracklets]).unsqueeze(0).to(self.device),# [1, N, T, 7]
                'embeddings': torch.stack([t.padded_features("embeddings", T_max) for t in tracklets]).unsqueeze(0).to(self.device),  # [1, N, T, 7*D]
                'index': torch.stack([t.padded_features("tracklab_id", T_max) for t in tracklets]).unsqueeze(0).to(self.device),  # [1, N, T]
                'bbox_conf': torch.stack([t.padded_features("bbox_conf", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device),  # [1, N, T, 1]
                'bbox_ltwh': torch.stack([t.padded_features("bbox_ltwh", T_max) for t in tracklets]).unsqueeze(0).to(self.device),  # [1, N, T, 4]
                'keypoints_xyc': torch.stack([t.padded_features("keypoints_xyc", T_max) for t in tracklets]).unsqueeze(0).to(self.device),  # [1, N, T, 17, 3]
                'age': self.frame_count - 1 - torch.stack([t.padded_features("frame_idx", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device),  # [1, N, T, 1]
                'im_width': torch.stack([t.padded_features("im_width", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device),  # [1, N, T, 1]
                'im_height': torch.stack([t.padded_features("im_height", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device),  # [1, N, T, 1]
            },
            'track_masks': torch.stack([torch.cat([torch.ones(len(t.detections), dtype=torch.bool), torch.zeros(T_max - len(t.detections), dtype=torch.bool)]) for t in tracklets]).unsqueeze(0).to(self.device),  # [1, N, T]
        }

        if hasattr(detections[0], "target"):
            batch["det_targets"] = torch.stack([det.target for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device)
            batch["track_targets"] = torch.stack([t.padded_features("target", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device)
        return batch

    @torch.no_grad()
    def update_state(self, tracklet):
        # Transition tracklet state based on simple rules
        s = tracklet.state
        if s == "init":
            if tracklet.hit_streak >= self.min_num_hits:
                new_state = "active"
            elif tracklet.time_wo_hits >= 1:
                new_state = "dead"
            else:
                new_state = "init"
        elif s == "active":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits < self.max_wo_hits:
                new_state = "lost"
            else:
                new_state = "dead"
        elif s == "lost":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits < self.max_wo_hits:
                new_state = "lost"
            else:
                new_state = "dead"
        elif s == "dead":
            new_state = "dead"
        else:
            raise ValueError(f"tracklet state is in undefined state {s}.")
        tracklet.state = new_state

    def train(self, tracking_dataset, pipeline, evaluator, datasets, *args, **kwargs):
        if self.datamodule_cfg.multi_dataset_training:
            datasets = {dn: dv for dn, dv in datasets.items() if dn in self.datamodule_cfg.multi_dataset_training}
        else:
            datasets = {}
        self.datamodule = instantiate(self.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=pipeline, datasets=datasets)
        save_best_loss = pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath="CAMEL",
            mode="min",
            filename="epoch={epoch}-loss={val/loss:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        callbacks = [
            save_best_loss,
            pl.callbacks.LearningRateMonitor(),
            PairsStatistics(),
            SimMetrics(),
            # pl.callbacks.EarlyStopping(monitor="val/loss", patience=self.train_cfg.pl_trainer.max_epochs / 5,
            #                           mode="min", check_on_train_epoch_end=False),
        ]
        if self.train_cfg.use_rich:
            callbacks.append(pl.callbacks.RichProgressBar())
        if self.train_cfg.use_wandb:
            logger = pl.loggers.WandbLogger(project="CAMEL", resume=True)
        else:
            logger = pl.loggers.WandbLogger(project="CAMEL", offline=True)
        tr_cfg = self.train_cfg.pl_trainer
        trainer = pl.Trainer(
            max_epochs=tr_cfg.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator=self.device,
            num_sanity_val_steps=tr_cfg.num_sanity_val_steps,
            fast_dev_run=tr_cfg.fast_dev_run,
            precision=tr_cfg.precision,
            gradient_clip_val=tr_cfg.gradient_clip_val,
            accumulate_grad_batches=tr_cfg.accumulate_grad_batches,
            log_every_n_steps=tr_cfg.log_every_n_steps,
            check_val_every_n_epoch=tr_cfg.check_val_every_n_epochs,
            val_check_interval=tr_cfg.val_check_interval,
            enable_progress_bar=tr_cfg.enable_progress_bar,
            profiler=tr_cfg.profiler,
            enable_model_summary=tr_cfg.enable_model_summary,
        )
        if not self.train_cfg.evaluate_only:
            last_path = Path("CAMEL/last.ckpt")
            ckpt_path = last_path if last_path.exists() else None
            trainer.fit(self.CAMEL, self.datamodule, ckpt_path=ckpt_path)
            if self.train_cfg.model_selection_criteria == "best_loss":
                checkpoint_path = save_best_loss.best_model_path if save_best_loss.best_model_path else save_best_loss.last_model_path
            elif self.train_cfg.model_selection_criteria == "last":
                checkpoint_path = save_best_loss.last_model_path
            else:
                log.warning(f"No recognized mode selection criteria {self.train_cfg.model_selection_criteria}. Using last checkpoint.")
                checkpoint_path = save_best_loss.last_model_path
            if checkpoint_path:
                log.info(f"Loading CAMEL checkpoint from `{Path(checkpoint_path).resolve()}`.")
                type(self.CAMEL).load_from_checkpoint(checkpoint_path, map_location=self.device)
            else:
                log.warning("No CAMEL checkpoint found.")
        trainer.validate(self.CAMEL, self.datamodule)

class Detection:
    def __init__(self, image_id, features, tracklab_id, frame_idx):
        for k, v in features.items():
            setattr(self, k, v)
        self.tracklab_id = tracklab_id
        self.image_id = image_id
        self.frame_idx = torch.tensor(frame_idx)
        self.similarity_with_tracklet = None
        self.similarities = None

class Tracklet(object):
    # MOT benchmark requires positive:
    count = 1

    def __init__(self, detection, max_gallery_size, first_tracks=False):
        self.last_detection = detection
        self.detections = [detection]
        self.state = "init" if not first_tracks else "active"
        self.id = Tracklet.count
        Tracklet.count += 1
        # Variables for tracklet management
        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_wo_hits = 0

        self.max_gallery_size = max_gallery_size

    def forward(self):
        self.age += 1
        self.time_wo_hits += 1
        if self.time_wo_hits > 1:
            self.hit_streak = 0

    def update(self, detection):
        self.detections.append(detection)
        self.detections = self.detections[-self.max_gallery_size:]
        # Variables for tracklet management
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0
        self.last_detection = detection

    def padded_features(self, name, size):
        features = torch.stack([getattr(det, name) for det in reversed(self.detections)])
        if features.shape[0] < size:
            features = torch.cat(
                [features, torch.zeros(size - features.shape[0], *features.shape[1:], device=features.device) + float('nan')]
            )
        return features

    def __str__(self):
        return (f"Tracklet(id={self.id}, state={self.state}, age={self.age}, "
                f"hits={self.hits}, hit_streak={self.hit_streak}, "
                f"time_wo_hits={self.time_wo_hits}, "
                f"num_detections={len(self.detections)})")
