import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tracklab.datastruct import TrackingDataset
from tracklab.pipeline import ImageLevelModule

from cameltrack.train.callbacks import SimMetrics

log = logging.getLogger(__name__)

def collate_fn(batch):  # FIXME collate_fn could handle a part of the preprocessing
    """
    :param batch: [(idxs,  [Detection, ...])]
    :return: ([idxs], [Detection, ...])
    """
    idxs, detections = batch[0]
    return ([idxs], detections)

class CAMELTrack(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = [] # MODIFIED AT RUNTIME !
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

        for temporal_encoder in self.CAMEL.temp_encs.values():
            self.input_columns += temporal_encoder.input_columns

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
        tracklab_ids = torch.tensor(detections.index)
        image_id = torch.tensor(metadata.id)
        detections["im_width"] = torch.tensor(image.shape[1])
        detections["im_height"] = torch.tensor(image.shape[0])

        features = {
            feature_name: torch.tensor(np.stack(detections[feature_name]), dtype=torch.float32).unsqueeze(0)
            for feature_name in self.input_columns + ["im_width", "im_height"]
            if len(detections[feature_name]) > 0
        }

        return [
            Detection(image_id, {k: v[0, i] for k, v in features.items()}, tracklab_ids[i], frame_idx=self.frame_count)
            for i in range(len(tracklab_ids)) if features["bbox_conf"][0, i] >= self.min_det_conf
        ]

    @torch.no_grad()
    def process(self, detections, detections_df: pd.DataFrame, metadatas: pd.DataFrame):
        # Update the states of the tracklets
        for track in self.tracklets:
            track.forward()

        # Associate detections to tracklets
        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.associate_dets_to_trks(self.tracklets, detections)

        #  Check that each track and detection index is present exactly once
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        # Update matched tracklets with assigned detections
        for m in matched:
            tracklet = self.tracklets[m[0]]
            detection = detections[m[1]]
            tracklet.update(detection)
            detection.similarity_with_tracklet = td_sim_matrix[m[0], m[1]]
            detection.similarities = td_sim_matrix[:len(self.tracklets), m[1]]

        # Create and initialise new tracklets for unmatched detections
        for i in unmatched_dets:
            # Check that confidence is high enough
            if detections[i].bbox_conf >= self.min_init_det_conf:
                self.tracklets.append(Tracklet(detections[i], self.max_track_gallery_size))

        # Handle tracklets outputs and cleaning
        actives = []
        for trk in self.tracklets:
            # Get active tracklets and for first frames, also return tracklets in init state
            self.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.frame_count < self.min_num_hits):
                actives.append(
                    {
                        "tracklab_id": trk.last_detection.tracklab_id.item(),
                        "track_id": trk.id,  # id is computed from a counter
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().item()) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in enumerate(trk.last_detection.similarities.cpu().numpy())} if trk.last_detection.similarities is not None else None,
                            "St": self.CAMEL.sim_threshold,
                    }
                })

        self.tracklets = [trk for trk in self.tracklets if trk.state != "dead"]
        self.frame_count += 1

        if actives:
            results = pd.DataFrame(actives).set_index("tracklab_id", drop=True)
            assert set(results.index).issubset(detections_df.index), "Mismatch of indexes during the tracking. The results should match the detections_df."
            return results
        else:
            return []

    @torch.no_grad()
    def associate_dets_to_trks(self, tracklets, detections):
        if not tracklets:
            return np.empty((0, 2)), [], list(range(len(detections))), np.empty((0,))
        if not detections:
            return np.empty((0, 2)), list(range(len(tracklets))), [], np.empty((0,))

        batch = self.build_camel_batch(tracklets, detections)
        association_matrix, association_result, td_sim_matrix = self.CAMEL.predict_step(batch, self.frame_count)
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)


    @torch.no_grad()
    def update_state(self, tracklet):
        s = tracklet.state
        if s == "init":
            new_state = "active" if tracklet.hit_streak >= self.min_num_hits else "dead" if tracklet.time_wo_hits >= 1 else "init"
        elif s == "active":
            new_state = "active" if tracklet.time_wo_hits == 0 else "lost" if tracklet.time_wo_hits < self.max_wo_hits else "dead"
        elif s == "lost":
            new_state = "active" if tracklet.time_wo_hits == 0 else "lost" if tracklet.time_wo_hits < self.max_wo_hits else "dead"
        elif s == "dead":
            new_state = "dead"
        else:
            raise ValueError(f"Tracklet {tracklet} is in undefined state {s}.")
        tracklet.state = new_state

    def train(self, tracking_dataset, pipeline, evaluator, datasets, *args, **kwargs):
        datasets = {dn: dv for dn, dv in datasets.items() if dn in self.datamodule_cfg.multi_dataset_training} if self.datamodule_cfg.multi_dataset_training else {}
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
            SimMetrics(),
        ]

        if self.train_cfg.use_rich:
            callbacks.append(pl.callbacks.RichProgressBar())

        logger = pl.loggers.WandbLogger(project="CAMEL", resume=True) if self.train_cfg.use_wandb else pl.loggers.WandbLogger(project="CAMEL", offline=True)

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
            ckpt_path = Path("CAMEL/last.ckpt") if Path("CAMEL/last.ckpt").exists() else None
            trainer.fit(self.CAMEL, self.datamodule, ckpt_path=ckpt_path)

            if self.train_cfg.model_selection_criteria == "best_loss":
                checkpoint_path = save_best_loss.best_model_path or save_best_loss.last_model_path
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

    @torch.no_grad()
    def build_camel_batch(self, tracklets, detections):  # TODO UGLY - refactor
        T_max = max(len(t.detections) for t in tracklets)
        device = self.device

        batch = {
            'image_id': detections[0].image_id,  # int
            'det_feats': {
                'visibility_scores': torch.stack([det.visibility_scores for det in detections]).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 7]
                'embeddings': torch.stack([det.embeddings for det in detections]).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 7*D]
                'index': torch.IntTensor([det.tracklab_id for det in detections]).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1]
                'bbox_conf': torch.stack([det.bbox_conf for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 1]
                'bbox_ltwh': torch.stack([det.bbox_ltwh for det in detections]).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 4]
                'keypoints_xyc': torch.stack([det.keypoints_xyc for det in detections]).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 17, 3]
                'age': torch.zeros(len(detections)).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device),  # [B, N, 1, 1]
                'im_width': torch.stack([det.im_width for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 1]
                'im_height': torch.stack([det.im_height for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device),  # [1, N, 1, 1]
            },
            'det_masks': torch.ones((1, len(detections), 1), dtype=torch.bool).to(device),  # [1, N, 1]
            'track_feats': {
                'visibility_scores': torch.stack([t.padded_features("visibility_scores", T_max) for t in tracklets]).unsqueeze(0).to(device),# [1, N, T, 7]
                'embeddings': torch.stack([t.padded_features("embeddings", T_max) for t in tracklets]).unsqueeze(0).to(device),  # [1, N, T, 7*D]
                'index': torch.stack([t.padded_features("tracklab_id", T_max) for t in tracklets]).unsqueeze(0).to(device),  # [1, N, T]
                'bbox_conf': torch.stack([t.padded_features("bbox_conf", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device),  # [1, N, T, 1]
                'bbox_ltwh': torch.stack([t.padded_features("bbox_ltwh", T_max) for t in tracklets]).unsqueeze(0).to(device),  # [1, N, T, 4]
                'keypoints_xyc': torch.stack([t.padded_features("keypoints_xyc", T_max) for t in tracklets]).unsqueeze(0).to(device),  # [1, N, T, 17, 3]
                'age': self.frame_count - torch.stack([t.padded_features("frame_idx", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device),  # [1, N, T, 1]
                'im_width': torch.stack([t.padded_features("im_width", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device),  # [1, N, T, 1]
                'im_height': torch.stack([t.padded_features("im_height", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device),  # [1, N, T, 1]
            },
            'track_masks': torch.stack([torch.cat([torch.ones(len(t.detections), dtype=torch.bool), torch.zeros(T_max - len(t.detections), dtype=torch.bool)]) for t in tracklets]).unsqueeze(0).to(device),  # [1, N, T]
        }

        if hasattr(detections[0], "target"):
            batch["det_targets"] = torch.stack([det.target for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device)
            batch["track_targets"] = torch.stack([t.padded_features("target", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device)

        return batch


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

    def __init__(self, detection, max_gallery_size):
        self.last_detection = detection
        self.detections = [detection]
        self.state = "init"
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
