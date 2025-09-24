import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from huggingface_hub import hf_hub_download
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
    """
    CAMELTrack class for managing object tracking using CAMEL and the tracklet management logic.
    Also used for training the CAMEL model using pytorch-lightning.
    """
    collate_fn = collate_fn
    input_columns = ["bbox_conf"]  # MODIFIED AT RUNTIME !
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
        """
        Args:
            CAMEL (DictConfig): The CAMEL model configuration.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
            min_det_conf (float): Minimum detection confidence threshold.
            min_init_det_conf (float): Minimum confidence to initialize a new track.
            min_num_hits (int): Minimum number of hits to activate a track.
            max_wo_hits (int): Maximum number of frames without hits before a track is terminated.
            max_track_gallery_size (int): Maximum size of the track gallery.
            override_camel_cfg (DictConfig, optional): Configuration to override CAMEL settings.
            checkpoint_path (str, optional): Path to the model checkpoint.
            training_enabled (bool): Whether training is enabled.
            train_cfg (DictConfig, optional): Training configuration.
            datamodule_cfg (DictConfig, optional): Data module configuration.
            tracking_dataset (TrackingDataset, optional): Dataset for tracking.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(batch_size=1)

        # Instantiate the CAMEL model and move it to the specified device
        self.CAMEL = instantiate(CAMEL, _recursive_=False)
        self.device = device

        # Tracking parameters
        self.min_det_conf = min_det_conf
        self.min_init_det_conf = min_init_det_conf
        self.min_num_hits = min_num_hits
        self.max_wo_hits = max_wo_hits
        self.max_track_gallery_size = max_track_gallery_size

        # Override CAMEL configuration if provided
        self.override_camel_cfg = override_camel_cfg
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_file():
                # Download the checkpoint if it does not exist locally
                hf_hub_download(
                    repo_id="trackinglaboratory/CAMELTrack",
                    filename=checkpoint_path.name,
                    local_dir=checkpoint_path.parent
                )
            # Load the CAMEL model from the checkpoint
            self.CAMEL = type(self.CAMEL).load_from_checkpoint(checkpoint_path, map_location="cpu", **override_camel_cfg)
            log.info(f"Loading CAMEL checkpoint from `{Path(checkpoint_path).resolve()}`.")

        # Training and dataset configurations
        self.training_enabled = training_enabled
        self.train_cfg = train_cfg
        self.datamodule_cfg = datamodule_cfg
        self.tracking_dataset = tracking_dataset

        # Generate input columns specific to the model
        for temporal_encoder in self.CAMEL.temp_encs.values():
            self.input_columns += temporal_encoder.input_columns

        # Reset the tracker state
        self.reset()

    def reset(self):
        """
        Reset the tracker state to start tracking in a new video.
        """
        self.CAMEL.eval()
        self.tracklets = []
        self.frame_count = 0

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        """
        Structures the inputs and detections for further steps of object tracking.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            detections (pd.DataFrame): A DataFrame containing detection information,
                where each row corresponds to a detected object.
            metadata (pd.Series): Metadata associated with the image, including its ID.

        Returns:
            list[Detection]: A list of `Detection` objects, each representing a processed detection
            that meets the minimum detection confidence threshold.
        """
        tracklab_ids = torch.tensor(detections.index)
        image_id = torch.tensor(metadata.id)
        detections["im_width"] = torch.tensor(image.shape[1])
        detections["im_height"] = torch.tensor(image.shape[0])

        # Create a dictionary of features for each detection
        features = {
            feature_name: torch.tensor(np.stack(detections[feature_name]), dtype=torch.float32).unsqueeze(0)
            for feature_name in self.input_columns + ["im_width", "im_height"]
            if len(detections[feature_name]) > 0
        }
        # Generate a list of Detection objects for detections above the confidence threshold
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
                        "index": trk.last_detection.index.item(),
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
            results = pd.DataFrame(actives).set_index("index", drop=True)
            assert set(results.index).issubset(detections_df.index), "Mismatch of indexes during the tracking. The results should match the detections_df."
            return results
        else:
            return []

    @torch.no_grad()
    def associate_dets_to_trks(self, tracklets, detections):
        """
        Associates detections to tracklets using the CAMEL model.

        Args:
            tracklets (list[Tracklet]): A list of existing tracklets to associate detections with.
            detections (list[Detection]): A list of new detections to be associated with tracklets.

        Returns:
            tuple: A tuple containing:
                - matched (np.ndarray): An array of matched tracklet and detection indices.
                - unmatched_trks (list[int]): A list of indices of unmatched tracklets.
                - unmatched_dets (list[int]): A list of indices of unmatched detections.
                - td_sim_matrix (np.ndarray): A similarity matrix between tracklets and detections.
        """
        # If there are no tracklets, return empty matches and all detections as unmatched
        if not tracklets:
            return np.empty((0, 2)), [], list(range(len(detections))), np.empty((0,))
        # If there are no detections, return empty matches and all tracklets as unmatched
        if not detections:
            return np.empty((0, 2)), list(range(len(tracklets))), [], np.empty((0,))

        # Build a batch for the CAMEL model using the tracklets and detections
        batch = self.build_camel_batch(tracklets, detections)
        # Perform prediction using the CAMEL model
        self.CAMEL.to(self.device)
        association_matrix, association_result, td_sim_matrix = self.CAMEL.predict_step(batch, self.frame_count)
        self.CAMEL.to("cpu")
        # Extract matched indices, unmatched tracklets, and unmatched detections
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)


    @torch.no_grad()
    def update_state(self, tracklet):
        """
        Updates the state of a given tracklet based on its current state, hit streak, and time without hits.

        Args:
            tracklet (Tracklet): The tracklet whose state needs to be updated.

        Raises:
            ValueError: If the tracklet is in an undefined state.

        Logic:
            - "init": Transitions to "active" if the hit streak meets the minimum required hits,
              otherwise transitions to "dead" if the time without hits exceeds 1 frame, or remains "init".
            - "active": Remains "active" if there are no missed hits, transitions to "lost" if the time
              without hits is less than the maximum allowed, or transitions to "dead" otherwise.
            - "lost": Transitions back to "active" if there are no missed hits, remains "lost" if the time
              without hits is within the allowed range, or transitions to "dead" otherwise.
            - "dead": Remains "dead".
        """
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
    def build_camel_batch(self, tracklets, detections):
        """
        Builds a batch of features for the CAMEL model, combining tracklet and detection data.

        Args:
            tracklets (list[Tracklet]): A list of tracklets, each containing historical detection data.
            detections (list[Detection]): A list of detections for the current frame.

        Returns:
            dict: A dictionary containing the following keys:
                - 'image_id' (torch.Tensor): The ID of the image associated with the detections.
                - 'det_feats' (dict): A dictionary of detection features, where keys are feature names
                  and values are tensors of shape [1, N, 1, X].
                - 'det_masks' (torch.Tensor): A mask tensor of shape [1, N, 1], indicating valid detections.
                - 'track_feats' (dict): A dictionary of tracklet features, where keys are feature names
                  and values are tensors of shape [1, N, T_max, X].
                - 'track_masks' (torch.Tensor): A mask tensor of shape [1, N, T], indicating valid tracklets.
                - 'det_targets' (torch.Tensor, optional): A tensor of detection targets (track_ids), if available.
                - 'track_targets' (torch.Tensor, optional): A tensor of tracklet targets (track_ids), if available.
        """
        T_max = max(len(t.detections) for t in tracklets)
        device = self.device
        detection_features = self.build_detection_features(detections)
        tracklet_features = self.build_tracklet_features(tracklets, T_max)

        batch = {
            'image_id': detections[0].image_id,  # int
            'det_feats': detection_features,  #
            'det_masks': torch.ones((1, len(detections), 1), dtype=torch.bool).to(device),  # [1, N, 1]
            'track_feats': tracklet_features,
            'track_masks': torch.stack([torch.cat([torch.ones(len(t.detections), dtype=torch.bool), torch.zeros(T_max - len(t.detections), dtype=torch.bool)]) for t in tracklets]).unsqueeze(0).to(device),  # [1, N, T]
        }

        # For training purposes, add targets - when available - to supervise the loss
        if hasattr(detections[0], "target"):
            batch["det_targets"] = torch.stack([det.target for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device)
            batch["track_targets"] = torch.stack([t.padded_features("target", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(self.device)

        return batch

    def build_detection_features(self, detections):
        """
        Builds a dictionary of detection features for a batch of detections.

        Args:
            detections (list[Detection]): A list of Detection objects containing features.

        Returns:
            dict: A dictionary where keys are feature names (e.g., "index", "age", "im_width", "im_height")
            and values are tensors of shape [1, N, 1, X], where N is the number of detections and X the feature size.
        """
        features = {}
        for feature in self.input_columns + ["index", "age", "im_width", "im_height"]:
            stacked_feature = torch.stack([det[feature] for det in detections])
            features[feature] = stacked_feature.unsqueeze(1).unsqueeze(0).to(self.device)

        return features

    def build_tracklet_features(self, tracklets, T_max):
        """
        Builds a dictionary of tracklet features for a batch of tracklets.

        Args:
            tracklets (list[Tracklet]): A list of Tracklet objects containing features.
            T_max (int): The maximum number of detections in any tracklet, used for padding.

        Returns:
            dict: A dictionary where keys are feature names (e.g., "index", "age", "im_width", "im_height")
            and values are tensors of shape [1, N, T_max, X], where N is the number of tracklets and X the feature size.
        """
        features = {}
        for feature in self.input_columns + ["index", "age", "im_width", "im_height"]:
            stacked_feature = torch.stack([t.padded_features(feature, T_max) for t in tracklets])
            features[feature] = stacked_feature.unsqueeze(0).to(self.device)

        return features


class Detection:
    """
    Represents a detection, storing information related to the object.

    Attributes:
        features (dict): Dictionary containing detection features.
        index (torch.Tensor): Unique identifier for the detection in the tracking context.
        image_id (int): ID of the image associated with the detection.
        frame_idx (torch.Tensor): Index of the frame where the detection was made.
        similarity_with_tracklet (float or None): Similarity score with an associated tracklet.
        similarities (torch.Tensor or None): Similarities with all tracklets.
        age (torch.Tensor): Age of the detection (number of frames since creation).
    """
    def __init__(self, image_id, features, tracklab_id, frame_idx):
        self.features = features
        for k, v in features.items():
            if len(v.shape) == 0:
                v = v.unsqueeze(0)
            setattr(self, k, v)
        self.index = tracklab_id.unsqueeze(0)
        self.image_id = image_id
        self.frame_idx = torch.tensor(frame_idx)
        self.similarity_with_tracklet = None
        self.similarities = None
        self.age = torch.tensor((0,))

    def __getitem__(self, item):
        return getattr(self, item)

class Tracklet(object):
    """
    Class representing a tracklet, used to track detected objects across video frames.

    Class Attributes:
        count (int): Global counter to assign unique IDs to each tracklet.

    Instance Attributes:
        last_detection (Detection): The most recent detection associated with this tracklet.
        detections (list): List of all detections associated with this tracklet.
        state (str): Current state of the tracklet ("init", "active", "lost", "dead").
        id (int): Unique identifier for the tracklet.
        age (int): Number of frames since the tracklet was created.
        hits (int): Total number of detections associated with the tracklet.
        hit_streak (int): Number of consecutive frames with associated detections.
        time_wo_hits (int): Number of consecutive frames without associated detection.
        max_gallery_size (int): Maximum number of detections to keep in the gallery.
    """
    count = 1  # MOT benchmark requires positive

    def __init__(self, detection, max_gallery_size):
        """
        Initialize a new tracklet.

        Args:
            detection (Detection): The first detection associated with this tracklet.
            max_gallery_size (int): Maximum number of detections to keep in the gallery.
        """
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
        """
        Update the tracklet parameters for a new frame.
        Increments age and time without detection, and updates the age of all associated detections.
        Resets hit_streak if no detection has been associated with more than one frame.
        """
        self.age += 1
        self.time_wo_hits += 1
        # Update the age of all previous detections
        for detection in self.detections:
            detection.age += 1

        if self.time_wo_hits > 1:
            self.hit_streak = 0

    def update(self, detection):
        """
        Update the tracklet with new detection.
        """
        self.detections.append(detection)
        self.detections = self.detections[-self.max_gallery_size:]
        # Variables for tracklet management
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0
        self.last_detection = detection

    def padded_features(self, name, size):
        """
        Get the features of the tracklet padded to the provided size for batching.
        """
        features = torch.stack([getattr(det, name) for det in reversed(self.detections)])
        if features.shape[0] < size:
            features = torch.cat([features, torch.zeros(size - features.shape[0], *features.shape[1:], device=features.device) + float('nan')])
        return features

    def __str__(self):
        """
        Helper function.
        Return a string representation of the tracklet.
        """
        return (f"Tracklet(id={self.id}, state={self.state}, age={self.age}, "
                f"hits={self.hits}, hit_streak={self.hit_streak}, "
                f"time_wo_hits={self.time_wo_hits}, "
                f"num_detections={len(self.detections)})")
