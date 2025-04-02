import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from hydra.utils import instantiate
from pathlib import Path
from cameltrack.train.callbacks import PairsStatistics

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)


class CAMELTrackLauncher(ImageLevelModule):
    input_columns = ["bbox_ltwh", "bbox_conf", "keypoints_xyc", "visibility_scores", "embeddings"]
    output_columns = ["track_id"]

    def __init__(
        self,
        CAMELTrack,
        CAMEL,
        device,
        datamodule,
        train_cfg,
        checkpoint_path,
        tracking_dataset,
        override_cfg=None,
        training_enabled: bool = False,
        callbacks=None,
        **kwargs,
    ):
        super().__init__(batch_size=1)
        self.device = device
        if override_cfg is None:
            override_cfg = {}
        self.tracking_dataset = tracking_dataset
        self.CAMEL = instantiate(CAMEL, _recursive_=False)
        self.train_cfg = train_cfg
        self.CAMELTrack = CAMELTrack
        self.override_cfg = override_cfg
        self.datamodule_cfg = datamodule
        self.training_enabled = training_enabled
        self.callbacks = callbacks

        if checkpoint_path:
            self.CAMEL = type(self.CAMEL).load_from_checkpoint(
                checkpoint_path, map_location=self.device, **override_cfg
            )
            log.info(f"Loading CAMEL checkpoint from `{checkpoint_path}`.")

        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.tracker = instantiate(self.CAMELTrack, self.CAMEL.to(self.device))

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        #keep_flags = []
        #for det_id, detection in detections.iterrows():
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
        #keep_flags = np.array(keep_flags)
        #return {"keep_flags": keep_flags}
        return image

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        #keep = batch["keep_flags"][0]
        #if not len(keep):
        #    return []
        tracklab_ids = torch.tensor(detections.index, dtype=torch.int32)  #[keep]
        features = {}
        #image = cv2_load_image(metadatas['file_path'].values[0])
        detections["im_width"] = batch.shape[2]
        detections["im_height"] = batch.shape[1]
        for feature_name in self.input_columns + ["im_width", "im_height"]:
            features[feature_name] = torch.tensor(np.stack(detections[feature_name]),  #[list(keep)],
                                                  dtype=torch.float32).unsqueeze(0)
        image_id = int(metadatas.index[0])
        results = self.tracker.update(features, tracklab_ids, image_id)
        if results:
            results = pd.DataFrame(results)
            results.set_index("tracklab_id", inplace=True, drop=True)
            assert set(results.index).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            return results
        else:
            return []

    def train(self, tracking_dataset, pipeline, evaluator, datasets, *args, **kwargs):
        if self.datamodule_cfg.multi_dataset_training:
            datasets = {dn: dv for dn, dv in datasets.items() if dn in self.datamodule_cfg.multi_dataset_training}
        else:
            datasets = {}
        self.datamodule = instantiate(self.datamodule_cfg, tracking_dataset=tracking_dataset,
                                      pipeline=pipeline, datasets=datasets)
        # self.datamodule = instantiate(self.datamodule_cfg, datasets=datasets, pipeline=pipeline)
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
            #pl.callbacks.EarlyStopping(monitor="val/loss", patience=self.train_cfg.pl_trainer.max_epochs / 5,
            #                           mode="min", check_on_train_epoch_end=False),
        ]
        if self.callbacks is not None:
            callbacks.extend([instantiate(cb, tracking_sets=self.datamodule.tracking_sets) for cb in self.callbacks])
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
            accelerator="cpu",
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
                log.warning(
                    f"No recognized mode selection criteria {self.train_cfg.model_selection_criteria}. Using last checkpoint.")
                checkpoint_path = save_best_loss.last_model_path
            if checkpoint_path:
                log.info(f"Loading CAMEL checkpoint from `{Path(checkpoint_path).resolve()}`.")
                type(self.CAMEL).load_from_checkpoint(checkpoint_path, map_location=self.device)
            else:
                log.warning("No CAMEL checkpoint found.")
        trainer.validate(self.CAMEL, self.datamodule)
