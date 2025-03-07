import logging
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_metric_learning import distances, losses, reducers

from cameltrack.utils.merge_token_strats import merge_token_strats
from cameltrack.utils.similarity_metrics import similarity_metrics
from cameltrack.utils import coordinates, assignment_strats

log = logging.getLogger(__name__)


@dataclass
class Tracklets:
    """Contains all information of tracklets.

    Args:
        features: dict of tensors float32 [B, N, T, F]
        feats_masks: tensor bool [B, N, T] of valid features for padding
        targets: optional, tensor float32 [B, N]

    Attributes:
        feats: dict of tensors float32 [B, N, T, F]
        feats_masks: tensor bool [B, N, T]
        masks: tensor bool [B, N]
        tokens: tensor float32 [B, N, E]
        embs: tensor float32 [B, N, E]
        targets: tensor float32 [B, N]
    """
    def __init__(self, features, feats_masks, targets=None):
        self.feats = features
        self.feats_masks = feats_masks
        self.masks = self.feats_masks.any(dim=-1)
        if targets is not None and len(targets.shape) > 2:
            self.targets = targets[:, :, 0]
        else:
            self.targets = targets


@dataclass
class Detections(Tracklets):
    """Contains all information of detections.

    Args:
        features: dict of tensors float32 [B, N, 1, F]
        feats_masks: tensor bool [B, N, 1] of valid features for padding
        targets: optional, tensor float32 [B, N]

    Attributes:
        feats: dict of tensors float32 [B, N, 1, F]
        feats_masks: tensor bool [B, N, 1]
        masks: tensor bool [B, N]
        tokens: tensor float32 [B, N, E]
        embs: tensor float32 [B, N, E]
        targets: tensor float32 [B, N]
    """
    def __init__(self, features, feats_masks, targets=None):
        assert feats_masks.shape[2] == 1
        super().__init__(features, feats_masks, targets)


class CAMEL(pl.LightningModule):
    """ CAMEL Transformer : .

    Args:
      transformer_cfg: dict containing the transformer backbone architecture
      tokenizers_cfg: dict containing the tokenizers for each cue (key: cue name)
      classifier_cfg: DEPRECATED
      train_cfg: dict containing training configuration
      batch_transforms: dict with "train"/"val"/"test" keys with an instantiatable dict.
      merge_token_strat: sum/identity/concat
      sim_strat: default_for_each_token_type/cosine/euclidean/norm_euclidean/iou/random
      assos_strat: hungarian/argmax/greedy
      sim_threshold: Threshold for similarity measure
      use_computed_threshold: if True, use threshold computed during validation
      tl_margin: ???
      loss_strat: triplet/infoNCE
      contrastive_loss_strat: inter/inter_intra/valid_inter/valid_inter_intra
      norm_coords: centered/positive/None
    """
    def __init__(
            self,
            gaffe_cfg: DictConfig,
            temp_enc_cfg: DictConfig,
            train_cfg: DictConfig = None,
            batch_transforms: DictConfig = None,
            merge_token_strat: str = "sum",
            sim_strat: str = "cosine",
            sim_threshold: int = 0.5,  # fixme this should be in CAMELTrack
            use_computed_threshold: bool = True,  # fixme this should be in CAMELTrack
    ):
        super().__init__()
        all_params = locals()  # Get all arguments passed to __init__
        ignore_keys = [key for key in all_params if "checkpoint_path" in key]
        self.save_hyperparameters(ignore=ignore_keys)

        self.gaffe = instantiate(gaffe_cfg)  # TODO : remove dependency to Hydra
        self.tokenizers = nn.ModuleDict({n: instantiate(t, name=n) for n, t in temp_enc_cfg.items() if t['_enabled_']})
        self.train_cfg = train_cfg
        self.merge_token_strat = merge_token_strat
        if sim_strat == "default_for_each_token_type":
            log.warning("sim_strat set to 'default_for_each_token_type', setting merge_token_strat to 'identity'.")
            self.merge_token_strat = "identity"
        self.sim_strat = sim_strat
        self.computed_sim_threshold = None
        self.computed_sim_threshold2 = None
        self.best_distr_overlap_threshold = None
        self.sim_threshold = sim_threshold
        self.final_tracking_threshold = sim_threshold
        self.use_computed_threshold = use_computed_threshold
        log.info(f"CAMEL initialized with final_tracking_threshold={sim_threshold} from yaml config. "
                 f"Will be overwritten later by an optimized threshold if CAMEL validation is enabled.")
        self.norm_coords = coordinates.norm_coords_strats["positive"]

        # handle instantiation functions
        if batch_transforms is not None:
            self.batch_transforms = {n: instantiate(t) for n, t in
                                     batch_transforms.items()}
        else:
            self.batch_transforms = {}

        # merging
        if self.merge_token_strat in merge_token_strats:
            self.merge = merge_token_strats[self.merge_token_strat]
        else:
            raise NotImplementedError

        # similarity
        if sim_strat in similarity_metrics:
            self.similarity_metric = similarity_metrics[sim_strat]
        else:
            raise NotImplementedError

        # association
        self.association = assignment_strats.hungarian_algorithm

        # loss on sim
        distance = distances.CosineSimilarity()
        reducer = reducers.AvgNonZeroReducer()
        self.sim_loss = losses.NTXentLoss(distance=distance, reducer=reducer)

    def training_step(self, batch, batch_idx):
        tracks, dets = self.train_val_preprocess(batch)
        self.sanity_checks(tracks, dets)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        loss = self.compute_loss(tracks, dets, td_sim_matrix)
        self.log_loss(loss, "train")
        return {
            "loss": loss,
            "dets": dets,
            "tracks": tracks,
            "td_sim_matrix": td_sim_matrix,
        }

    def validation_step(self, batch, batch_idx):
        tracks, dets = self.train_val_preprocess(batch)
        self.sanity_checks(tracks, dets)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        loss = self.compute_loss(tracks, dets, td_sim_matrix)
        self.log_loss(loss, "val")
        return {
            "loss": loss,
            "tracks": tracks,
            "dets": dets,
            "td_sim_matrix": td_sim_matrix,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        tracks, dets = self.predict_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets)
        association_matrix, association_result = self.association(td_sim_matrix,
                                                                  tracks.masks,
                                                                  dets.masks,
                                                                  sim_threshold=self.final_tracking_threshold,
                                                                  special_tokens=tracks.special_tokens or dets.special_tokens)
        # plt = display_bboxes(tracks, dets, None, batch["images"])
        # plt.show()
        return association_matrix, association_result, td_sim_matrix

    def forward(self, tracks, dets):
        tracks, dets = self.tokenize(tracks, dets)  # feats -> list(tokens)
        tracks, dets = self.merge(tracks, dets)  # list(tokens) -> tokens
        tracks, dets = self.gaffe(tracks, dets)  # tokens -> embs
        td_sim_matrix = self.similarity(tracks, dets)  # embs -> sim_matrix
        return tracks, dets, td_sim_matrix

    def train_val_preprocess(self,
                             batch):  # TODO merge with predict_preprocess, compute det/trask masks in getitem
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """
        if self.norm_coords is not None:
            batch = self.norm_coords(batch)
        tracks = Tracklets(batch["track_feats"], ~batch["track_targets"].isnan(), batch["track_targets"])
        dets = Detections(batch["det_feats"], ~batch["det_targets"].isnan(), batch["det_targets"])
        return tracks, dets

    def predict_preprocess(self, batch):
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """

        if self.norm_coords is not None:
            batch = self.norm_coords(batch)
        tracks = Tracklets(batch["track_feats"], batch["track_masks"],
                           batch["track_targets"] if "track_targets" in batch else None)
        dets = Detections(batch["det_feats"], batch["det_masks"],
                          batch["det_targets"] if "det_targets" in batch else None)
        return tracks, dets

    def sanity_checks(self, tracks, dets):
        """

        """

        # Function to check for duplicates in each batch
        def check_no_duplicate_integers(tensor):
            B, N = tensor.shape

            for i in range(B):
                batch = tensor[i]
                # Filter out NaNs using ~torch.isnan()
                filtered_integers = batch[~torch.isnan(batch)].int()
                # Check if the number of unique values matches the length of the filtered integers
                if len(filtered_integers) != len(torch.unique(filtered_integers)):
                    raise AssertionError(f"Duplicate target found in batch {i}")
            print("No duplicates found!")
            return True

        # assert check_no_duplicate_integers(tracks.targets)
        # assert check_no_duplicate_integers(dets.targets)

    def tokenize(self, tracks, dets):
        """
        Operate the tokenization step for the differents tokenizers
        :param dets: Detections
        :param tracks: Tracklets
        :return: updated dets and tracks with partial tokens in a dict not merged
        """
        tracks.tokens = {}
        dets.tokens = {}
        for n, t in self.tokenizers.items():
            tracks.tokens[n] = t(tracks)
            dets.tokens[n] = t(dets)
        return tracks, dets

    def similarity(self, tracks, dets):
        """
        Compute the similarity matrix between the tokens of dets and tracks.
        if tokens is a list of N tensors and not a single tensor, N similarity matrices are computed and averaged
        """
        # FIXME similarity_metric should be a list, a different metric could be used for each type of token
        if isinstance(tracks.embs, dict):
            td_sim_matrix = []
            for (tokenizer_name, t), (_, d) in zip(tracks.embs.items(), dets.embs.items()):
                # each token type has its own default distance (reid = cosine, bbox = iou, etc).
                # Use that default distance for a heuristic that would, for instance, combine IoU with cosine distance.
                if self.sim_strat == "default_for_each_token_type":
                    sm = similarity_metrics[self.tokenizers[tokenizer_name].default_similarity_metric]
                    td_sim_matrix.append(sm(t, tracks.masks, d, dets.masks))
                else:
                    td_sim_matrix.append(self.similarity_metric(t, tracks.masks, d, dets.masks))
            td_sim_matrix = torch.stack(td_sim_matrix).mean(dim=0)
        else:
            td_sim_matrix = self.similarity_metric(tracks.embs, tracks.masks, dets.embs, dets.masks)
        return td_sim_matrix

    def compute_loss(self, tracks, dets, *args):
        """
        :param dets: dataclass
            embs tensor float32 dim [B, D, E]
            confs tensor float32 dim [B, D]
            masks tensor bool dim [B, D]
            targets tensor float32 dim [B, D]
        :param tracks: dataclass
            embs tensor float32 dim [B, T, E]
            masks tensor bool dim [B, T]
            targets tensor float32 dim [B, T]
        :param td_sim_matrix: unused
        :return: sim_loss float32 and cls_loss float32
        """
        if not isinstance(tracks.embs, dict):
            tracks.embs = {"default": tracks.embs}
            dets.embs = {"default": dets.embs}

        n_tokens = len(tracks.embs.keys())
        B = list(tracks.embs.values())[0].shape[0]

        # Initialize loss variables
        sim_loss = torch.zeros((n_tokens, B), dtype=torch.float32, device=self.device)
        mask_sim_loss = torch.zeros((n_tokens, B), dtype=torch.bool, device=self.device)

        for h, token_name in enumerate(tracks.embs.keys()):
            tracks_embs = tracks.embs[token_name]
            dets_embs = dets.embs[token_name]

            for i in range(B):
                masked_track_embs = tracks_embs[i, tracks.masks[i]]
                masked_track_targets = tracks.targets[i, tracks.masks[i]]
                masked_det_embs = dets_embs[i, dets.masks[i]]
                masked_det_targets = dets.targets[i, dets.masks[i]]

                if (not tracks.special_tokens and (len(masked_det_embs) != 0 or len(masked_track_embs) != 0)) \
                    or (tracks.special_tokens and (len(masked_det_embs) > 1 or len(masked_track_embs) > 1)):
                    mask_sim_loss[h, i] = True

                    # Compute embeddings loss on all tracks/detections (track_ids >= 0)
                    valid_tracks = masked_track_targets >= 0
                    valid_dets = masked_det_targets >= 0
                    embeddings = torch.cat([masked_track_embs[valid_tracks],
                                            masked_det_embs[valid_dets]], dim=0)
                    labels = torch.cat([masked_track_targets[valid_tracks],
                                        masked_det_targets[valid_dets]], dim=0)
                    sim_loss[h, i] = self.sim_loss(embeddings, labels)

        # Compute mean losses over valid items in the batch
        sim_loss = sim_loss[mask_sim_loss].mean()
        # Handle NaN values
        sim_loss = sim_loss.nan_to_num(0)

        return sim_loss

    def log_loss(self, loss, step):
        loss_dict = {f"{step}/loss": loss}
        self.log_dict(
            loss_dict,
            on_epoch=True,
            on_step="train" == step,
            prog_bar="train" == step,
            logger=True,
        )

    def configure_optimizers(self):
        if hasattr(self.train_cfg, "finetune_tokenizers") and self.train_cfg.finetune_tokenizers:
            # Split parameters into tokenizer and non-tokenizer groups
            param_groups = [
                {
                    'params': [p for n, p in self.named_parameters() if
                               not n.startswith('tokenizers')],
                    'lr': self.train_cfg.init_lr
                },
                {
                    'params': [p for n, p in self.named_parameters() if
                               n.startswith('tokenizers')],
                    'lr': self.train_cfg.init_lr / 10
                    # Slower learning rate for tokenizer
                }
            ]
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.train_cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.train_cfg.init_lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        estimated_steps = self.trainer.estimated_stepping_batches
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=estimated_steps / 20,
            # FIXME very slow, call a lot of get_items
            num_training_steps=estimated_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_callbacks(self):
        from cameltrack.train.callbacks import SimMetrics # Only used for training
        callbacks = [SimMetrics()]
        return callbacks

    def on_save_checkpoint(self, checkpoint):
        # Add custom attributes to the checkpoint dictionary
        checkpoint['computed_sim_threshold'] = self.computed_sim_threshold
        checkpoint['computed_sim_threshold2'] = self.computed_sim_threshold2
        checkpoint['best_distr_overlap_threshold'] = self.best_distr_overlap_threshold

    def on_load_checkpoint(self, checkpoint):
        # Load custom attributes from the checkpoint dictionary
        self.computed_sim_threshold = checkpoint.get('computed_sim_threshold', None)
        self.computed_sim_threshold2 = checkpoint.get('computed_sim_threshold2', None)
        self.best_distr_overlap_threshold = checkpoint.get('best_distr_overlap_threshold', None)
        self.on_validation_end()

    def on_validation_end(self):
        if self.final_tracking_threshold is None or self.use_computed_threshold:
            assert self.computed_sim_threshold is not None, \
                "computed_sim_threshold or final_tracking_threshold must be set"
            self.final_tracking_threshold = self.computed_sim_threshold
            log.info(f"final_tracking_threshold set to {self.final_tracking_threshold} "
                     f"(use_computed_threshold={self.use_computed_threshold})")
        else:
            log.info(f"final_tracking_threshold already set to {self.final_tracking_threshold}. "
                     f"Skipping automatically computed value.")
        return