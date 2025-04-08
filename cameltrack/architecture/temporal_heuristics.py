import torch
import torch.nn as nn
import numpy as np

from typing import List

from torchvision.ops import MLP

from cameltrack.architecture.base_module import Module
from cameltrack.utils.kalman_filter import Detection, Track

class LastBbox(Module):
    """
    The last bbox of a tracklet will be directly encoded with the MLP to produce the output token.
    """

    def __init__(self,
            output_dim: int,
            use_mlp: bool = False,
            hidden_channels: List[int] = [16, 32],
            checkpoint_path: str = None,
            **kwargs
         ):
        super().__init__()

        self.output_dim = output_dim
        self.use_mlp = use_mlp
        self.default_similarity_metric = "cosine" if use_mlp else "iou"

        if self.use_mlp:
            self.hidden_channels = hidden_channels
            self.mlp = MLP(
                in_channels=4,
                hidden_channels=self.hidden_channels+[output_dim],
                norm_layer=nn.BatchNorm1d,
                activation_layer=nn.ReLU,
                bias=True,
            )

        self.init_weights(checkpoint_path=checkpoint_path, module_name="temp_encs.lastbbox")

    def forward(self, x):
        bboxes, masks = x.feats['bbox_ltwh'][:, :, 0], x.feats_masks[:, :, 0]
        if self.use_mlp:
            bboxes = self.mlp(bboxes[masks])
            results = torch.zeros(
                (*masks.shape, self.output_dim),
                device=masks.device,
                dtype=torch.float32,
            )
        else:
            bboxes = bboxes[masks]
            results = torch.zeros(
                (*masks.shape, 4),
                device=masks.device,
                dtype=torch.float32,
            )
        results[masks] = bboxes
        return results.squeeze(dim=2)



class PartsEmbeddingsEMA(Module):
    default_similarity_metric = "cosine"

    def __init__(self,
             output_dim: int,
             use_lin_proj: bool = False,
             use_parts: bool = True,
             num_parts: int = 5,
             emb_size: int = 128,
             dropout: float = 0.1,
             alpha: float = 0.9,
             checkpoint_path: str = None,
             **kwargs
        ):
        super().__init__()

        self.output_dim = output_dim
        self.use_lin_proj = use_lin_proj
        self.use_parts = use_parts
        self.num_parts = num_parts
        self.emb_size = emb_size
        self.alpha = alpha

        self.linears = nn.ModuleList([nn.Linear(emb_size, output_dim, bias=True)] * (num_parts + 1 if use_parts else 1))
        self.drop = nn.Dropout(dropout)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="temp_encs.partsembeddingsema")

    def forward(self, x):
        # embs.shape : [B, N, T, num_parts, emb_dim]
        embs, vis, masks = x.feats["embeddings"], x.feats["visibility_scores"], x.feats_masks
        if masks.shape[2] > 1:
            embs, vis, masks = self.smart_ema(embs, vis, masks, self.alpha)

        if self.use_lin_proj:
            output = self.linears[0](embs[masks][:, 0, :])
            for i, linear in enumerate(self.linears[1:]):
                output += linear(embs[masks][:, i + 1, :]) * vis[masks][:, i + 1].unsqueeze(-1)
            output = self.drop(output)
            results = torch.zeros(
                (*masks.shape, self.output_dim),
                device=masks.device,
                dtype=torch.float32,
            )
            results[masks] = output
        else:
            results = embs
            if not self.use_parts:
                results = embs[:, :, :, 0]
            results = torch.flatten(results, start_dim=3)
        return results.squeeze(dim=2)  # results.shape : [B, N, output_dim]

    def smart_ema(self, embs, vis, masks, alpha=0.9):
        """
        :param embs: [B, N, T, num_parts, emb_dim]
        :param vis: [B, N, T, num_parts]
        :param masks: [B, N, T]
        :return: new_embs [B, N, 1, num_parts, emb_dim], new_vis [B, N, 1, num_parts], new_masks [B, N, 1]
        """
        # FIXME batch it
        new_embeddings = torch.zeros(
            (embs.shape[0], embs.shape[1], 1, embs.shape[3], embs.shape[4]),
            device=embs.device,
            dtype=torch.float32,
        )
        new_vis = torch.zeros(
            (embs.shape[0], embs.shape[1], 1, vis.shape[3]), device=embs.device, dtype=torch.float32
        )
        new_masks = torch.zeros(
            (embs.shape[0], embs.shape[1], 1), device=embs.device, dtype=torch.bool
        )
        for b in range(masks.shape[0]):
            for n in range(masks.shape[1]):
                for t in reversed(range(masks.shape[2])):
                    if masks[b, n, t]:
                        # For a given body part P, if:
                        # - P is visible in both the tracklet and the detection, xor = False and ema_scores_tracklet=ema_alpha and ema_scores_detection=1-ema_alpha -> both tracklet and det features are used in a normal EMA update step
                        # - P is visible in the tracklet but not in the detection, xor = True and ema_scores_tracklet=1 and ema_scores_detection=0 -> smooth_feat=tracklet_features
                        # - P is visible in the detection but not in the tracklet, xor = True and ema_scores_tracklet=0 and ema_scores_detection=1 -> smooth_feat=detection_features
                        # - P is not visible in both the tracklet and the detection, xor = False and ema_scores_tracklet=0 and ema_scores_detection=0 -> smooth_feat=1 (TODO why?)

                        tracklet_features = new_embeddings[b, n, 0]
                        detection_features = embs[b, n, t]

                        xor = torch.logical_xor(new_vis[b, n, 0], vis[b, n, t])
                        ema_scores_tracklet = (
                            new_vis[b, n, 0] * vis[b, n, t]
                        ) * alpha + xor * new_vis[b, n, 0]
                        ema_scores_detection = (new_vis[b, n, 0] * vis[b, n, t]) * (
                            1 - alpha
                        ) + xor * vis[b, n, t]
                        smooth_feat = (
                            ema_scores_tracklet.unsqueeze(dim=1) * tracklet_features
                            + ema_scores_detection.unsqueeze(dim=1) * detection_features
                        )
                        new_embeddings[b, n, 0] = smooth_feat

                        smooth_visibility_scores = torch.maximum(new_vis[b, n, 0], vis[b, n, t])
                        new_vis[b, n, 0] = smooth_visibility_scores

                        new_masks[b, n, 0] = True
        return new_embeddings, new_vis, new_masks


class KFBbox(Module):
    # FIXME could be cleaned a bit more
    def __init__(self,
             output_dim: int,
             max_length: int = 20,
             use_mlp: bool = False,
             use_confidence: bool = False,
             hidden_channels: List[int] = [16, 32],
             checkpoint_path: str = None,
             **kwargs
         ):
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_confidence = use_confidence

        self.use_mlp = use_mlp
        self.hidden_channels = hidden_channels
        self.mlp = MLP(
            in_channels=4,
            hidden_channels=hidden_channels + [self.output_dim],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
        )

        self.default_similarity_metric = "cosine" if self.use_mlp else "iou"

        self.init_weights(checkpoint_path=checkpoint_path, module_name="temp_encs.kfbbox")

    def forward(self, x):
        kf_pred__bboxes = self._get_kf_predictions(x) if x.feats['bbox_ltwh'].shape[2] > 1 else x.feats['bbox_ltwh'][:, :, 0]
        if not self.use_mlp:
            return kf_pred__bboxes
        valid_last_bboxes_mask = x.feats_masks[..., 0].flatten()
        valid_last_bboxes = kf_pred__bboxes.flatten(0, 1)[valid_last_bboxes_mask]
        valid_bbox_features = self.mlp(valid_last_bboxes)
        bbox_features = torch.zeros((len(valid_last_bboxes_mask), self.output_dim), device=valid_bbox_features.device)
        bbox_features[valid_last_bboxes_mask] = valid_bbox_features
        bbox_features = bbox_features.unflatten(0, x.feats_masks[..., 0].shape)
        return bbox_features

    def _get_kf_predictions(self, tracklets):
        device = tracklets.feats['bbox_ltwh'].device

        # Convert tensors to numpy and handle batch dimension
        tracklets_bbox_ltwh = tracklets.feats['bbox_ltwh'].cpu().numpy()
        tracklets_age = tracklets.feats['age'].squeeze(-1).cpu().numpy()
        tracklets_conf = tracklets.feats['bbox_conf'].squeeze(-1).cpu().numpy()
        # Initialize list to store predictions for each batch
        batch_predictions = []

        # Iterate over each element in the batch
        for batch_idx in range(tracklets_bbox_ltwh.shape[0]):  # Handle batch dimension
            tracklet_bbox_ltwh_batch = tracklets_bbox_ltwh[batch_idx][:, :self.max_length]
            tracklet_age_batch = tracklets_age[batch_idx][:, :self.max_length]
            tracklet_conf_batch = tracklets_conf[batch_idx][:, :self.max_length]

            predictions = []
            for t_idx, tracklet_bbox_ltwh in enumerate(tracklet_bbox_ltwh_batch):
                if tracklets.masks[batch_idx, t_idx]:
                    # Check if each bbox has all coordinates valid (i.e., no NaNs in the bbox)
                    valid_bbox_mask = ~np.isnan(tracklet_bbox_ltwh).any(axis=1)

                    if valid_bbox_mask.any():
                        # Find the highest index in the first dimension where all bbox coordinates are available
                        if valid_bbox_mask.any():
                            highest_non_nan_idx = np.max(np.where(valid_bbox_mask)[0])
                        else:
                            highest_non_nan_idx = -1  # Return -1 if no valid bbox is found

                        bbox_ltwh = tracklet_bbox_ltwh[highest_non_nan_idx]
                        confidence = tracklet_conf_batch[t_idx, highest_non_nan_idx] if self.use_confidence else 1
                        detection = Detection(id=0, bbox_ltwh=bbox_ltwh, confidence=confidence)
                        tracklet_kf = Track(detection=detection, track_id=0, class_id=0, conf=confidence, n_init=0, max_age=10, ema_alpha=0.1, max_kalman_prediction_without_update=7)
                        detections_per_age = {age: (bbox_ltwh, conf) for age, bbox_ltwh, conf in
                                              zip(tracklet_age_batch[t_idx, :highest_non_nan_idx], tracklet_bbox_ltwh[:highest_non_nan_idx], tracklet_conf_batch[t_idx, :highest_non_nan_idx]) if age > 0}


                        if len(tracklet_age_batch[t_idx, :highest_non_nan_idx]) > 0 and not np.isnan(max(tracklet_age_batch[t_idx, :highest_non_nan_idx])):
                            oldest = int(max(tracklet_age_batch[t_idx, :highest_non_nan_idx]))
                            for age in range(oldest, 0, -1):
                                tracklet_kf.predict()
                                if age in detections_per_age:
                                    bbox_ltwh, conf = detections_per_age[age]
                                    confidence = conf if self.use_confidence else 1
                                    detection = Detection(id=0, bbox_ltwh=bbox_ltwh, confidence=confidence)
                                    tracklet_kf.update(detection, 0, confidence)

                        tracklet_kf.predict()
                        predicted_bbox_ltwh = tracklet_kf.to_ltwh()
                        if np.any(np.isnan(predicted_bbox_ltwh)):
                            predicted_bbox_ltwh = tracklet_kf.last_detection.to_ltwh()
                        predictions.append(predicted_bbox_ltwh)
                    else:
                        raise ValueError(f"No valid bbox found in tracklet {t_idx}")
                else:
                    predictions.append(np.nan * np.ones(4))


            # Stack predictions for this batch element
            predictions = np.stack(predictions, axis=0)
            batch_predictions.append(predictions)

        # Stack all batch predictions back together
        batch_predictions = np.stack(batch_predictions, axis=0)

        # Convert the result back to a tensor and maintain batch dimension
        tracks_pred_bbox_ltwh = torch.from_numpy(batch_predictions).to(device)

        # to float32
        tracks_pred_bbox_ltwh = tracks_pred_bbox_ltwh.to(torch.float32)

        return tracks_pred_bbox_ltwh
