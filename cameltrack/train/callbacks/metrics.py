import pytorch_lightning as pl
import torch
import torchmetrics
import logging
import wandb
from torch import Tensor
from torchmetrics import Metric

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class SimMetrics(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.acc = Accuracy().to(pl_module.device)
        self.roc = torchmetrics.classification.BinaryROC()
        self.auroc = torchmetrics.classification.BinaryAUROC()
        
        # fixme new roc check if it is ok and clean
        self.running_sim_matrix = torch.tensor([], device=pl_module.device)
        self.running_gt_matrix = torch.tensor([], dtype=torch.bool, device=pl_module.device)
        self.running_tracks_mask = torch.tensor([], dtype=torch.bool, device=pl_module.device)
        self.running_dets_mask = torch.tensor([], dtype=torch.bool, device=pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        tracks = outputs["tracks"]
        dets = outputs["dets"]
        td_sim_matrix = outputs["td_sim_matrix"]
        gt_ass_matrix = tracks.targets.unsqueeze(2) == dets.targets.unsqueeze(1)
        gt_ass_matrix[~tracks.masks.unsqueeze(2).repeat(1, 1, gt_ass_matrix.shape[2])] = False
        gt_ass_matrix[~dets.masks.unsqueeze(1).repeat(1, gt_ass_matrix.shape[1], 1)] = False

        # roc
        valid_idx = tracks.masks.unsqueeze(2) * dets.masks.unsqueeze(1)
        valid_sim_matrix = outputs["td_sim_matrix"][valid_idx]
        valid_gt_ass_matrix = gt_ass_matrix[valid_idx].to(torch.int32)
        self.roc.update(valid_sim_matrix, valid_gt_ass_matrix)
        self.auroc.update(valid_sim_matrix, valid_gt_ass_matrix)

        if tracks.special_tokens or dets.special_tokens:
            binary_ass_matrix, _ = pl_module.association(td_sim_matrix, tracks.masks, dets.masks, special_tokens=tracks.special_tokens or dets.special_tokens)  # FIXME use best th from previous epoch
            actual_binary_ass_matrix = binary_ass_matrix[:, :-1, :-1]
        else:
            # old accuracy code, could be removed
            intersection = torch.eq(tracks.targets.unsqueeze(dim=2), dets.targets.unsqueeze(dim=1))
            inter_tracks_masks = intersection.any(dim=2)
            inter_dets_masks = intersection.any(dim=1)
            binary_ass_matrix, _ = pl_module.association(td_sim_matrix, inter_tracks_masks, inter_dets_masks)
            actual_binary_ass_matrix = binary_ass_matrix

        # assert one to one match:
        assert actual_binary_ass_matrix.sum(dim=2).max() < 2
        assert actual_binary_ass_matrix.sum(dim=1).max() < 2

        if tracks.special_tokens or dets.special_tokens:
            preds, targets = compute_tracklet_detection_preds_targets(binary_ass_matrix, gt_ass_matrix, tracks.masks, dets.masks)
        else:
            # binary_ass_matrix and gt_ass_matrix computed before
            idx = torch.arange(td_sim_matrix.numel(), device=td_sim_matrix.device).reshape(td_sim_matrix.shape)
            preds = idx[binary_ass_matrix]
            targets = idx[gt_ass_matrix]
        self.acc.update(preds, targets)

        # fixme new roc check if it is ok and clean
        self.running_sim_matrix = torch.cat((self.running_sim_matrix, td_sim_matrix), dim=0)
        self.running_gt_matrix = torch.cat((self.running_gt_matrix, gt_ass_matrix), dim=0)
        self.running_tracks_mask = torch.cat((self.running_tracks_mask, tracks.masks), dim=0)
        self.running_dets_mask = torch.cat((self.running_dets_mask, dets.masks), dim=0)

    def on_validation_epoch_end(self, trainer, pl_module):
        best_roc_threshold = log_roc(self.roc, self.auroc, pl_module, trainer.current_epoch, "sim")
        pl_module.computed_sim_threshold = best_roc_threshold
        log.info(f"Best computed_sim_threshold found on validation set: {best_roc_threshold:.3f}")
        pl_module.log_dict({"val/best_roc_threshold": best_roc_threshold}, logger=True, on_step=False,
                           on_epoch=True)
        pl_module.log_dict(
            {"val/sim_acc": self.acc.compute().item()}, logger=True, on_step=False, on_epoch=True
        )

        # fixme new roc check if it is ok and clean
        # thresholds = torch.linspace(0., 1., 101, device=pl_module.device)
        # accuracies = torch.zeros_like(thresholds, device=pl_module.device)
        # for i, th in enumerate(thresholds):
        #     binary_ass_matrix, _ = pl_module.association(
        #         self.running_sim_matrix, self.running_tracks_mask, self.running_dets_mask, th,
        #     )
        #     correct = binary_ass_matrix == self.running_gt_matrix
        #     correct_tracks = torch.all(correct, dim=1)[self.running_tracks_mask]
        #     correct_dets = torch.all(correct, dim=2)[self.running_dets_mask]
        #     accuracies[i] = (correct_tracks.sum() + correct_dets.sum()) / (self.running_tracks_mask.sum() + self.running_dets_mask.sum())
        # computed_sim_threshold2 = thresholds[torch.argwhere(accuracies == torch.amax(accuracies)).flatten()[-1]]
        # pl_module.computed_sim_threshold2 = computed_sim_threshold2
        # pl_module.log_dict({"val/computed_sim_threshold2": computed_sim_threshold2}, logger=True, on_step=False, on_epoch=True)
        # log.info(f"Best computed_sim_threshold2 found on validation set: {pl_module.computed_sim_threshold2:.3f}")

class ClsMetrics(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        self.roc = torchmetrics.classification.BinaryROC()
        self.auroc = torchmetrics.classification.BinaryAUROC()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        dets = outputs["dets"]
        preds = dets.confs[dets.masks]
        targets = (dets.targets[dets.masks] >= 0).to(torch.int32)
        self.roc.update(preds, targets)
        self.auroc.update(preds, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        best_roc_cls_threshold = log_roc(self.roc, self.auroc, pl_module, trainer.current_epoch, "cls")
        pl_module.best_roc_cls_threshold = best_roc_cls_threshold
        log.info(f"Best best_roc_cls_threshold found on validation set: {best_roc_cls_threshold:.3f}")

class Accuracy(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


def compute_tracklet_detection_preds_targets(binary_ass_matrix, gt_ass_matrix, tracks_masks, dets_masks):
    B, N_plus_1, M_plus_1 = binary_ass_matrix.shape  # B = batch size, N+1 = tracklets, M+1 = detections
    N, M = N_plus_1 - 1, M_plus_1 - 1  # Exclude the special token row/column for tracklets and detections

    # Convert boolean tensors to integers for argmax
    binary_ass_matrix = binary_ass_matrix.to(torch.int64)
    gt_ass_matrix = gt_ass_matrix.to(torch.int64)

    # Masking out invalid tracklets (where tracklet does not exist)
    # tracks_masks is (B, N+1), the last element corresponds to the special token
    valid_tracks = tracks_masks[:, :N].to(torch.bool)  # Tracklets except the special token
    valid_dets = dets_masks[:, :M].to(torch.bool)  # Detections except the special token

    # Vectorized extraction of tracklet assignments (rows) for each batch
    preds_tracklets = torch.argmax(binary_ass_matrix[:, :N, :], dim=2)  # Predicted tracklet assignment, shape (B, N)
    targets_tracklets = torch.argmax(gt_ass_matrix[:, :N, :], dim=2)  # Ground truth tracklet assignment, shape (B, N)

    # Vectorized extraction of detection assignments (columns) for each batch
    preds_detections = torch.argmax(binary_ass_matrix[:, :, :M], dim=1)  # Predicted detection assignment, shape (B, M)
    targets_detections = torch.argmax(gt_ass_matrix[:, :, :M], dim=1)  # Ground truth detection assignment, shape (B, M)

    # Apply track masks (exclude non-existing tracklets)
    valid_preds_tracklets = preds_tracklets[valid_tracks].view(-1)  # Flattened valid predictions for tracklets
    valid_targets_tracklets = targets_tracklets[valid_tracks].view(-1)  # Flattened valid targets for tracklets

    # Apply detection masks (exclude non-existing detections)
    valid_preds_detections = preds_detections[valid_dets].view(-1)  # Flattened valid predictions for detections
    valid_targets_detections = targets_detections[valid_dets].view(-1)  # Flattened valid targets for detections

    # Concatenate the valid tracklet and detection assignments
    preds = torch.cat([valid_preds_tracklets, valid_preds_detections], dim=0)  # Combined valid preds (N + M)
    targets = torch.cat([valid_targets_tracklets, valid_targets_detections], dim=0)  # Combined valid targets (N + M)

    return preds, targets


def log_roc(roc, auroc, pl_module, epoch, name):
    fpr, tpr, thresholds = roc.compute()
    best_threshold = thresholds[torch.argmax(tpr - fpr)]
    b_auroc = auroc.compute()
    fig_, ax_ = roc.plot(score=True)
    ax_.set_aspect("equal")
    ax_.set_title(f"val/{name}_ROC - epoch {epoch}")
    idx = [i for i in range(0, len(fpr), max(1, len(fpr) // 20))]
    for i, j in enumerate(idx):
        ax_.annotate(
            f"{thresholds[j]:.3f}",
            xy=(fpr[j], tpr[j]),
            xytext=((-1) ** i * 20, (-1) ** (i + 1) * 20),
            textcoords="offset points",
            fontsize=6,
            arrowprops=dict(arrowstyle="->"),
            ha="center",
        )
    log_dict = {}
    log_dict[f"val/{name}_opt_th"] = best_threshold
    log_dict[f"val/{name}_auroc"] = b_auroc.item()
    pl_module.log_dict(log_dict, logger=True, on_step=False, on_epoch=True)
    pl_module.logger.experiment.log({f"val/{name}_ROC": wandb.Image(fig_)})
    plt.close(fig_)
    return best_threshold
