import torch
import pytorch_lightning as pl
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

        # acc
        intersection = torch.eq(tracks.targets.unsqueeze(dim=2), dets.targets.unsqueeze(dim=1))
        inter_tracks_masks = intersection.any(dim=2)
        inter_dets_masks = intersection.any(dim=1)
        binary_ass_matrix, _ = pl_module.association(td_sim_matrix, inter_tracks_masks, inter_dets_masks)

        # assert one to one match:
        assert binary_ass_matrix.sum(dim=2).max() < 2
        assert binary_ass_matrix.sum(dim=1).max() < 2

        # binary_ass_matrix and gt_ass_matrix computed before
        idx = torch.arange(td_sim_matrix.numel(), device=td_sim_matrix.device).reshape(td_sim_matrix.shape)
        preds = idx[binary_ass_matrix]
        targets = idx[gt_ass_matrix]
        self.acc.update(preds, targets)

        self.running_sim_matrix = torch.cat((self.running_sim_matrix, td_sim_matrix), dim=0)
        self.running_gt_matrix = torch.cat((self.running_gt_matrix, gt_ass_matrix), dim=0)
        self.running_tracks_mask = torch.cat((self.running_tracks_mask, tracks.masks), dim=0)
        self.running_dets_mask = torch.cat((self.running_dets_mask, dets.masks), dim=0)

    def on_validation_epoch_end(self, trainer, pl_module):
        best_roc_threshold = log_roc(self.roc, self.auroc, pl_module, trainer.current_epoch, "sim")
        pl_module.computed_sim_threshold = best_roc_threshold
        log.info(f"Best computed_sim_threshold found on validation set: {best_roc_threshold:.3f}")
        pl_module.log_dict({"val/computed_sim_threshold": best_roc_threshold}, logger=True, on_step=False, on_epoch=True)
        pl_module.log_dict({"val/sim_acc": self.acc.compute().item()}, logger=True, on_step=False, on_epoch=True)
        log_hist(self.running_sim_matrix, self.running_gt_matrix, best_roc_threshold, pl_module, trainer.current_epoch)

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

def log_roc(roc, auroc, pl_module, epoch, name):
    fpr, tpr, thresholds = roc.compute()
    fpr, tpr, thresholds = fpr.cpu(), tpr.cpu(), thresholds.cpu()
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
    log_dict[f"val/{name}_auroc"] = b_auroc.item()
    pl_module.log_dict(log_dict, logger=True, on_step=False, on_epoch=True)
    pl_module.logger.experiment.log({f"val/{name}_ROC": wandb.Image(fig_)})
    plt.close(fig_)
    return best_threshold

def log_hist(sim_matrix, gt_matrix, best_threshold, pl_module, epoch):
    sim_pos = sim_matrix[gt_matrix]
    sim_neg = sim_matrix[~gt_matrix & (sim_matrix != -float("inf"))]

    plt.figure()
    plt.hist(sim_pos.cpu(), bins=50, alpha=0.5, color='green', density=True, label="Positive")
    plt.hist(sim_neg.cpu(), bins=50, alpha=0.5, color='red', density=True, label="Negative")
    plt.axvline(x=best_threshold, color='blue', linestyle='--')
    plt.xlabel("Similarity")
    plt.ylabel("Density")
    plt.legend(loc='upper left')
    plt.title(f"Histogram of Similarity - epoch {epoch}")
    plt.tight_layout()
    pl_module.logger.experiment.log({f"val/sim_distribution": wandb.Image(plt.gcf())})
    plt.close()
