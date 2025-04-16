import torch
import torch.nn as nn

from cameltrack.architecture.base_module import Module


class BBoxLinProj(Module):
    input_columns = ["bbox_ltwh", "bbox_conf"]
    training_columns = ["drop_bbox"]

    def __init__(self, hidden_dim, use_conf, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_conf = use_conf
        self.dropout = dropout

        in_features = 4 + (1 if use_conf else 0)
        self.linear = nn.Linear(in_features, hidden_dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        bbox_feats = x.feats["bbox_ltwh"]
        if self.use_conf:
            bbox_feats = torch.cat([bbox_feats, x.feats["bbox_conf"]], dim=-1)
        output = self.drop(self.linear(bbox_feats[x.feats_masks]))
        if "drop_bbox" in x.feats:
            output[x.feats["drop_bbox"][x.feats_masks].squeeze() == 1] = 0.
        tokens = torch.zeros(
            (*x.feats_masks.shape, self.hidden_dim),
            device=x.feats_masks.device,
            dtype=torch.float32,
        )
        tokens[x.feats_masks] = output
        return tokens


class KeypointsLinProj(Module):
    input_columns = ["keypoints_xyc"]
    training_columns = ["drop_kps"]

    def __init__(self, hidden_dim, use_conf, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_conf = use_conf
        self.dropout = dropout

        in_features = 17 * 2 + (17 if use_conf else 0)
        self.linear = nn.Linear(in_features, hidden_dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        keypoints_feats = x.feats["keypoints_xyc"]
        if not self.use_conf:
            keypoints_feats = keypoints_feats[..., :2]
        keypoints_feats = keypoints_feats.reshape(*x.feats_masks.shape, -1)
        output = self.drop(self.linear(keypoints_feats[x.feats_masks]))
        if "drop_kps" in x.feats:
            output[x.feats["drop_kps"][x.feats_masks].squeeze() == 1] = 0.
        tokens = torch.zeros(
            (*x.feats_masks.shape, self.hidden_dim),
            device=x.feats_masks.device,
            dtype=torch.float32,
        )
        tokens[x.feats_masks] = output
        return tokens


class PartsEmbeddingsLinProj(Module):
    input_columns = ["embeddings", "visibility_scores"]
    training_columns = ["drop_app"]

    def __init__(self, hidden_dim, use_parts, num_parts: int = 5, emb_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_parts = use_parts
        self.num_parts = num_parts
        self.emb_size = emb_size
        self.dropout = dropout

        self.linears = nn.ModuleList([nn.Linear(emb_size, hidden_dim, bias=True)] * (num_parts + 1 if use_parts else 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        embeddings = x.feats["embeddings"] * x.feats["visibility_scores"].unsqueeze(-1)  # [B, N, S, num_parts, emb_size]
        output = self.linears[0](embeddings[x.feats_masks][:, 0, :])
        for i, linear in enumerate(self.linears[1:]):
            output += linear(embeddings[x.feats_masks][:, i + 1, :]) * x.feats["visibility_scores"][x.feats_masks][:, i + 1].unsqueeze(-1)
        output = self.drop(output)
        if "drop_app" in x.feats:
            output[x.feats["drop_app"][x.feats_masks].squeeze() == 1] = 0.
        tokens = torch.zeros(
            (*x.feats_masks.shape, self.hidden_dim),
            device=x.feats_masks.device,
            dtype=torch.float32,
        )
        tokens[x.feats_masks] = output
        return tokens
