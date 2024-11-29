import torch
import torch.nn as nn
import math

from cameltrack.architecture.base_module import Module
from ..camel import Detections  # TODO: find better place

from hydra.utils import instantiate


class TemporalEncoder(Module):
    """
    Tokenize detections or tracklets into a single token.
    Detections are simply projected into a token space using det_tokenizer.
    Tracklets are projected into a token space using the projected detections by det_tokenizer and then encoded using a
    transformer encoder.
    """
    default_similarity_metric = "norm_euclidean"

    def __init__(
        self,
        det_tokenizer,
        hidden_dim,
        output_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
        freeze: bool = False,
        name: str = None,
        *args, **kwargs,
    ):
        super().__init__()
        # transformer parameter related
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # parameters and layers
        self.det_tokenizer = instantiate(det_tokenizer, hidden_dim=hidden_dim, _recursive_=False)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.occluded_token = nn.Parameter(torch.zeros(hidden_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.src_drop = nn.Dropout(dropout)
        self.src_norm = nn.LayerNorm(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear_out = nn.Linear(hidden_dim, output_dim, bias=True)

        # init weights and freeze
        if name is not None:
            self.init_weights(checkpoint_path=checkpoint_path, module_name=f"tokenizers.{name}")
        else:
            self.init_weights(checkpoint_path=checkpoint_path, module_name="tokenizers.TrackletEncoder")
        if freeze:
            for name, param in self.named_parameters():
                if name.startswith('linear_out'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        self.freeze = freeze

    def forward(self, x):
        B, N, S, E, O = *x.feats_masks.shape, self.hidden_dim, self.output_dim
        output = torch.zeros(
            (B, N, O),
            device=x.feats_masks.device,
            dtype=torch.float32,
        )

        tokens = self.det_tokenizer(x)  # [B, N, S, E]
        tokens = self.pos_encoder(tokens, x.feats["age"], x.feats_masks)
        tokens = self.src_norm(tokens)
        tokens = self.src_drop(tokens)

        src = tokens.permute(2, 0, 1, 3).reshape(S, B * N, E)  # [S, B*N, E]

        cls = self.cls.expand(-1, B * N, -1)
        src = torch.cat([src, cls], dim=0)

        cls_mask = torch.ones((B * N, 1), dtype=torch.bool, device=x.feats_masks.device)
        src_mask = torch.cat([x.feats_masks.reshape(B * N, S), cls_mask], dim=1)

        src = self.encoder(src, src_key_padding_mask=~src_mask)  # [S, B*N, E]
        src = src.permute(1, 0, 2).reshape(B, N, S + 1, E)  # [B, N, S, E]
        tokens = src[:, :, -1, :]  # [B, N, E]
        output[x.masks] = self.linear_out(tokens[x.masks])
        return output  # [B, N, O]


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x, age, mask):
        B, N, S, E = x.shape
        x = x.view(B * N, S, E)
        age = age.view(B * N, S).to(torch.long)
        mask = mask.view(B * N, S)
        age[mask] = age[mask].clamp(min=0, max=self.max_len - 1)
        x[mask] = x[mask] + self.pe[age[mask]]
        return x.view(B, N, S, E)


class BBoxLinProj(Module):
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
        return output


class KeypointsLinProj(Module):
    def __init__(self, hidden_dim, use_conf, dropout: float = 0.1):
        super().__init__()
        self.token_dim = hidden_dim
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
        return output


class EmbeddingsLinProj(Module):
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
        return output


class DetTokenizer(Module):
    def __init__(self, hidden_dim, feats_tokenizers):
        super().__init__()
        self.hidden_dim = hidden_dim
        module_list = []
        for feats_tokenizer in feats_tokenizers:
            module_list.append(instantiate(feats_tokenizer, hidden_dim=hidden_dim))
        self.feats_tokenizers = nn.ModuleList(module_list)

    def forward(self, x):
        tokens = torch.zeros(
            (*x.feats_masks.shape, self.hidden_dim),
            device=x.feats_masks.device,
            dtype=torch.float32,
        )
        for feats_tokenizer in self.feats_tokenizers:
            tokens[x.feats_masks] += feats_tokenizer(x)
        return tokens
