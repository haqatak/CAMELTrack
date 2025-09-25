import logging
import torch
import torch.nn as nn

from cameltrack.architecture.base_module import Module

log = logging.getLogger(__name__)

class GAFFE(Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Required when a batch is empty, because without the cls token the attention would be totally false on some rows, and the model would return nan
        self.cls = nn.Parameter(torch.zeros(emb_dim))

        self.src_norm = nn.LayerNorm(emb_dim)
        self.src_drop = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            self.emb_dim, self.n_heads, self.dim_feedforward, self.dropout, batch_first=True, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, self.n_layers)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="gaffe")

    def forward(self, tracks, dets):
        """
        Standard forward pass for a transformer encoder.
        Inputs are ordered by [detections, tracklets, cls_token]
        Outputs are tracklet and detection objects updated with the processed embeddings
        """
        src = torch.cat([dets.tokens, tracks.tokens, self.cls.repeat(dets.masks.shape[0], 1, 1)], dim=1)
        src = self.src_norm(src)
        src = self.src_drop(src)

        src_mask = torch.cat(
            [
                dets.masks,
                tracks.masks,
                torch.ones((dets.masks.shape[0], 1), device=dets.masks.device, dtype=torch.bool),
            ],
            dim=1,
        )
        if src.device.type == "mps":
            x = self.encoder.to("cpu")(src.to("cpu"), src_key_padding_mask=~src_mask.to("cpu"))
            x = x.to(src.device)
        else:
            x = self.encoder(src, src_key_padding_mask=~src_mask)

        tracks.embs = x[:, dets.masks.shape[1]: dets.masks.shape[1] + tracks.masks.shape[1]]  # [B, T(+P), E]
        dets.embs = x[:, :dets.masks.shape[1]]  # [B, D(+P), E]

        return tracks, dets
