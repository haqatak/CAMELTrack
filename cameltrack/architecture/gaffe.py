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
        n_layers: int = 6,
        dim_feedforward: int = 4096,
        dropout: int = 0.1,
        activation_fn: str = "gelu",
        checkpoint_path: str = None,
        use_processed_track_tokens: bool = True,
        src_key_padding_mask: bool = True,
        enable_track_det_token: bool = False,
        enable_src_norm: bool = True,
        enable_src_drop: bool = True,
        enable_xavier_init: bool = True,
        enable_cls_token: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.use_processed_track_tokens = use_processed_track_tokens
        self.src_key_padding_mask = src_key_padding_mask

        self.enable_track_det_token = enable_track_det_token
        self.enable_src_norm = enable_src_norm
        self.enable_src_drop = enable_src_drop
        self.enable_xavier_init = enable_xavier_init
        self.enable_cls_token = enable_cls_token

        self.det_encoder = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.track_encoder = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        self.cls = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)  # required when a batch is empty, because without the cls token the attention would be totally false on some rows, and the model would return nan@
        self.src_norm = nn.LayerNorm(emb_dim)
        self.src_drop = nn.Dropout(dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            self.emb_dim, self.n_heads, self.dim_feedforward, self.dropout, batch_first=True, activation=self.activation_fn
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, self.n_layers)

        self.init_weights(checkpoint_path=checkpoint_path, module_name="transformer")

        if self.enable_xavier_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, tracks, dets):
        if self.enable_track_det_token:
            tracks.tokens[tracks.masks] += self.track_encoder
            dets.tokens[dets.masks] += self.det_encoder

        assert not torch.any(torch.isnan(tracks.tokens))  # FIXME still nans from MB or reid?
        assert not torch.any(torch.isnan(dets.tokens))

        if self.enable_cls_token:
            src = torch.cat(
                [dets.tokens, tracks.tokens, self.cls.repeat(dets.masks.shape[0], 1, 1)], dim=1
            )
        else:
            src = torch.cat([dets.tokens, tracks.tokens], dim=1)

        if self.enable_src_norm:
            src = self.src_norm(src)

        if self.enable_src_drop:
            src = self.src_drop(src)

        if self.src_key_padding_mask:
            if self.enable_cls_token:
                src_mask = torch.cat(
                    [
                        dets.masks,
                        tracks.masks,
                        torch.ones((dets.masks.shape[0], 1), device=dets.masks.device, dtype=torch.bool),
                    ],
                    dim=1,
                )
            else:
                src_mask = torch.cat([dets.masks, tracks.masks], dim=1)
            x = self.encoder(src, src_key_padding_mask=~src_mask)
        else:
            src_mask = self.mask(dets.masks, tracks.masks)
            x = self.encoder(src, mask=src_mask)  # [B, D(+P) + T(+P) + 1, E]
        if self.use_processed_track_tokens:
            tracks.embs = x[:, dets.masks.shape[1]: dets.masks.shape[1] + tracks.masks.shape[1]]  # [B, T(+P), E]
        else:
            tracks.embs = tracks.tokens  # [B, T(+P), E]
        dets.embs = x[:, :dets.masks.shape[1]]  # [B, D(+P), E]

        assert not torch.any(torch.isnan(tracks.embs))
        assert not torch.any(torch.isnan(dets.embs))

        return tracks, dets

    def mask(self, dets_masks, tracks_masks):
        """
        dets_masks: Tensor [B, D(+P)]
        tracks_masks: Tensor [B, T(+P)]

        returns: mask of shape [B * n_heads, D(+P)+T(+P)+1, D(+P)+T(+P)+1]
            padded values are set to True else is False
        """
        src_mask = torch.cat(
            [
                dets_masks,
                tracks_masks,
                torch.ones((dets_masks.shape[0], 1), device=dets_masks.device, dtype=torch.bool),
            ],
            dim=1,
        )

        src_mask = ~(src_mask.unsqueeze(2) * src_mask.unsqueeze(1))

        # just keep self-attention (otherwise becomes nan)
        indices = torch.arange(src_mask.shape[1])
        src_mask[:, indices, indices] = False

        # repeat for the n_heads
        src_mask = src_mask.repeat_interleave(self.n_heads, dim=0)
        return src_mask
