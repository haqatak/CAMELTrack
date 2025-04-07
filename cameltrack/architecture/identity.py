import logging

from cameltrack.architecture.base_module import Module

log = logging.getLogger(__name__)

class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tracks, dets):
        dets.embs = dets.tokens
        tracks.embs = tracks.tokens
        return tracks, dets
