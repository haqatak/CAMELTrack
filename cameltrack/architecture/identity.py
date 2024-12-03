import logging

from plugins.track.dd_sort.simformer.modules.transformers.base_module import Module

log = logging.getLogger(__name__)

class Identity(Module):
    def __init__(self, checkpoint_path: str = None, *args, **kwargs):
        super().__init__()
        self.init_weights(checkpoint_path=checkpoint_path, module_name="transformer")

    def forward(self, tracks, dets):
        dets.embs = dets.tokens
        tracks.embs = tracks.tokens
        return tracks, dets
