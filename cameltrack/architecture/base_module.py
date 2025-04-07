import logging

import torch
import torch.nn as nn

from pathlib import Path
from collections import OrderedDict


log = logging.getLogger(__name__)

class Module(nn.Module):
    def __init__(self, checkpoint_path: str = None, **kwargs):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def init_weights(self, checkpoint_path: str = None, module_name: str = None):
        if checkpoint_path and module_name:
            if not Path(checkpoint_path).exists():
                log.warning(f"Checkpoint path {checkpoint_path} not found. Will use random weights.")
                return
            state_dict = torch.load(checkpoint_path)["state_dict"]
            state_dict = OrderedDict(
                (k, state_dict[k]) for k in state_dict if k.startswith(module_name)
            )
            state_dict = OrderedDict(
                (k.replace(module_name + ".", ""), v) for k, v in state_dict.items()
            )
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded checkpoint weights for {module_name} from `{checkpoint_path}`.")
            if missing:
                log.warning(f"Missing keys while loading: {missing}. Initializing random weights for those.")
            if unexpected:
                log.warning(f"Unexpected keys while loading: {unexpected}. Initializing random weights for those.")
            params_to_init = missing + unexpected
        else:
            params_to_init = self.named_modules()
        modules = dict(self.named_modules())
        # for key, _ in params_to_init:
        #     if key in modules:
        #         print("init weights for", key)
        #         layer = modules[key]
        #         if layer.dim() > 1:
        #             nn.init.xavier_uniform_(layer)
        #         else:
        #             nn.init.uniform_(layer)
