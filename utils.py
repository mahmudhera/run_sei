import numpy as np
from scipy.stats import pearsonr
from collections import OrderedDict
import torch


def load_state_dict_flexible(ckpt_path: str, map_location="cpu"):
    """
    Loads a checkpoint and returns a cleaned state_dict that matches Sei() keys.

    Handles common cases:
      - saved from DataParallel: keys start with "module."
      - saved from a wrapper with attribute "model": keys start with "model."
      - both: "module.model."
    """
    obj = torch.load(ckpt_path, map_location=map_location)

    # Some checkpoints store {"state_dict": ...} or {"model_state_dict": ...}
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state = obj["model_state_dict"]
        else:
            # might already be a raw state_dict
            state = obj
    else:
        # very uncommon, but just in case
        state = obj

    new_state = OrderedDict()
    for k, v in state.items():
        nk = k

        # strip DataParallel
        if nk.startswith("module."):
            nk = nk[len("module."):]

        # strip wrapper attribute
        if nk.startswith("model."):
            nk = nk[len("model."):]

        # also handle the combined case (module.model.)
        if nk.startswith("model."):
            nk = nk[len("model."):]

        new_state[nk] = v

    return new_state


def correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]