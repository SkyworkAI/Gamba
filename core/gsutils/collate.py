
import random
from collections.abc import Mapping, Sequence
import numpy as np

import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")
    if isinstance(batch[0], dict):
        new_batch = {}
        for key in batch[0].keys():
            if key in ["gsparams", "offset"]:
                if isinstance(batch[0][key], np.ndarray):
                    new_batch[key] = torch.cat([torch.from_numpy(data[key]) for data in batch], dim=0)
                else:
                    # for torch.Tensor
                    # # for unitest 
                    # new_batch[key] = torch.stack([data[key] for data in batch], dim=0)
                    new_batch[key] = torch.cat([data[key] for data in batch], dim=0)
            else:
                new_batch[key] = collate_fn([data[key] for data in batch])

        for key in new_batch.keys():
            if "offset" in key:
                new_batch[key] = torch.cumsum(new_batch[key], dim=0)
        return new_batch
    else:
        return default_collate(batch)