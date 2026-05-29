import numpy as np


def get_distributed_items(items, n=16):
    if len(items) <= n:
        return items

    indices = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    distributed_items = [items[i] for i in indices]
    return distributed_items
