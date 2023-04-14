from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def augmentation_random_cut(X, y, n_random_cut=8, n_cut=0, seed=None, apply=True):
    if n_random_cut % 2 == 1 or n_cut % 2 == 1:
        raise ValueError('n_random_cut and n_cut must be even')
    if not apply:
        return X, y
    random_state = np.random.RandomState(seed)

    start = random_state.randint(low=0, high=n_random_cut, size=X.shape[0]) + n_cut // 2
    end = X.shape[1] - n_random_cut - n_cut + start

    X_new = np.empty(shape=(X.shape[0], X.shape[1] - n_random_cut - n_cut))
    for i in range(X.shape[0]):
        X_new[i] = X[i, start[i]:end[i]]

    y_new = y - start
    return X_new, y_new


def save_plt(path: Path | str, **kwargs) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parents[0].mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)
