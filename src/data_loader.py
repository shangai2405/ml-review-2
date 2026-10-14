"""Loading and preprocessing for both datasets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


# ---------------------------------------------------------------- classification
def load_classification():
    """Return the Mobile Price dataframe (2000 x 21)."""
    return pd.read_csv(C.CLASSIFICATION_CSV)


def split_classification(df=None):
    """Stratified 80/20 split - identical seed to Review 1 so all 10 models share it."""
    df = load_classification() if df is None else df
    X = df.drop(columns=[C.TARGET_COL])
    y = df[C.TARGET_COL]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y,
                            random_state=C.RANDOM_STATE)


# ---------------------------------------------------------------- clustering
def load_mfeat(sets=None):
    """Load and horizontally stack the chosen mfeat feature files.

    Returns
    -------
    X : ndarray (2000, n_features)
    y : ndarray (2000,) digit labels 0-9, derived from row order
        (rows 0-199 are '0', 200-399 are '1', ...). Used ONLY for post-hoc
        validation, never passed to a clustering fit.
    feature_names : list[str]
    """
    sets = sets or C.MFEAT_SETS
    blocks, names = [], []
    for s in sets:
        arr = np.loadtxt(C.MFEAT_DIR / f"mfeat-{s}")
        blocks.append(arr)
        names += [f"{s}_{i}" for i in range(arr.shape[1])]
    X = np.hstack(blocks)
    y = np.repeat(np.arange(10), 200)
    return X, y, names


def pixel_image(row_idx, pix=None):
    """Rebuild the 15 x 16 digit image from mfeat-pix (values 0-6)."""
    pix = np.loadtxt(C.MFEAT_DIR / "mfeat-pix") if pix is None else pix
    return pix[row_idx].reshape(16, 15)
