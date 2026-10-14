"""Central paths and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MFEAT_DIR = DATA_DIR / "mfeat"
CLASSIFICATION_CSV = DATA_DIR / "train.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "price_range"
PRICE_LABELS = {0: "Low cost", 1: "Medium cost", 2: "High cost", 3: "Very high cost"}

# Order matters: rows line up across files, and we concatenate in this order.
MFEAT_SETS = ["fou", "fac", "kar", "pix", "zer", "mor"]
MFEAT_DESC = {
    "fou": "Fourier coefficients (76)",
    "fac": "Profile correlations (216)",
    "kar": "Karhunen-Loeve coefficients (64)",
    "pix": "Pixel averages 2x3 windows (240)",
    "zer": "Zernike moments (47)",
    "mor": "Morphological features (6)",
}
K_RANGE = range(2, 16)          # range searched by the elbow method
# The inertia curve is smooth (no sharp elbow). The marginal drop in inertia
# falls from 3.2% (k 8->9) to 2.6% (k 9->10) and stays below ~2.7% afterwards,
# and the dataset documentation states 10 digit classes (domain knowledge, not
# label use). Set to None to use the automatic knee instead.
K_FINAL = 10
K_SENSITIVITY = [6, 9, 10]      # automatic knee, best silhouette, final
LINKAGES = ["ward", "complete", "average", "single"]

for d in (MODELS_DIR, REPORTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
