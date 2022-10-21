"""
Get the pred probabilities from the saved pred pickles
"""

from src.utils import get_pickle
from pathlib import Path
from decouple import config
import numpy as np
import pandas as pd
from torchmetrics import AUROC, Accuracy

run_name = "likely-lake-279"
MODEL_DIR = Path(config("MODEL_DIR"))

# Load predictions
preds_fname = MODEL_DIR / f"{run_name}_test-preds.pkl"
preds = get_pickle(preds_fname)
pos_preds = preds[:, 1]

# Convert to desired formats and save
df = pd.DataFrame(pos_preds, columns=["prediction"])
df.index.name = "case"
df.to_csv(
    MODEL_DIR / f"{run_name}_pos-test-preds.csv", sep=",", float_format="%.6f"
)

# Print some stats
targets_fname = MODEL_DIR / f"{run_name}_test-targets.pkl"
targets = get_pickle(targets_fname)
calc_auc = AUROC(num_classes=2, pos_label=1)
calc_acc = Accuracy()

auc = calc_auc(preds, targets)
acc = calc_acc(preds, targets)
print(f"This run achieved an AUC of {auc:.4f} and an accuracy of {acc:.4f}")
