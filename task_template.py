import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.stats import norm as scipy_norm

# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "120b00f66fa0d99ed1040e67d025da15"
TASK_ID = "01-mia"  #DONOT CHANGE


# Use all CPU cores for maximum speed
torch.set_num_threads(torch.get_num_threads())
device = torch.device("cpu")
print(f"Using device: cpu with {torch.get_num_threads()} threads")

# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]
    

# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

def safe_collate(batch):
    ids    = torch.stack([torch.tensor(b[0]) if not isinstance(b[0], torch.Tensor) else b[0] for b in batch])
    imgs   = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch])
    return ids, imgs, labels

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform

pub_membership  = np.array(pub_ds.membership)
pub_labels_arr  = np.array(pub_ds.labels)
priv_labels_arr = np.array(priv_ds.labels)

print(f"pub.pt:  {len(pub_ds)} samples ({pub_membership.sum()} members, {(pub_membership==0).sum()} non-members)")
print(f"priv.pt: {len(priv_ds)} samples")

# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# create random submission (remove this later or it will rewrite your actual submission)
'''print("Creating random submission...")
ids = [str(i) for i in priv_ds.ids]

df = pd.DataFrame({
    "id": ids,
    "score": [random.random() for _ in ids]
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)'''

# ── Extract signals in ONE pass over both datasets
def extract_signals(dataset):
    loader = DataLoader(dataset, batch_size=2048, shuffle=False,
                        num_workers=0, collate_fn=safe_collate)
    all_ids, all_loss, all_margin, all_logit_gap = [], [], [], []

    with torch.no_grad():
        for ids, imgs, labels in loader:
            logits = model(imgs)

            # Loss on true label — strongest MIA signal
            loss = F.cross_entropy(logits, labels, reduction='none')

            # Margin: top1 - top2 logit
            top2   = torch.topk(logits, 2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]

            # Correct logit vs mean of others
            correct_logit = logits[torch.arange(len(labels)), labels]
            mean_other    = (logits.sum(dim=1) - correct_logit) / (logits.shape[1] - 1)
            logit_gap     = correct_logit - mean_other

            all_ids.extend(ids.tolist())
            all_loss.extend(loss.tolist())
            all_margin.extend(margin.tolist())
            all_logit_gap.extend(logit_gap.tolist())

    return {
        'ids':       all_ids,
        'loss':      np.array(all_loss),
        'margin':    np.array(all_margin),
        'logit_gap': np.array(all_logit_gap),
    }

print("Extracting signals from pub.pt... (30-60 sec)")
pub_sig  = extract_signals(pub_ds)
print("Extracting signals from priv.pt... (30-60 sec)")
priv_sig = extract_signals(priv_ds)
print("Done extracting.")

# ── Per-class Gaussian LiRA 
def gaussian_lira(query_vals, query_labels,
                  ref_vals,   ref_labels, ref_membership, n_classes=9):
    scores = np.zeros(len(query_vals))
    for c in range(n_classes):
        q_mask  = query_labels == c
        m_mask  = (ref_labels  == c) & (ref_membership == 1)
        nm_mask = (ref_labels  == c) & (ref_membership == 0)

        if q_mask.sum() == 0:
            continue
        if m_mask.sum() < 3 or nm_mask.sum() < 3:
            scores[q_mask] = -query_vals[q_mask]
            continue

        mu_in,  std_in  = ref_vals[m_mask].mean(),  ref_vals[m_mask].std()  + 1e-4
        mu_out, std_out = ref_vals[nm_mask].mean(), ref_vals[nm_mask].std() + 1e-4

        log_p_in  = scipy_norm.logpdf(query_vals[q_mask], mu_in,  std_in)
        log_p_out = scipy_norm.logpdf(query_vals[q_mask], mu_out, std_out)
        scores[q_mask] = log_p_in - log_p_out
    return scores

def tpr_at_fpr(scores, labels, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[max(0, idx - 1)])

print("Computing LiRA scores...")

# ── Validate on pub.pt 
pub_s1 = gaussian_lira(pub_sig['loss'],       pub_labels_arr, pub_sig['loss'],       pub_labels_arr, pub_membership)
pub_s2 = gaussian_lira(-pub_sig['margin'],    pub_labels_arr, -pub_sig['margin'],    pub_labels_arr, pub_membership)
pub_s3 = gaussian_lira(-pub_sig['logit_gap'], pub_labels_arr, -pub_sig['logit_gap'], pub_labels_arr, pub_membership)

t1 = tpr_at_fpr(pub_s1, pub_membership)
t2 = tpr_at_fpr(pub_s2, pub_membership)
t3 = tpr_at_fpr(pub_s3, pub_membership)

print(f"\n── Signal quality (TPR@5%FPR on pub.pt) ──")
print(f"  LiRA loss:      {t1:.4f}")
print(f"  LiRA margin:    {t2:.4f}")
print(f"  LiRA logit_gap: {t3:.4f}")

# ── Score priv.pt 
priv_s1 = gaussian_lira(priv_sig['loss'],       priv_labels_arr, pub_sig['loss'],       pub_labels_arr, pub_membership)
priv_s2 = gaussian_lira(-priv_sig['margin'],    priv_labels_arr, -pub_sig['margin'],    pub_labels_arr, pub_membership)
priv_s3 = gaussian_lira(-priv_sig['logit_gap'], priv_labels_arr, -pub_sig['logit_gap'], pub_labels_arr, pub_membership)

n = len(priv_ds)
r1 = priv_s1.argsort().argsort() / (n - 1)
r2 = priv_s2.argsort().argsort() / (n - 1)
r3 = priv_s3.argsort().argsort() / (n - 1)

#final_scores = 0.6 * r1 + 0.2 * r2 + 0.2 * r3
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

priv_s1 = normalize(priv_s1)
priv_s3 = normalize(priv_s3)
final_scores = 0.3 * r1 + 0.7 * r3
final_scores = final_scores.argsort().argsort() / (n - 1)

# Estimate on pub
np_ = len(pub_ds)
pr1 = pub_s1.argsort().argsort() / (np_ - 1)
pr2 = pub_s2.argsort().argsort() / (np_ - 1)
pr3 = pub_s3.argsort().argsort() / (np_ - 1)
pub_combo = 0.6 * pr1 + 0.2 * pr2 + 0.2 * pr3
print(f"  Combined:       {tpr_at_fpr(pub_combo, pub_membership):.4f}  ← expected leaderboard score")

# ── Save 
df = pd.DataFrame({"id": priv_sig['ids'], "score": final_scores})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")


# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)