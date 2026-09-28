import os
BASE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score
plt.rcParams.update({"font.family": "DejaVu Sans"})
out = os.path.join(BASE, "figures") + "/"; os.makedirs(out, exist_ok=True)
a = pd.read_pickle(os.path.join(BASE, "a.pkl"))

# ---- Figure 1: flow ----
fig, ax = plt.subplots(figsize=(9, 7)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, title, sub, hi=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                 fc="#e8f0fb" if hi else "white", ec="#1f5fbf" if hi else "#555", lw=1.6 if hi else 1.1))
    ax.text(x + w / 2, y + h - 3, title, ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + (h - 5) / 2, sub, ha="center", va="center", fontsize=8, linespacing=1.35)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1))
L = lambda xs, ys: ax.plot(xs, ys, color="#555", lw=1.1)
box(35, 90, 30, 8, "221 patients", "I-SPY 1 clinical workbook")
box(71, 79, 27, 8, "6 excluded", "no surgery, pCR unavailable")
box(35, 68, 30, 8, "215 analysed", "58 pCR, 157 non-pCR", hi=True)
arr(50, 90, 50, 76.5); L([50, 70.5], [83, 83]); arr(70, 83, 70.7, 83)
L([50, 50], [68, 62]); L([16, 84], [62, 62])
for x in (16, 50, 84): arr(x, 62, x, 57.3)
box(2, 40, 28, 17, "Survival", "unadjusted: 215 (59 events)\nadjusted / stratified:\n210 (57 events)")
box(36, 40, 28, 17, "Logistic regression", "201 complete cases (54 pCR)\nlikelihood-ratio tests")
box(70, 40, 28, 17, "Reference models", "primary: 210 (known subtype)\nsensitivity: 215 and 201\n50 × 5-fold CV")
L([33, 33], [62, 30]); L([16, 84], [30, 30])
for x in (16, 50, 84): arr(x, 30, x, 25.3)
box(2, 8, 28, 17, "Landmark survival", "180 days: 207 (54 events)\n270 days: 203 (50 events)")
box(36, 8, 28, 17, "Diameter by timepoint", "T1: 213    T2: 204\nT3: 196    T4: 204")
box(70, 8, 28, 17, "RCB analysis", "201 with RCB class\n54 events")
plt.savefig(out + "Fig1_flow.png", dpi=300, bbox_inches="tight"); plt.close()

# ---- Figure 4: AUC by timepoint ----
rng = np.random.default_rng(42)
def bauc(y, s, B=2000):
    y, s = np.asarray(y), np.asarray(s); b = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): b.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(b, 2.5), np.percentile(b, 97.5)
fig, ax = plt.subplots(figsize=(6.4, 4.2)); labs = ["T1\nbaseline", "T2\nafter AC cycle 1", "T3\ninter-regimen", "T4\npre-surgery"]
for vs, col, name, off in [(["LD1", "LD2", "LD3", "LD4"], "#1f77b4", "LD (mm)", -0.06), ([None, "dLD2", "dLD3", "dLD4"], "#d62728", "ΔLD from baseline (%)", 0.06)]:
    xs, ys, lo, hi = [], [], [], []
    for k, v in enumerate(vs):
        if v is None: continue
        t = a[[v, "PCR"]].dropna(); r = bauc(t.PCR, -t[v]); xs.append(k + off); ys.append(r[0]); lo.append(r[0] - r[1]); hi.append(r[2] - r[0])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=name)
ax.axhline(0.5, color="grey", lw=0.8, ls=":"); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylim(0.35, 0.9); ax.set_ylabel("AUC for pCR (95% bootstrap CI)"); ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout(); plt.savefig(out + "Fig4_AUC_timepoints.png", dpi=300); plt.close(); print("ok")
