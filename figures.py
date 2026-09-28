import os
BASE = os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
a = pd.read_pickle(os.path.join(BASE, "a.pkl")); R = pd.read_pickle(os.path.join(BASE, "primary.pkl"))
out = os.path.join(BASE, "figures") + "/"; os.makedirs(out, exist_ok=True)

# Figure 2: KM by pCR with number at risk
s = a[a.RFS.notna()]; fig, ax = plt.subplots(figsize=(6, 4.6)); kms = []
for g, lab, col in [(1, "pCR", "#1f77b4"), (0, "No pCR", "#d62728")]:
    t = s[s.PCR == g]; km = KaplanMeierFitter(label=lab).fit(t.RFS / 365.25, t.rfs_ind)
    km.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=col, censor_styles={"ms": 4}); kms.append(km)
ax.set_xlim(0, 6); ax.set_ylim(0, 1.02); ax.set_xlabel("Years from start of neoadjuvant chemotherapy"); ax.set_ylabel("Recurrence-free survival")
ax.text(0.2, 0.08, "Log-rank p = 0.016", fontsize=9)
add_at_risk_counts(*kms, ax=ax, rows_to_show=["At risk"], xticks=[0, 1, 2, 3, 4, 5, 6]); plt.tight_layout()
plt.savefig(out + "Fig2_KM_pCR.png", dpi=300); plt.close()

# Figure 3: KM by RCB
fig, ax = plt.subplots(figsize=(6, 5)); kms = []
for g, col in zip([0, 1, 2, 3], ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
    t = s[s.RCBClass == g]; km = KaplanMeierFitter(label=f"RCB {['0','I','II','III'][g]}").fit(t.RFS / 365.25, t.rfs_ind)
    km.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=col, censor_styles={"ms": 4}); kms.append(km)
ax.set_xlim(0, 6); ax.set_ylim(0, 1.02); ax.set_xlabel("Years from start of neoadjuvant chemotherapy"); ax.set_ylabel("Recurrence-free survival")
ax.text(0.2, 0.08, "Log-rank (3 df) p < 0.001", fontsize=9)
add_at_risk_counts(*kms, ax=ax, rows_to_show=["At risk"], xticks=[0, 1, 2, 3, 4, 5, 6]); plt.tight_layout()
plt.savefig(out + "Fig3_KM_RCB.png", dpi=300); plt.close()

# Figure 5: calibration + DCA with bootstrap band
y, P = R["y"], R["P"]; fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
cols = {"R1": "#1f77b4", "R4": "#ff7f0e", "R5": "#2ca02c"}
names = {"R1": "R1 clinical", "R4": "R4 ΔLD T1→T2", "R5": "R5 clinical + LD + ΔLD"}
for m in cols:
    q = pd.qcut(P[m], 5, labels=False, duplicates="drop"); g = pd.DataFrame({"p": P[m], "y": y, "q": q}).groupby("q")
    mp, oy, n = g.p.mean(), g.y.mean(), g.size(); se = np.sqrt(oy * (1 - oy) / n)
    a1.errorbar(mp, oy, yerr=1.96 * se, fmt="o-", color=cols[m], label=names[m], capsize=2, ms=4)
a1.plot([0, 1], [0, 1], "k--", lw=0.8); a1.set_xlim(0, 0.8); a1.set_ylim(0, 0.8)
a1.set_xlabel("Predicted probability of pCR"); a1.set_ylabel("Observed proportion"); a1.set_title("(a) Calibration (quintiles, 95% CI)"); a1.legend(fontsize=8)
th = np.arange(0.05, 0.61, 0.01); rng = np.random.default_rng(1); n = len(y)
def nb(yy, p, t): pos = p >= t; return ((pos & (yy == 1)).sum() - (pos & (yy == 0)).sum() * t / (1 - t)) / len(yy)
idx = [rng.integers(0, n, n) for _ in range(500)]
for m in cols:
    v = np.array([nb(y, P[m], t) for t in th]); b = np.array([[nb(y[i], P[m][i], t) for t in th] for i in idx])
    a2.plot(th, v, color=cols[m], label=names[m]); a2.fill_between(th, np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0), color=cols[m], alpha=0.12)
a2.plot(th, [y.mean() - (1 - y.mean()) * t / (1 - t) for t in th], color="grey", lw=1, label="Treat all as responders")
a2.axhline(0, color="k", lw=0.8, label="Treat none"); a2.set_ylim(-0.05, 0.3); a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit")
a2.set_title("(b) Decision curves (95% bootstrap bands)"); a2.legend(fontsize=8); plt.tight_layout()
plt.savefig(out + "Fig5_calibration_DCA.png", dpi=300); plt.close()
print("ok")
