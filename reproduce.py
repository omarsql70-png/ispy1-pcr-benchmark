import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import os
BASE = os.path.dirname(os.path.abspath(__file__))
X = os.environ.get("ISPY_XLSX", os.path.join(BASE, "I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx"))
c = pd.read_excel(X, "TCIA Patient Clinical Subset")
o = pd.read_excel(X, "TCIA Outcomes Subset")
c.columns = [s.strip() for s in c.columns]; o.columns = [s.strip() for s in o.columns]
d = c.merge(o.drop(columns=["DataExtractDt"]), on="SUBJECTID")
d = d.rename(columns={"MRI LD Baseline": "LD1", "MRI LD 1-3dAC": "LD2", "MRI LD InterReg": "LD3",
                      "MRI LD PreSurg": "LD4", "HR Pos": "HR", "Her2MostPos": "HER2"})
print("rows in workbook:", len(d), " with pCR:", d.PCR.notna().sum())
d["HR"] = d.HR.where(d.HR.isin([0, 1]))
d["HER2"] = d.HER2.where(d.HER2.isin([0, 1]))
for k in (2, 3, 4):
    d[f"dLD{k}"] = 100 * (d[f"LD{k}"] - d.LD1) / d.LD1
d.loc[d.LD1 == 0, ["dLD2", "dLD3", "dLD4"]] = np.nan
a = d[d.PCR.notna()].copy(); a["PCR"] = a.PCR.astype(int)
a["sub"] = a.HR_HER2_CATEGORY  # 1 HR+/HER2-, 2 HER2+, 3 TN
p1, p0 = a[a.PCR == 1], a[a.PCR == 0]
print("\n=== Table 1 ===  N", len(a), "pCR", len(p1), f"({100*len(p1)/len(a):.1f}%)")
for s, n in [(1, "HR+/HER2-"), (2, "HER2+"), (3, "TN")]:
    t = a[a["sub"] == s]; print(n, len(t), t.PCR.sum(), f"{100*t.PCR.mean():.1f}%")
print("unknown subtype", a["sub"].isna().sum(), "pCR", a[a["sub"].isna()].PCR.sum())
ct = pd.crosstab(a["sub"], a.PCR); print("chi2 p", stats.chi2_contingency(ct)[1])
def mw(v):
    x1, x0 = p1[v].dropna(), p0[v].dropna()
    q = lambda x: f"{x.median():.1f} ({x.quantile(.25):.1f}-{x.quantile(.75):.1f})"
    return q(x1), q(x0), round(stats.mannwhitneyu(x1, x0).pvalue, 3)
print("age mean", round(p1.age.mean(),1), round(p0.age.mean(),1), "sd", round(p1.age.std(),1), round(p0.age.std(),1),
      "p", round(stats.mannwhitneyu(p1.age, p0.age).pvalue, 3))
for v in ["LD1", "LD2", "dLD2"]: print(v, mw(v))

print("\n=== Survival ===")
s = a[a.RFS.notna()]
e1, e0 = s[s.PCR == 1], s[s.PCR == 0]
print("events", e1.rfs_ind.sum(), len(e1), e0.rfs_ind.sum(), len(e0))
lr = logrank_test(e1.RFS, e0.RFS, e1.rfs_ind, e0.rfs_ind); print("logrank chi2", round(lr.test_statistic, 2), "p", round(lr.p_value, 3))
for g, n in [(e1, "pCR"), (e0, "non")]:
    km = KaplanMeierFitter().fit(g.RFS / 365.25, g.rfs_ind)
    sv = km.survival_function_at_times(5).iloc[0]; ci = km.confidence_interval_
    print(n, "5y RFS", round(100 * sv, 1))
cx = CoxPHFitter().fit(s[["RFS", "rfs_ind", "PCR"]], "RFS", "rfs_ind")
print("Cox unadj HR", np.round(cx.summary.loc["PCR", ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].values, 3))
sa = s[s["sub"].notna()].copy(); sa["HER2p"] = (sa["sub"] == 2).astype(int); sa["TN"] = (sa["sub"] == 3).astype(int)
cx2 = CoxPHFitter().fit(sa[["RFS", "rfs_ind", "PCR", "HER2p", "TN", "age"]], "RFS", "rfs_ind")
print("Cox adj n", len(sa), "events", sa.rfs_ind.sum(), "HR", np.round(cx2.summary.loc["PCR", ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].values, 3))
r = s[s.RCBClass.notna()]
print("RCB n", r.groupby("RCBClass").size().to_dict(), "event%", (100 * r.groupby("RCBClass").rfs_ind.mean()).round(1).to_dict())
m = multivariate_logrank_test(r.RFS, r.RCBClass, r.rfs_ind); print("RCB logrank chi2", round(m.test_statistic, 2), "p", m.p_value)
print("pCR vs RCB0 crosstab\n", pd.crosstab(a.PCR, a.RCBClass, dropna=False))

print("\n=== Logistic ===")
L = a.copy(); L["age10"] = L.age / 10; L["LD1_10"] = L.LD1 / 10; L["dLD2_10"] = L.dLD2 / 10
L["HER2p"] = (L["sub"] == 2).astype(float); L["TN"] = (L["sub"] == 3).astype(float)
cc = L.dropna(subset=["sub", "age10", "LD1_10", "dLD2_10"])
print("complete cases", len(cc), "pCR", cc.PCR.sum())
cov = ["age10", "HER2p", "TN", "LD1_10", "dLD2_10"]
def orr(res, v): return f"{np.exp(res.params[v]):.2f} ({np.exp(res.conf_int().loc[v,0]):.2f}-{np.exp(res.conf_int().loc[v,1]):.2f}) p={res.pvalues[v]:.3f}"
for v in cov:
    cols = ["HER2p", "TN"] if v in ("HER2p", "TN") else [v]
    r1 = sm.Logit(cc.PCR, sm.add_constant(cc[cols])).fit(disp=0); print("uni", v, orr(r1, v))
mv = sm.Logit(cc.PCR, sm.add_constant(cc[cov])).fit(disp=0)
for v in cov: print("multi", v, orr(mv, v))

print("\n=== Diameter AUCs ===")
rng = np.random.default_rng(42)
def bauc(y, sc, B=2000):
    y, sc = np.asarray(y), np.asarray(sc); b = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): b.append(roc_auc_score(y[i], sc[i]))
    return roc_auc_score(y, sc), np.percentile(b, 2.5), np.percentile(b, 97.5)
for v in ["LD1", "LD2", "LD3", "LD4", "dLD2", "dLD3", "dLD4"]:
    t = a[[v, "PCR"]].dropna()
    au = bauc(t.PCR, -t[v]); print(v, len(t), t.PCR.sum(), "medians", t[t.PCR==1][v].median().round(1), t[t.PCR==0][v].median().round(1), "AUC %.3f (%.3f-%.3f)" % au)

def delong(y, s1, s2):
    y = np.asarray(y); pos, neg = y == 1, y == 0
    def comp(s):
        s = np.asarray(s); X, Y = s[pos], s[neg]
        V = (X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])
        return V.mean(), V.mean(1), V.mean(0)
    a1, v10, v01 = comp(s1); a2, w10, w01 = comp(s2)
    S10 = np.cov(np.vstack([v10, w10])); S01 = np.cov(np.vstack([v01, w01]))
    var = (S10[0,0]+S10[1,1]-2*S10[0,1])/pos.sum() + (S01[0,0]+S01[1,1]-2*S01[0,1])/neg.sum()
    z = (a1 - a2) / np.sqrt(var); return a1 - a2, 2 * stats.norm.sf(abs(z))
t = a[["dLD2", "dLD4", "PCR"]].dropna(); print("DeLong T4 vs T2 n", len(t), np.round(delong(t.PCR, -t.dLD4, -t.dLD2), 3))

print("\n=== Radiological response ===")
def cat(v):
    return np.where(v <= -100 + 1e-9, "CR", np.where(v <= -30, "PR", np.where(v >= 20, "PD", "SD")))
for k in (2, 4):
    t = a[[f"LD{k}", f"dLD{k}", "PCR"]].dropna().copy()
    t["c"] = np.where(t[f"LD{k}"] == 0, "CR", cat(t[f"dLD{k}"]))
    print(f"T{k}", t.groupby("c").PCR.agg(["size", "sum"]).to_dict())
    pos = t.c.isin(["CR"] if k == 4 else ["CR", "PR"])
    tp = (pos & (t.PCR == 1)).sum(); fp = (pos & (t.PCR == 0)).sum(); fn = (~pos & (t.PCR == 1)).sum(); tn = (~pos & (t.PCR == 0)).sum()
    print("  sens %.3f spec %.3f ppv %.3f npv %.3f" % (tp/(tp+fn), tn/(tn+fp), tp/(tp+fp), tn/(tn+fn)))

print("\n=== Subtype dLD2 AUC ===")
for sb in (1, 2, 3):
    t = a[a["sub"] == sb][["dLD2", "PCR"]].dropna()
    print(sb, len(a[a["sub"]==sb]), "AUC %.3f (%.3f-%.3f)" % bauc(t.PCR, -t.dLD2), t[t.PCR==1].dLD2.median().round(1), t[t.PCR==0].dLD2.median().round(1))

print("\n=== CV reference models ===")
M = {"R1": ["age", "HR", "HER2"], "R2": ["LD1"], "R3": ["age", "HR", "HER2", "LD1"], "R4": ["dLD2"], "R5": ["age", "HR", "HER2", "LD1", "dLD2"]}
y = a.PCR.values; strat = a.PCR.astype(str) + "_" + a["sub"].fillna(0).astype(int).astype(str)
oof = {m: np.zeros((3, len(a))) for m in M}; res = {m: [] for m in M}
for rep, seed in enumerate([0, 1, 2]):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(a, strat):
        for m, f in M.items():
            Xtr, Xte = a.iloc[tr][f].astype(float), a.iloc[te][f].astype(float)
            med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
            itr, ival = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y[tr], random_state=seed)
            sc = StandardScaler().fit(Xtr.iloc[itr]); lr_ = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr.iloc[itr]), y[tr][itr])
            fpr, tpr, th = roc_curve(y[tr][ival], lr_.predict_proba(sc.transform(Xtr.iloc[ival]))[:, 1]); thr = th[np.argmax(tpr - fpr)]
            sc = StandardScaler().fit(Xtr); lr_ = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), y[tr])
            p = lr_.predict_proba(sc.transform(Xte))[:, 1]; oof[m][rep, te] = p
            pr = p >= thr; yt = y[te]
            res[m].append([(pr & (yt==1)).sum()/(yt==1).sum(), (~pr & (yt==0)).sum()/(yt==0).sum(), roc_auc_score(yt, p), average_precision_score(yt, p)])
P = {m: oof[m].mean(0) for m in M}
for m in M:
    r_ = np.array(res[m]); pa = bauc(y, P[m])
    print(m, "sens %.3f±%.3f spec %.3f±%.3f AUC %.3f±%.3f PR %.3f±%.3f" % (r_[:,0].mean(), r_[:,0].std(ddof=1), r_[:,1].mean(), r_[:,1].std(ddof=1), r_[:,2].mean(), r_[:,2].std(ddof=1), r_[:,3].mean(), r_[:,3].std(ddof=1)), "pooled %.3f (%.3f-%.3f)" % pa)
for x1, x2 in [("R5", "R1"), ("R5", "R3"), ("R5", "R2"), ("R5", "R4")]:
    print(x1, "vs", x2, np.round(delong(y, P[x1], P[x2]), 4))

print("\n=== Calibration / DCA ===")
for m in M:
    p = np.clip(P[m], 1e-6, 1-1e-6); lp = np.log(p/(1-p))
    sl = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    ic = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    print(m, "Brier %.3f int %.3f slope %.3f" % (np.mean((p-y)**2), ic, sl))
def nb(p, t): pos = p >= t; n = len(y); return ((pos & (y==1)).sum() - (pos & (y==0)).sum()*t/(1-t))/n
for t in (0.2, 0.3):
    print(t, {m: round(nb(P[m], t), 3) for m in ["R1", "R4", "R5"]}, "all", round(y.mean() - (1-y.mean())*t/(1-t), 3))
a.to_pickle(os.path.join(BASE, "a.pkl"))
