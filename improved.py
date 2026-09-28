import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import os, argparse
BASE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="I-SPY 1 pCR benchmark: survival analyses and cross-validated reference models R1-R5")
ap.add_argument("--ids", help="text/CSV file with one SUBJECTID per line; the reference models are recomputed on this subset only")
ap.add_argument("--reps", type=int, default=50, help="number of repetitions of 5-fold cross-validation (default 50)")
ap.add_argument("--out", default=BASE, help="folder for exported fold assignments and out-of-fold predictions")
args = ap.parse_args()
a = pd.read_pickle(os.path.join(BASE, "a.pkl"))
out = {}

# ---------- 1. Survival: correct unadjusted HR + landmark analyses ----------
s = a[a.RFS.notna()].copy()
def cox(df, cov):
    c = CoxPHFitter().fit(df[["RFS", "rfs_ind"] + cov], "RFS", "rfs_ind"); r = c.summary.loc["PCR"]
    return f"HR {r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}), p={r.p:.3f}"
s["HER2p"] = (s["sub"] == 2).astype(int); s["TN"] = (s["sub"] == 3).astype(int)
print("Unadjusted, n=215:", cox(s, ["PCR"]), "events", s.rfs_ind.sum())
s210 = s[s["sub"].notna()]
print("Unadjusted, n=210:", cox(s210, ["PCR"]), "events", s210.rfs_ind.sum())
print("Adjusted, n=210:", cox(s210, ["PCR", "HER2p", "TN", "age"]))
print("PH test (adjusted):")
cx = CoxPHFitter().fit(s210[["RFS", "rfs_ind", "PCR", "HER2p", "TN", "age"]], "RFS", "rfs_ind")
from lifelines.statistics import proportional_hazard_test
print(proportional_hazard_test(cx, s210[["RFS", "rfs_ind", "PCR", "HER2p", "TN", "age"]], time_transform="rank").summary[["p"]].round(3).T)
for L in (180, 270, 365):
    t = s210[s210.RFS > L].copy(); t["RFS"] = t.RFS - L
    ev_before = ((s210.RFS <= L) & (s210.rfs_ind == 1)).sum()
    print(f"Landmark {L} d: n={len(t)}, events={t.rfs_ind.sum()} (excluded events before landmark={ev_before});",
          "unadj", cox(t, ["PCR"]), "| adj", cox(t, ["PCR", "HER2p", "TN", "age"]))
print("Early events by pCR (RFS<=270d):", s[(s.RFS <= 270) & (s.rfs_ind == 1)].PCR.value_counts().to_dict())


cxs = CoxPHFitter().fit(s210[["RFS", "rfs_ind", "PCR", "age", "sub"]], "RFS", "rfs_ind", strata=["sub"]); r = cxs.summary.loc["PCR"]
print(f"Stratified by subtype (adj. age): HR {r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}), p={r.p:.3f}")
for k, nm in [(1, "HR+/HER2-"), (2, "HER2+"), (3, "TN")]:
    t = s210[s210["sub"] == k]
    print(nm, "n", len(t), "events", t.rfs_ind.sum(), "events in pCR", t[t.PCR == 1].rfs_ind.sum(), "|", cox(t, ["PCR"]) if t[t.PCR == 1].rfs_ind.sum() > 0 else "not estimable")
from lifelines import KaplanMeierFitter
print("Median follow-up (reverse KM), years:", round(KaplanMeierFitter().fit(s.RFS / 365.25, 1 - s.rfs_ind).median_survival_time_, 1))

# ---------- 2. Nested-model likelihood-ratio tests (in-sample, complete cases) ----------
L_ = a[a["sub"].notna()].dropna(subset=["LD1", "dLD2", "HR", "HER2"]).copy()
def ll(cols): return sm.Logit(L_.PCR, sm.add_constant(L_[cols].astype(float))).fit(disp=0)
base = ["age", "HR", "HER2"]
for b0, extra in ((base, ["LD1"]), (base, ["dLD2"]), (base, ["LD1", "dLD2"]), (base + ["LD1"], ["dLD2"])):
    m0, m1 = ll(b0), ll(b0 + extra); lr = 2 * (m1.llf - m0.llf); df = len(extra)
    print(f"LRT {b0} + {extra}: n={len(L_)}, chi2={lr:.2f}, df={df}, p={stats.chi2.sf(lr, df):.3f}")

# ---------- 3. Repeated CV (50 x 5-fold), no Youden threshold ----------
M = {"R1": ["age", "HR", "HER2"], "R2": ["LD1"], "R3": ["age", "HR", "HER2", "LD1"], "R4": ["dLD2"], "R5": ["age", "HR", "HER2", "LD1", "dLD2"]}
def run_cv(df, reps=None):
    reps = reps or args.reps
    y = df.PCR.values; strat = df.PCR.astype(str) + "_" + df["sub"].fillna(0).astype(int).astype(str)
    oof = {m: np.zeros((reps, len(df))) for m in M}; fold_auc = {m: [] for m in M}; folds = np.zeros((reps, len(df)), dtype=int)
    for r in range(reps):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(df, strat)):
            folds[r, te] = k
            for m, f in M.items():
                Xtr, Xte = df.iloc[tr][f].astype(float), df.iloc[te][f].astype(float)
                med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
                sc = StandardScaler().fit(Xtr); lr_ = LogisticRegression(C=1e6, max_iter=2000).fit(sc.transform(Xtr), y[tr])
                p = lr_.predict_proba(sc.transform(Xte))[:, 1]; oof[m][r, te] = p; fold_auc[m].append(roc_auc_score(y[te], p))
    run_cv.last_folds = folds
    return y, {m: oof[m].mean(0) for m in M}, fold_auc

def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    sl = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    ic = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return np.mean((p - y) ** 2), ic, sl
def nb(y, p, t): pos = p >= t; return ((pos & (y == 1)).sum() - (pos & (y == 0)).sum() * t / (1 - t)) / len(y)

def report(df, label, B=2000, seed=7):
    y, P, fa = run_cv(df)
    rng = np.random.default_rng(seed); n = len(y)
    idx = [i for i in (rng.integers(0, n, n) for _ in range(B)) if y[i].min() != y[i].max()]
    rows = []
    for m in M:
        au = roc_auc_score(y, P[m]); bs = [roc_auc_score(y[i], P[m][i]) for i in idx]
        br, ic, sl = calib(y, P[m]); bsl = []
        for i in idx[:500]: bsl.append(calib(y[i], P[m][i])[2])
        rows.append([m, f"{np.mean(fa[m]):.3f} ± {np.std(fa[m], ddof=1):.3f}", f"{au:.3f} ({np.percentile(bs,2.5):.3f}-{np.percentile(bs,97.5):.3f})",
                     f"{average_precision_score(y, P[m]):.3f}", f"{br:.3f}", f"{ic:.3f}", f"{sl:.2f} ({np.percentile(bsl,2.5):.2f}-{np.percentile(bsl,97.5):.2f})"])
    T = pd.DataFrame(rows, columns=["Model", f"Fold AUC mean±SD ({args.reps*5} folds)", "Pooled AUC (95% CI)", "PR-AUC", "Brier", "Intercept", "Slope (95% CI)"])
    print(f"\n=== {label}: n={n}, pCR={y.sum()} ===\n" + T.to_string(index=False))
    comp = []
    for x1, x2 in [("R3", "R1"), ("R5", "R1"), ("R5", "R3"), ("R5", "R2"), ("R5", "R4")]:
        d0 = roc_auc_score(y, P[x1]) - roc_auc_score(y, P[x2]); bd = [roc_auc_score(y[i], P[x1][i]) - roc_auc_score(y[i], P[x2][i]) for i in idx]
        comp.append([f"{x1} vs {x2}", d0, np.percentile(bd, 2.5), np.percentile(bd, 97.5), 2 * min(np.mean(np.array(bd) <= 0), np.mean(np.array(bd) >= 0))])
    C = pd.DataFrame(comp, columns=["Comparison", "dAUC", "lo", "hi", "p_boot"])
    C["p_Holm"] = np.minimum(1, np.maximum.accumulate((C.p_boot.sort_values() * np.arange(len(C), 0, -1)).values))[np.argsort(np.argsort(C.p_boot.values))]
    print(C.round(3).to_string(index=False))
    for t in (0.15, 0.2, 0.3, 0.4):
        d = [nb(y[i], P["R5"][i], t) - nb(y[i], P["R1"][i], t) for i in idx]
        print(f"NB t={t}: R1 {nb(y,P['R1'],t):.3f}  R5 {nb(y,P['R5'],t):.3f}  treat-all {y.mean()-(1-y.mean())*t/(1-t):.3f}  "
              f"R5-R1 {nb(y,P['R5'],t)-nb(y,P['R1'],t):+.3f} ({np.percentile(d,2.5):+.3f} to {np.percentile(d,97.5):+.3f})")
    return y, P, T, C

main = a[a["sub"].notna()].copy()                       # primary: known subtype (n=210)
tag = "primary"
if args.ids:
    ids = set(int(x) for x in pd.read_csv(args.ids, header=None, comment="#").iloc[:, 0])
    missing = ids - set(main.SUBJECTID)
    main = main[main.SUBJECTID.isin(ids)].copy(); tag = "subset"
    print(f"\nSubset from {args.ids}: {len(ids)} IDs supplied, {len(main)} analysable (pCR recorded, known subtype); {len(missing)} not analysable")
y, P, T, C = report(main.reset_index(drop=True), "PRIMARY (known subtype)" if tag == "primary" else "USER-SUPPLIED SUBSET")
fo = run_cv.last_folds
exp = pd.DataFrame({"SUBJECTID": np.repeat(main.SUBJECTID.values[None, :], fo.shape[0], 0).ravel(),
                    "repetition": np.repeat(np.arange(fo.shape[0]), fo.shape[1]), "fold": fo.ravel()})
exp.to_csv(os.path.join(args.out, f"folds_{tag}.csv"), index=False)
pd.DataFrame({"SUBJECTID": main.SUBJECTID.values, "pCR": y, **{f"pred_{m}": P[m] for m in P}}).to_csv(os.path.join(args.out, f"oof_predictions_{tag}.csv"), index=False)
print(f"Exported folds_{tag}.csv (seed = repetition index; 5-fold stratified on pCR x subtype) and oof_predictions_{tag}.csv")
if tag == "primary":
    pd.to_pickle({"y": y, "P": P, "T": T, "C": C}, os.path.join(BASE, "primary.pkl"))
    report(a, "SENSITIVITY (all 215, median-imputed receptors)")
    sub = main.dropna(subset=["LD1", "LD2"])
    report(sub, "SENSITIVITY (complete T1+T2 MRI diameter)")

# ---------- 4. Final model specification (TRIPOD+AI item 22) and subgroup performance by race (item 23a) ----------
if tag == "primary":
    print("\n=== Final models fitted on the primary set (original scale; median imputation of missing diameters) ===")
    for name in ("R1", "R5"):
        Xf = main[M[name]].astype(float); Xf = Xf.fillna(Xf.median())
        fit = sm.Logit(main.PCR.values, sm.add_constant(Xf)).fit(disp=0)
        print(name, {k: round(v, 4) for k, v in fit.params.items()}, "| imputation medians:", Xf.median().round(2).to_dict())
    race = main.race_id.map({1: "White", 3: "African American"}).fillna("Other or unknown").values
    rng2 = np.random.default_rng(3)
    print("\n=== Cross-validated AUC by race (primary set) ===")
    for g in ("White", "African American"):
        k = race == g
        for m in ("R1", "R5"):
            yy, pp = y[k], P[m][k]; bs = []
            for _ in range(2000):
                i = rng2.integers(0, len(yy), len(yy))
                if yy[i].min() != yy[i].max(): bs.append(roc_auc_score(yy[i], pp[i]))
            print(f"{g} {m}: n={k.sum()}, pCR={yy.sum()}, AUC {roc_auc_score(yy, pp):.3f} ({np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f})")
