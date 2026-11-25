#!/usr/bin/env python3
"""
Alchemy 2026 Case 4 - Working model for PU lacquer scratch hardness (Kratzhaerte).

Data: PURformance dataset, Zenodo record 10.5281/zenodo.13742097 (v13742098), CC-BY-4.0.
Paper: DOI 10.1016/j.porgcoat.2025.109265.
Source file: Experiment.zip -> Experiment/Iteration_4/PURformance_dataset.xlsx (sheet doe_actual).

Data audit (performed before training, see README):
- The 55-row sheet is strictly chronological: rows 0-9 cold start, 10-16 iter1,
  17-26 iter2, 27-41 iter3, 42-54 random-test formulations with NO measured target.
- Rows 17/18 are one formulation measured twice (16.765 / 13.994).
- Only the 42 labeled rows are ever used as training/test labels. No invented labels.

Evaluation design (retrospective temporal backtest):
- Split A (train cold start 10 -> test iter1+2 additions 17): model selection
  and interval calibration.
- Split B (train cold+iter1+iter2 27 -> test iter3 additions 15): later-period
  check. Both splits were inspected during initial exploration, so this is a
  backtest of a rule chosen after seeing the data, not a pristine holdout.
- Grouped leave-one-out CV over the 42 labeled rows: the duplicated recipe
  (rows 17/18) is held out TOGETHER, so no identical formulation sits in train
  and test at once. LOO is descriptive fit quality only; it is NOT used for
  calibration or model selection.
- Every model is reported on every split.

Uncertainty:
- Raw GP intervals (with WhiteKernel) are reported.
- A multiplicative inflation factor is calibrated on split A residuals
  (z-scores of held-out split-A points) and applied to split B and to
  recommendations. A simple post-hoc adjustment, not conformal prediction.

Acquisition (next-experiment suggestion):
- Expected improvement computed with the LATENT predictive distribution
  (WhiteKernel observation noise removed from the posterior variance), against
  the best observed value. This is a heuristic ranking for lab discussion,
  not a promise of improvement; suggested recipes are NOT validated results.
- A diverse batch (greedy max-EI with a minimum pairwise L2 distance of 0.25,
  an arbitrary algorithm choice) is reported alongside the single best point,
  because the unconstrained top-5 are near-duplicates of each other.
- Exclusion zone: candidates within L2 0.05 (algorithm choice) of any of the 55
  already-formulated recipes are dropped.

Source-scale values:
- The authors' own rdm_doe_actual_neu.xlsx (sheet 'unnorm') provides
  un-normalized values. Matched on 51 of 55 rows by exact HDI-IPDI values, the
  mapping is exactly linear per column (R2 = 1.000000):
    vernetzung_source_scale       = 0.399598 * vernetzung + 0.806557
    hdi_ipdi_source_scale         = hdi_ipdi
    mattierungsmittel_source_scale = 4.980258 * mattierungsmittel
    pigmentpaste_source_scale     = 19.983911 * pigmentpaste + 4.997254
  The authors do not document the unit labels, so these are source-scale
  numbers, not verified physical recipes. Per-coordinate ranges do not prove
  the joint combination is feasible.

Requires: numpy, pandas, scikit-learn, scipy, matplotlib, joblib. sklearn 1.7.2.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.linalg import solve_triangular
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent
DATA = OUT / "data" / "purformance_labeled.csv"
UNLABELED = OUT / "data" / "purformance_unlabeled_randomtest.csv"

FEATS = ["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]
TARGET = "kratzhaerte"

# verified against the authors' unnorm sheet (R2 = 1.000000 on 51 matched rows)
DENORM = {  # feature: (slope, intercept)
    "vernetzung": (0.399598, 0.806557),
    "hdi_ipdi": (1.0, 0.0),
    "mattierungsmittel": (4.980258, 0.0),
    "pigmentpaste": (19.983911, 4.997254),
}

df = pd.read_csv(DATA)
X = df[FEATS].to_numpy()
y = df[TARGET].to_numpy()
stages = df["stage"].to_numpy()
print(f"Loaded {len(df)} labeled rows. Target range {y.min():.3f} to {y.max():.3f}.")


def denorm_row(row):
    return {f: DENORM[f][0] * v + DENORM[f][1] for f, v in zip(FEATS, row)}


def make_models():
    """Fresh instances per fit: hyperparameters are re-fit on training data only.
    Kernel bounds are fixed in the code: the paper kernel's bounds come from the
    authors' published code; the anisotropic variant's bounds were chosen during
    exploration."""
    paper_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=1e-1, length_scale_bounds=(1e-3, 1.0)
    ) + WhiteKernel(noise_level=5e-2, noise_level_bounds=(1e-3, 1e3))
    tuned_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=[0.3, 0.3, 0.3, 0.3], length_scale_bounds=(1e-2, 10.0)
    ) + WhiteKernel(noise_level=5e-2, noise_level_bounds=(1e-3, 1e3))
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "knn_5": KNeighborsRegressor(n_neighbors=5),
        "random_forest": RandomForestRegressor(
            n_estimators=500, random_state=SEED, n_jobs=-1),
        "grad_boost": GradientBoostingRegressor(random_state=SEED),
        "gp_paper_kernel": GaussianProcessRegressor(
            kernel=paper_kernel, alpha=0.0, random_state=SEED,
            n_restarts_optimizer=2, normalize_y=True),
        "gp_anisotropic": GaussianProcessRegressor(
            kernel=tuned_kernel, alpha=0.0, random_state=SEED,
            n_restarts_optimizer=2, normalize_y=True),
    }


def metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


print("stage: splits"); cold = np.where(stages == "cold_start")[0]
i1 = np.where(stages == "iter1")[0]
i2 = np.where(stages == "iter2")[0]
i3 = np.where(stages == "iter3")[0]
trA, teA = cold, np.concatenate([i1, i2])
trB, teB = np.concatenate([cold, i1, i2]), i3

results = {"seed": SEED, "n_labeled": int(len(df)),
           "test_target_std_splitB": float(np.std(y[teB]))}

# ---------- all models on both temporal splits (validation A, final test B) --
def run_split(train_idx, test_idx, inflate=1.0):
    rows, gp_store = {}, {}
    for mname, model in make_models().items():
        model.fit(X[train_idx], y[train_idx])
        if mname.startswith("gp"):
            pred, std = model.predict(X[test_idx], return_std=True)
            z_raw = (y[test_idx] - pred) / std
            m = metrics(y[test_idx], pred)
            m["coverage_95_raw"] = float(np.mean(np.abs(z_raw) <= 1.96))
            z_cal = (y[test_idx] - pred) / (std * inflate)
            m["coverage_95_calibrated"] = float(np.mean(np.abs(z_cal) <= 1.96))
            gp_store[mname] = (pred, std)
        else:
            pred = model.predict(X[test_idx])
            m = metrics(y[test_idx], pred)
        rows[mname] = m
    return rows, gp_store

# Split A first: raw (inflate=1.0) to get calibration residuals
rowsA_raw, gpA = run_split(trA, teA, inflate=1.0)
results["splitA_validation_train10_test17"] = {"n_train": len(trA),
                                               "n_test": len(teA), "models": rowsA_raw}

# Model selection on split A (lowest RMSE), fixed BEFORE touching split B
sel = min(rowsA_raw, key=lambda m: rowsA_raw[m]["rmse"])
results["model_selected_on_splitA"] = sel

# Calibration on split A residuals of the SELECTED model, if it is a GP;
# otherwise calibrate with the best GP on A. (Selection result: see results.json)
cal_model = sel if sel.startswith("gp") else "gp_paper_kernel"
predA, stdA = gpA[cal_model]
zA = (y[teA] - predA) / stdA
inflate = float(np.sqrt(np.mean(zA ** 2)))
results["calibration"] = {
    "calibrated_on": "split A residuals (17 out-of-sample points), model: " + cal_model,
    "inflation_factor": inflate,
    "note": "multiplicative widening of GP std; calibrated on split A; simple post-hoc adjustment, not conformal prediction",
}

# Split B: final test, evaluated once with the calibrated intervals
rowsB, gpB = run_split(trB, teB, inflate=inflate)
results["splitB_final_test_train27_test15"] = {"n_train": len(trB),
                                               "n_test": len(teB), "models": rowsB}

# ---------- rolling forward replay: train on rounds 0..k, test on round k+1 ----
# Mirrors how the data was actually collected, so no later round informs an
# earlier prediction. Reported next to splits A/B, never used to pick the model.
print("stage: forward replay")
order_stages = ["cold_start", "iter1", "iter2", "iter3"]
replay = []
for k in range(len(order_stages) - 1):
    tr_idx = np.where(np.isin(stages, order_stages[:k + 1]))[0]
    te_idx = np.where(stages == order_stages[k + 1])[0]
    rows_k, _ = run_split(tr_idx, te_idx, inflate=inflate)
    replay.append({"train_rounds": order_stages[:k + 1], "test_round": order_stages[k + 1],
                   "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                   "models": rows_k})
results["forward_replay_by_round"] = replay

# ---------- grouped LOO (descriptive only; duplicate recipe held out together)
print("stage: grouped loo"); uniq, inv = np.unique(X.round(9), axis=0, return_inverse=True)
groups = [np.where(inv == g)[0] for g in range(len(uniq))]
loo_pred = {m: np.full(len(df), np.nan) for m in make_models()}
loo_std = {m: np.full(len(df), np.nan) for m in make_models()}
for g in groups:
    te = g
    tr = np.setdiff1d(np.arange(len(df)), te)
    for mname, model in make_models().items():
        model.fit(X[tr], y[tr])
        if mname.startswith("gp"):
            p, s = model.predict(X[te], return_std=True)
            loo_pred[mname][te] = p
            loo_std[mname][te] = s
        else:
            loo_pred[mname][te] = model.predict(X[te])
loo_rows = {}
for mname in loo_pred:
    m = metrics(y, loo_pred[mname])
    if mname.startswith("gp"):
        z = (y - loo_pred[mname]) / loo_std[mname]
        m["coverage_95_raw"] = float(np.mean(np.abs(z) <= 1.96))
    loo_rows[mname] = m
results["grouped_loo_descriptive"] = {
    "models": loo_rows,
    "note": "descriptive fit quality only; duplicate recipe rows 17/18 held out "
            "together; NOT used for selection or calibration",
}

# ---------- final model on all 42 rows; latent-EI acquisition ----------------
final_models = make_models()
gp = final_models[cal_model]
gp.fit(X, y)
results["final_model"] = {"name": cal_model, "kernel": str(gp.kernel_)}
joblib.dump(gp, OUT / "model.joblib")

# latent posterior std (WhiteKernel noise removed)
def latent_std(gp_model, Xq):
    K_lat = gp_model.kernel_.k1  # ConstantKernel * RBF part
    kXX = K_lat(gp_model.X_train_, Xq)
    v = solve_triangular(gp_model.L_, kXX, lower=True)
    # latent prior diagonal of ConstantKernel*RBF is a scalar (RBF diag == 1);
    # computing np.diag(K_lat(Xq)) would materialize an n x n matrix (OOM)
    k_diag = float(K_lat(Xq[:1])[0, 0])
    var_lat = k_diag - np.sum(v ** 2, axis=0)
    return np.sqrt(np.clip(var_lat, 0, None))

print("stage: candidates"); N_CAND = 300_000
cand = rng.random((N_CAND, 4))
unl = pd.read_csv(UNLABELED)[FEATS].to_numpy()
existing = np.vstack([X, unl])
keep = np.ones(N_CAND, dtype=bool)
for i in range(0, N_CAND, 20_000):
    d = np.linalg.norm(cand[i:i + 20_000, None, :] - existing[None, :, :], axis=2)
    keep[i:i + 20_000] = d.min(axis=1) > 0.05
cand = cand[keep]
print(f"Candidate pool after exclusion: {len(cand)}")

means, lat_stds, obs_stds = [], [], []
for i in range(0, len(cand), 20_000):
    m, s_obs = gp.predict(cand[i:i + 20_000], return_std=True)
    means.append(m)
    obs_stds.append(s_obs)
    lat_stds.append(latent_std(gp, cand[i:i + 20_000]))
mean = np.concatenate(means)
std_lat = np.concatenate(lat_stds)
std_obs = np.concatenate(obs_stds)

y_best = y.max()
# Noise-robust EI: with a noisy objective the best OBSERVED value is inflated by
# measurement noise, so improvement is measured against the best LATENT estimate
# (max GP posterior mean at the training points) using the LATENT predictive
# std (WhiteKernel observation noise removed). This is the standard plug-in
# variant of EI for noisy objectives; it is still a heuristic ranking.
mu_train = gp.predict(X)
y_best_latent = float(np.max(mu_train))
imp = mean - y_best_latent
ei = imp * norm.cdf(imp / std_lat) + std_lat * norm.pdf(imp / std_lat)
ei[std_lat < 1e-9] = 0.0  # latent EI: observation noise excluded

def rec_dict(rank, j):
    r = {"rank": rank}
    r.update({f: round(float(cand[j, k]), 4) for k, f in enumerate(FEATS)})
    r.update({f + "_source_scale": round(denorm_row(cand[j])[f], 3) for f in FEATS})
    r["predicted_kratzhaerte"] = round(float(mean[j]), 3)
    r["std_latent"] = round(float(std_lat[j]), 3)
    r["std_observation_calibrated"] = round(float(std_obs[j] * inflate), 3)
    r["ci95_halfwidth_calibrated"] = round(float(1.96 * std_obs[j] * inflate), 3)
    r["expected_improvement_latent"] = round(float(ei[j]), 4)
    return r

print("stage: acquisition done"); top1 = int(np.argmax(ei))
recs = [rec_dict(1, top1)]

# diverse batch: greedy max-EI with min pairwise L2 0.25 (algorithm choice)
chosen = [top1]
order = np.argsort(-ei)
for j in order:
    if len(chosen) >= 5:
        break
    if all(np.linalg.norm(cand[j] - cand[c]) >= 0.25 for c in chosen):
        chosen.append(int(j))
for rank, j in enumerate(chosen[1:], 2):
    recs.append(rec_dict(rank, j))

results["best_measured_kratzhaerte"] = float(y_best)
results["best_latent_estimate_kratzhaerte"] = y_best_latent
results["recommendation_notes"] = (
    "Heuristic ranking (latent EI), NOT validated results and no promise of "
    "improvement. Batch entries 2-5 are diversity-constrained (min L2 0.25). "
    "Source-scale values use the authors' own unnorm mapping; unit labels are not "
    "documented in the source data. Tested per-coordinate source-scale ranges (joint feasibility not implied): vernetzung "
    "0.807-1.206, hdi_ipdi 0-0.979, mattierungsmittel 0-4.98, pigmentpaste 5.68-24.98."
)
results["recommendations"] = recs
pd.DataFrame(recs).to_csv(OUT / "next_experiments.csv", index=False)

model_card = {
    "model": cal_model, "kernel": str(gp.kernel_),
    "inflation_factor": inflate,
    "features_normalized_0_1": FEATS,
    "denorm": {f: {"slope": DENORM[f][0], "intercept": DENORM[f][1]} for f in FEATS},
    "target": "kratzhaerte (scratch hardness, critical load)",
    "best_measured": float(y_best),
    "trained_on": "42 labeled PURformance rows (Zenodo 10.5281/zenodo.13742097, CC-BY-4.0)",
    "seed": SEED,
}
with open(OUT / "model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)

_lo = loo_pred[cal_model] - 1.96 * loo_std[cal_model]
_hi = loo_pred[cal_model] + 1.96 * loo_std[cal_model]
pd.DataFrame({
    "row_id": df["row_id"], "stage": stages,
    "round_index": [order_stages.index(t) for t in stages],
    "actual": y,
    "loo_pred_" + cal_model: loo_pred[cal_model],
    "residual": loo_pred[cal_model] - y,
    "loo_std_" + cal_model: loo_std[cal_model],
    "raw_95_lower": _lo, "raw_95_upper": _hi,
    "covered_raw_95": ((y >= _lo) & (y <= _hi)).astype(int),
}).to_csv(OUT / "loo_predictions.csv", index=False)

# ---------- plots ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].errorbar(y, loo_pred[cal_model], yerr=1.96 * loo_std[cal_model],
               fmt="o", capsize=3, alpha=0.7, color="#1f5fb4")
lims = [min(y.min(), np.nanmin(loo_pred[cal_model])) - 1,
        max(y.max(), np.nanmax(loo_pred[cal_model])) + 1]
ax[0].plot(lims, lims, "k--", lw=1)
ax[0].set_xlabel("Measured Kratzhaerte")
ax[0].set_ylabel("Grouped-LOO prediction (raw 95% interval)")
m0 = results["grouped_loo_descriptive"]["models"][cal_model]
ax[0].set_title(f"Grouped LOO (n=42): RMSE {m0['rmse']:.2f}, R2 {m0['r2']:.2f} (descriptive)")

predB, stdB = gpB[cal_model]
ax[1].errorbar(range(len(teB)), predB, yerr=1.96 * stdB * inflate, fmt="o-",
               capsize=3, color="#1f5fb4", label="GP prediction (calibrated 95%)")
ax[1].plot(range(len(teB)), y[teB], "s", color="#c0392b", label="Measured (iter 3)")
ax[1].set_xlabel("Iteration-3 formulation index")
ax[1].set_ylabel("Kratzhaerte")
mb = results["splitB_final_test_train27_test15"]["models"][cal_model]
ax[1].set_title(f"Final test B: train 27 -> test 15, RMSE {mb['rmse']:.2f}, R2 {mb['r2']:.2f}")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT / "validation_parity.png", dpi=160)

ls = gp.kernel_.k1.k2.length_scale
order2 = np.argsort(np.atleast_1d(ls))[:2] if np.ndim(ls) else [0, 1]
fig2, ax2 = plt.subplots(figsize=(7, 6))
sc = ax2.scatter(cand[:, order2[0]], cand[:, order2[1]], c=ei, s=2, cmap="viridis")
ax2.scatter(existing[:, order2[0]], existing[:, order2[1]], c="red", s=18,
            marker="x", label="55 formulated recipes")
names = {"vernetzung": "Vernetzung", "hdi_ipdi": "HDI-IPDI",
         "mattierungsmittel": "Mattierungsmittel", "pigmentpaste": "Pigmentpaste"}
ax2.set_xlabel(names[FEATS[int(order2[0])]])
ax2.set_ylabel(names[FEATS[int(order2[1])]])
ax2.set_title("Latent expected improvement over candidate space (projected)")
fig2.colorbar(sc, label="EI (latent)")
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT / "acquisition_landscape.png", dpi=160)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps({k: results[k] for k in
                  ["model_selected_on_splitA", "calibration", "final_model",
                   "recommendations"]}, indent=2))
