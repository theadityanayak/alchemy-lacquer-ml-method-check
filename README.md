# Alchemy 2026 Case 4 - PU lacquer scratch-hardness model

Method demonstration on real, openly licensed lab data: a Gaussian-process model of
PU lacquer scratch hardness (Kratzhaerte) with uncertainty estimates and a
next-experiment suggestion. Not a claim about architectural paint performance.

Status: working prototype. The validation below is a retrospective temporal
backtest on 42 measured formulations; broad prospective validation is still needed
before any result is trusted for lab decisions.

## Data provenance

- Zenodo 10.5281/zenodo.13742097 (resolves to v13742098), license CC-BY-4.0.
- `Experiment.zip` -> `Experiment/Iteration_4/PURformance_dataset.xlsx`, sheet `doe_actual`.
- Paper: DOI 10.1016/j.porgcoat.2025.109265.
- 55 rows; features in [0,1] (vernetzung/crosslinking, hdi_ipdi ratio,
  mattierungsmittel/matting agent, pigmentpaste); target kratzhaerte (scratch hardness).

## Data audit (checked before training)

1. Only 42 of 55 rows have a measured target. Rows 42-54 (the final random-test
   batch) were released without Kratzhaerte. No labels invented; those rows serve
   only as an exclusion zone for candidate generation.
2. The sheet is strictly chronological (prefix-matched against per-iteration files):
   10 cold start, 7 iter1, 10 iter2, 15 iter3, 13 unlabeled.
3. Rows 17/18 are one recipe measured twice: 16.765 vs 13.994 (spread 2.77).
4. Stage-wise means rise 14.08 -> 15.97; best observed value 22.55 found in iter1,
   consistent with the campaign maximizing scratch hardness.

## On measurement noise (hypothesis, not established fact)

Three observations are consistent with high measurement noise: the duplicate-recipe
spread of 2.77; the fitted GP putting more kernel variance on WhiteKernel noise
(0.714) than on signal (0.668^2, normalized target); and the gap between the best
observed value (22.55) and the best latent estimate (16.99). With only one
duplicated recipe, a variance decomposition into noise vs. signal is not possible,
so this stays a hypothesis. Verifying it needs repeat measurements per recipe.

## Evaluation: retrospective temporal backtest

The data was collected adaptively (Bayesian optimization), so random splits would
be optimistically biased. The chronological structure allows temporal splits:

- **Split A.** Train cold start (10) -> test iter1+2 additions (17).
- **Split B.** Train cold+iter1+iter2 (27) -> test iter3 additions (15).
- **Forward replay.** Train on rounds 0..k, test on round k+1 (three steps).
- **Grouped LOO** over the 42 labeled rows: the duplicated recipe (rows 17/18) is
  held out together. Reported as descriptive fit quality only.

Model choice and interval calibration use split A; split B is the later-period
check. Honesty caveat: both splits were inspected during initial exploration, so
this is a backtest of a rule chosen after seeing the data, not a pristine
holdout. A genuinely fresh validation (new formulations measured in the lab) is
still required.

- Model selection rule: lowest split-A RMSE -> selects `gp_paper_kernel`
  (the kernel from the authors' published code: ConstantKernel * RBF + WhiteKernel).
- Calibration: a multiplicative std inflation of 1.693, computed from the z-scores
  of the 17 held-out split-A points; a simple post-hoc adjustment, not conformal
  prediction.
- All 7 models are reported on all splits below.
- Hyperparameters are re-fit inside each training fold (fresh model instances).
  Kernel bounds are fixed in the code; the paper kernel's bounds come from the
  authors' published code, the anisotropic variant's bounds were chosen during
  exploration.

## Results

### Split A (train 10 -> test 17)

| model | RMSE | MAE | 95% cov (raw) |
|---|---|---|---|
| mean baseline | 3.68 | 2.88 | - |
| ridge | 3.55 | 2.67 | - |
| kNN-5 | 3.61 | 2.61 | - |
| random forest | 3.62 | 2.75 | - |
| gradient boosting | 3.97 | 3.17 | - |
| **GP, paper kernel (best on A)** | **3.27** | **2.46** | 0.76 |
| GP, anisotropic RBF | 3.85 | 2.97 | 0.35 |

The anisotropic GP's raw interval coverage moving from 0.35 on A to 1.00 on B is
consistent with covariate shift between the adaptively sampled periods; that is a
possible explanation, not an established cause.

### Split B (train 27 -> test 15)

| model | RMSE | MAE | R2 | 95% cov raw | 95% cov calibrated |
|---|---|---|---|---|---|
| mean baseline | 2.11 | 1.80 | -1.60 | - | - |
| ridge | 1.98 | 1.69 | -1.30 | - | - |
| kNN-5 | 2.10 | 1.80 | -1.57 | - | - |
| random forest | 1.92 | 1.64 | -1.15 | - | - |
| gradient boosting | 2.25 | 1.83 | -1.95 | - | - |
| GP, paper kernel (selected on A) | 1.76 | 1.50 | -0.81 | 1.00 | 1.00 |
| GP, anisotropic RBF | 1.72 | 1.45 | -0.72 | 1.00 | 1.00 |

Reading, without varnish:
- R2 is negative for every model, measured against the split-B test mean. The 15
  iter3 targets sit in a narrow band (std 1.31), which makes this R2 harsh, but
  the fact stands: on this batch no model explains the variation.
- The comparison that matters operationally is against carrying the training mean
  forward (the mean baseline row): the selected GP improves RMSE 2.11 -> 1.76
  (~17%) and MAE 1.80 -> 1.50. The anisotropic GP is marginally better on B
  (1.72/1.45) but was worse on A (3.85 vs 3.27); with only 15-17 test points
  it is uncertain whether this ranking flip means anything (not statistically
  tested), and the selection rule followed split A.
- The GP's 95% intervals covered all 15 held-out points, raw and calibrated. The
  intervals are wide; that is the honest state of the information in 42 points.

### Forward replay by round (train on rounds 0..k, test on round k+1)

This follows the order the data was collected in, so no later round informs an earlier
prediction. Intervals use the split-A inflation factor.

| train rounds | test round | n train / test | model | RMSE | MAE | R2 |
|---|---|---|---|---|---|---|
| cold_start | iter1 | 10 / 7 | mean baseline | 3.90 | 3.03 | -0.24 |
| cold_start | iter1 | 10 / 7 | random forest | 3.02 | 2.22 | 0.26 |
| cold_start | iter1 | 10 / 7 | GP, paper kernel | 2.18 | 1.70 | 0.61 |
| cold_start+iter1 | iter2 | 17 / 10 | mean baseline | 3.70 | 2.77 | -0.13 |
| cold_start+iter1 | iter2 | 17 / 10 | random forest | 4.56 | 3.35 | -0.72 |
| cold_start+iter1 | iter2 | 17 / 10 | GP, paper kernel | 5.48 | 3.70 | -1.49 |
| cold_start+iter1+iter2 | iter3 | 27 / 15 | mean baseline | 2.11 | 1.80 | -1.60 |
| cold_start+iter1+iter2 | iter3 | 27 / 15 | random forest | 1.92 | 1.64 | -1.15 |
| cold_start+iter1+iter2 | iter3 | 27 / 15 | GP, paper kernel | 1.76 | 1.50 | -0.81 |

GP calibrated 95% coverage per round: 1.00, 0.80, 1.00.
The GP beats the mean baseline for iter1 and iter3 but loses clearly on iter2 (RMSE 5.48 vs 3.70).
The model rule and calibration came from split A, which overlaps rounds iter1 and iter2, so those two rows
are not independent of model selection. Only the iter3 row is a clean forward test.

### Grouped LOO over all 42 rows (descriptive)

| model | RMSE | R2 | 95% cov raw |
|---|---|---|---|
| mean baseline | 2.93 | -0.05 | - |
| ridge | 2.70 | +0.11 | - |
| GP, paper kernel | 2.97 | -0.08 | 0.90 |
| GP, anisotropic RBF | 3.25 | -0.29 | 0.83 |

No model explains much variance in LOO. The model can beat the naive baseline
out-of-sample on this backtest, but it cannot reliably rank similar formulations,
and broad prospective validation is still open.

## Next-experiment suggestions (heuristic ranking, NOT validated results)

Acquisition: expected improvement computed with the LATENT predictive std
(WhiteKernel observation noise removed from the posterior variance) against the
best LATENT estimate (max GP posterior mean at the training points, 16.99) - the
plug-in variant of EI used for noisy objectives. The choice matters here:
measured against the best observed value (22.55) with the latent std, EI is numerically zero (to
printed precision) across the candidate pool, because the noiseless function
is not expected to beat a noise-inflated observation. The plug-in variant is a modeling choice, not the
only reasonable one; treat the output as a heuristic for lab discussion.

- Rank 1 (normalized): vernetzung 0.852, hdi_ipdi 0.533, mattierungsmittel 0.990,
  pigmentpaste 0.472. Predicted 17.43; calibrated 95% interval half-width 8.72
  (roughly [8.7, 26.2] - wide, honest).
- Source-scale values: the authors' `unnorm` sheet gives an exactly linear mapping
  per coordinate (R2 = 1.000000 on 51 matched rows). Rank 1 in source-scale:
  vernetzung 1.147, hdi_ipdi 0.533, mattierungsmittel 4.93, pigmentpaste 14.43.
  The source data does not document the unit labels, so these are source-scale
  numbers, not verified physical recipes. Each coordinate lies inside the tested
  per-coordinate ranges, but that does not prove the joint combination is
  feasible or safe to run; a formulator must judge that.
- Batch: ranks 2-5 add diversity via greedy max-EI with a minimum pairwise L2 of
  0.25 (an explicit algorithm choice); the unconstrained top-5 are near-identical.
- Candidate exclusion: L2 < 0.05 around all 55 formulated recipes (algorithm choice).
- Full list with both coordinate systems in `next_experiments.csv`.

## Interfaces

- `predict.py 0.852 0.533 0.990 0.472` - CLI prediction with calibrated 95%
  interval; `--source-scale` accepts authors'-unnorm values; extrapolation outside
  [0,1] is flagged.
- `app.py` - local interactive demo at http://127.0.0.1:8000 (pure stdlib server +
  saved model; runs offline, no public hosting).
- `test_model.py` - 23 checks (data integrity, selection/calibration consistency,
  recommendation sanity, CLI smoke tests). All passing.

## Reproduce

```
pip install -r requirements.txt
./run_all.sh                       # rebuilds results from data/ and runs the tests
python3 alchemy_lacquer_model.py   # rebuilds everything from data/
python3 test_model.py              # 23 checks
python3 predict.py 0.8524 0.5331 0.9905 0.4717
python3 app.py                     # local demo on :8000
```

`data/` holds the exact training CSVs (verbatim export from the Zenodo xlsx,
anglicized column names, no value changes). `model.joblib` + `model_card.json`
are the trained artifact. Fixed seed 42 throughout.

## Scope

This package covers a method check on an open PU-lacquer dataset only. It does not contain, and does not
validate, any conductive-coating (PEDOT:PSS/WPU) model; no conductive-coating data was used.
No deep learning was used or tested.

## AI assistance

AI tools assisted with the code, analysis and documentation in this package. Results are reproduced by `run_all.sh`.

## Licence

Data: CC-BY-4.0 by the original authors (see NOTICE_DATA.md). Code licence: to be chosen by the project authors.
