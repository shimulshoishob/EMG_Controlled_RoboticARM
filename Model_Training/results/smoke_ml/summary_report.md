# EMG gesture-model benchmark - smoke_ml

_Generated 2026-11-23 02:43:14 - 12 model(s) evaluated, 0 failed, 13 skipped._

> **WARNING - synthetic demo data.** These numbers only prove that the pipeline runs end to end. They say nothing about real gesture recognition; re-run on labelled BioWave recordings.

## 1. Evaluation protocol

- **Split mode:** `subject` - held-out data is independent of: *subject*.
- Windows: 200 ms (100 samples @ 500 Hz), stride 50 ms. Windows are created **after** splitting, inside each split, so no window straddles two splits.
- train / val / test windows: 5820 / 1940 / 1940.
- train subjects: S01, S02, S04
- val subjects: S03
- test subjects: S05
- Normalisation, feature scaling and class weights are fitted on the training split only. All models share identical windows, seed (42) and test data. Latency: CPU, 1 thread(s), full pipeline (filter + features/normalisation + model) per window.

## 2. Dataset

100 recordings, 8 channels, 5 subject(s), 10 session(s), 500 s total.
Labelled seconds per gesture: closed_fist: 50, double_fist: 50, down: 50, left: 50, pinch: 50, rest: 50, right: 50, thumbs_up: 50, up: 50, victory: 50.

## 3. Recommendations

| Criterion | Model | Evidence |
|---|---|---|
| Best overall classification performance | **Random Forest** | score 0.474 (acc 0.486, bal.acc 0.486, macro-F1 0.462) |
| Best Macro F1-score | **XGBoost** | 0.467 |
| Best subject-independent generalisation | **XGBoost** | mean macro-F1 over held-out subjects 0.467 +/- 0.000 |
| Fastest model for inference | **Gradient Boosting** | P95 9.884 ms, median 6.210 ms |
| Best model for real-time deployment | **Gradient Boosting** | rt-score 0.529: macro-F1 0.397, P95 9.88 ms (comfortable), 162 KB |
| Best lightweight model | **LightGBM** | 360.7 KB, macro-F1 0.432 |
| Best deep learning architecture | **n/a** | - |
| Best hybrid architecture | **n/a** | - |

- *Best overall classification performance* - Weighted blend of macro-F1/balanced-accuracy/accuracy ({'f1_macro': 0.5, 'balanced_accuracy': 0.25, 'accuracy': 0.25}); not accuracy alone.
- *Best Macro F1-score* - Treats every gesture equally, so it exposes weak classes that accuracy hides.
- *Best subject-independent generalisation* - Trained without the test subjects; ranked by mean minus half the std across held-out subjects (rewards consistency).
- *Fastest model for inference* - Lowest P95 end-to-end latency among models within 0.15 macro-F1 of the best (absolute fastest: Logistic Regression at 7.98 ms, but macro-F1 only 0.194).
- *Best model for real-time deployment* - Max of {'f1_macro': 0.6, 'latency': 0.25, 'size': 0.15} over accuracy, latency headroom vs the 50 ms budget and size; models over budget are excluded.
- *Best lightweight model* - Smallest on-disk model among those within 0.05 macro-F1 of the best.
- *Best deep learning architecture* - No model of this family completed.
- *Best hybrid architecture* - No model of this family completed.

## 4. Accuracy vs efficiency trade-offs

**Reference - best predictive model: Random Forest** (macro-F1 0.462, P95 25.54 ms, 3224 KB).

- *Best model for real-time deployment* - **Gradient Boosting**: macro-F1 0.397 (-0.064 vs reference), P95 latency 9.88 ms (2.6x faster), 162 KB (19.9x smaller).

- *Best lightweight model* - **LightGBM**: macro-F1 0.432 (-0.030 vs reference), P95 latency 12.17 ms (2.1x faster), 361 KB (8.9x smaller).

**Pareto-optimal models** (no other model is both more accurate and faster): Logistic Regression (0.194 F1 @ 7.98 ms), MLP (0.240 F1 @ 8.07 ms), Gradient Boosting (0.397 F1 @ 9.88 ms), LightGBM (0.432 F1 @ 12.17 ms), Random Forest (0.462 F1 @ 25.54 ms), XGBoost (0.467 F1 @ 30.56 ms).

**Statistical caution:** test accuracy has a ~95% half-width of about +/-0.043 (effective sample size shrinks because windows overlap). Differences smaller than that are not evidence that one model is better; prefer the cheaper model when scores are tied.

## 5. Leaderboard (sorted by overall performance score)

| Model | Family | Acc | Bal.Acc | Macro-F1 | AUC | P95 ms | Size KB | Train s | Real-time |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | ml | 0.486 | 0.486 | 0.462 | 0.864 | 25.5 | 3224.4 | 5.36 | marginal |
| XGBoost | ml | 0.473 | 0.473 | 0.467 | 0.853 | 30.6 | 642.1 | 39.8 | marginal |
| Extra Trees | ml | 0.466 | 0.466 | 0.443 | 0.867 | 26.7 | 4170.5 | 1.09 | marginal |
| LightGBM | ml | 0.440 | 0.440 | 0.432 | 0.823 | 12.2 | 360.7 | 23.9 | comfortable |
| Gradient Boosting | ml | 0.432 | 0.432 | 0.397 | 0.862 | 9.88 | 162.4 | 22.7 | comfortable |
| CatBoost | ml | 0.396 | 0.396 | 0.372 | 0.861 | 10.0 | 392.3 | 68.7 | comfortable |
| k-NN | ml | 0.329 | 0.329 | 0.307 | 0.687 | 14.7 | 3963.9 | 0.00 | marginal |
| MLP | ml | 0.271 | 0.271 | 0.240 | 0.756 | 8.07 | 384.3 | 8.32 | comfortable |
| SVM (RBF) | ml | 0.261 | 0.261 | 0.239 | 0.748 | 14.7 | 3861.4 | 5.49 | marginal |
| LDA | ml | 0.258 | 0.258 | 0.240 | 0.678 | 11.3 | 319.1 | 0.61 | comfortable |
| SVM (linear) | ml | 0.251 | 0.251 | 0.217 | 0.786 | 12.6 | 2155.6 | 3.72 | marginal |
| Logistic Regression | ml | 0.218 | 0.218 | 0.194 | 0.714 | 7.98 | 8.17 | 68.8 | comfortable |

## 6. Real-time feasibility (budget 50 ms = decision interval)

- **comfortable** (P95 <= 25% of budget): Logistic Regression, MLP, Gradient Boosting, CatBoost, LDA, LightGBM
- **marginal** (P95 within budget but little headroom): SVM (linear), k-NN, SVM (RBF), Random Forest, Extra Trees, XGBoost
- **too slow** (P95 exceeds the budget): none

## 7. Skipped / failed

- `cnn1d` skipped: family 'dl' not enabled
- `cnn2d` skipped: family 'dl' not enabled
- `lstm` skipped: family 'dl' not enabled
- `bilstm` skipped: family 'dl' not enabled
- `gru` skipped: family 'dl' not enabled
- `transformer` skipped: family 'dl' not enabled
- `cnn_lstm` skipped: family 'hybrid' not enabled
- `cnn_gru` skipped: family 'hybrid' not enabled
- `cnn_bilstm` skipped: family 'hybrid' not enabled
- `cnn_tcn` skipped: family 'hybrid' not enabled
- `cnn_attention` skipped: family 'hybrid' not enabled
- `cnn_transformer` skipped: family 'hybrid' not enabled
- `cnn_rf` skipped: family 'hybrid' not enabled

## 8. Files

`model_comparison.csv` / `metrics.json` (all metrics), `leaderboard.csv`, `recommendations.json`, `plots/` (rankings, latency, size, trade-off, radar, heatmaps), `confusion_matrices/`, `per_model/` (raw per-model records), `models/<key>/` (deployable bundles; `models/recommended_realtime/` is the pick above), `config.json`, `environment.json`, `training.log`.

Load a model: `from Model_Training.inference import GestureClassifier; clf = GestureClassifier.load('<models dir>/<key>')`.
