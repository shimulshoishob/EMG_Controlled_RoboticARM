<div align="center">

# 🧬 `models/`

**The 25 models and how they are built**

[↑ Model_Training](../README.md) · [Models in detail](../../docs/MACHINE_LEARNING.md#5-all-models)

</div>

| File | Role |
|---|---|
| [`registry.py`](registry.py) | `SPECS`: one `ModelSpec` per model (key, name, family, input kind, required libraries, factory, search space); the single source of truth for the CLI and the Studio |
| [`base.py`](base.py) | the common model interface used by training, evaluation, benchmarking and deployment |
| [`ml_models.py`](ml_models.py) | the 12 classical models on the 204 engineered features: Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, SVM (RBF / linear), k-NN, Logistic Regression, LDA, Gradient Boosting, MLP |
| [`dl_nets.py`](dl_nets.py) | the PyTorch networks, each defined once: 1D CNN, 2D CNN, LSTM, BiLSTM, GRU, Transformer and the CNN hybrids (shared `ConvFrontEnd`) |
| [`torch_wrapper.py`](torch_wrapper.py) | `TorchModel` (training loop, early stopping, augmentation, time cap, save/load) and `HybridCNNRF` (CNN embeddings → Random Forest) |

## Families

| Family | Key examples | Input |
|---|---|---|
| `ml` | `rf`, `extra_trees`, `xgboost`, `lightgbm`, `catboost`, `svm_rbf`, `svm_linear`, `knn`, `logreg`, `lda`, `gradient_boosting`, `mlp` | 204 features (standardised) |
| `dl` | `cnn1d`, `cnn2d`, `lstm`, `bilstm`, `gru`, `transformer` | raw normalised window (channels × samples) |
| `hybrid` | `cnn_lstm`, `cnn_gru`, `cnn_bilstm`, `cnn_tcn`, `cnn_attention`, `cnn_transformer`, `cnn_rf` | raw normalised window |

## Adding a model

1. Write a factory that returns an object following `base.py`.
2. Add a `ModelSpec(...)` line in `registry.py`.
3. It appears in `--list-models`, in the CLI run, and in the Training Studio automatically.

A model whose library is missing is **skipped with a `pip install` hint**; it never breaks the others.
