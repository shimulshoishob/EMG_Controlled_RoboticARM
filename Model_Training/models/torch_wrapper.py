"""Training / inference wrapper that gives PyTorch networks the same interface as sklearn models,
plus the CNN -> Random Forest hybrid."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from ..utils import LOG
from .base import BaseModel


def resolve_device(pref: str) -> str:
    from ..utils import have
    if not have("torch"):
        return "cpu"
    import torch
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


class TorchModel(BaseModel):
    kind_name = "torch"
    input_kind = "raw"

    def __init__(self, key, family, arch, c_in, n_samples, n_classes, train_cfg=None, hp=None, device="cpu", seed=0):
        super().__init__()
        from .dl_nets import build_net
        import torch
        self.key, self.family, self.arch = key, family, arch
        self.c_in, self.n_samples, self.n_classes = c_in, n_samples, n_classes
        self.cfg, self.hp, self.device, self.seed = train_cfg, dict(hp or {}), device, seed
        torch.manual_seed(seed)
        self.net = build_net(arch, c_in, n_samples, n_classes, self.hp)

    # ---------------------------------------------------------------------------------------- training
    def fit(self, train, val=None):
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import f1_score
        cfg = self.cfg
        torch.set_num_threads(max(1, int(cfg.torch_threads)))
        dev = torch.device(self.device)
        net = self.net.to(dev)
        Xtr = torch.from_numpy(train["X"]).float()
        ytr = torch.from_numpy(train["y"]).long()
        n = len(Xtr)
        epochs = int(self.hp_epochs(val))
        bs = int(cfg.batch_size)
        steps = max(1, (n + bs - 1) // bs)
        opt_cls = torch.optim.Adam if str(getattr(cfg, "optimizer", "adamw")).lower() == "adam" else torch.optim.AdamW
        opt = opt_cls(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=epochs * steps, pct_start=0.2)
        weight = None
        if cfg.class_weight:
            counts = torch.bincount(ytr, minlength=self.n_classes).float().clamp(min=1)
            weight = (counts.sum() / (self.n_classes * counts)).to(dev)
        gen = torch.Generator().manual_seed(self.seed)
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
        best_state, best_score, bad = None, -1.0, 0
        t_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(n, generator=gen)
            running = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
                if cfg.augment:
                    xb = xb * (1.0 + 0.1 * torch.randn(len(xb), 1, 1, device=dev)) + 0.05 * torch.randn_like(xb)
                loss = F.cross_entropy(net(xb), yb, weight=weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step() if sched.last_epoch < epochs * steps - 1 else None
                running += float(loss) * len(idx)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(running / n)
            if val is not None:
                vloss, vacc, vf1 = self._eval(net, val, dev, f1_score)
                hist["val_loss"].append(vloss)
                hist["val_acc"].append(vacc)
                hist["val_macro_f1"].append(vf1)
                score = vf1 - 1e-3 * vloss
                if score > best_score:
                    best_score, bad = score, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                LOG.info("    %-16s epoch %3d/%d  train_loss %.4f%s", self.key, epoch, epochs, hist["train_loss"][-1],
                         f"  val_loss {hist['val_loss'][-1]:.4f}  val_f1 {hist['val_macro_f1'][-1]:.3f}" if val is not None else "")
            if val is not None and bad >= cfg.patience:
                LOG.info("    %-16s early stop at epoch %d (best val macro-F1 %.3f)", self.key, epoch, best_score)
                break
            if cfg.max_train_seconds and time.perf_counter() - t_start > cfg.max_train_seconds:
                LOG.warning("    %-16s hit the %.0fs time cap at epoch %d - keeping best weights so far",
                            self.key, cfg.max_train_seconds, epoch)
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.to("cpu").eval()                    # deployment / benchmarking is CPU
        self.history = hist
        return self

    def hp_epochs(self, val):
        return self.cfg.epochs if val is not None else min(self.cfg.epochs, self.cfg.cv_epochs)

    def _eval(self, net, val, dev, f1_score):
        import torch
        import torch.nn.functional as F
        net.eval()
        X = torch.from_numpy(val["X"]).float()
        y = torch.from_numpy(val["y"]).long()
        losses, preds = 0.0, []
        with torch.inference_mode():
            for i in range(0, len(X), 512):
                logits = net(X[i:i + 512].to(dev))
                losses += float(F.cross_entropy(logits, y[i:i + 512].to(dev), reduction="sum"))
                preds.append(logits.argmax(1).cpu())
        p = torch.cat(preds).numpy()
        yt = y.numpy()
        return losses / len(X), float((p == yt).mean()), float(f1_score(yt, p, average="macro", zero_division=0))

    # -------------------------------------------------------------------------------------- inference
    def _forward(self, X, fn):
        import torch
        outs = []
        with torch.inference_mode():
            for i in range(0, len(X), 1024):
                outs.append(fn(torch.from_numpy(np.ascontiguousarray(X[i:i + 1024])).float()))
        return torch.cat(outs).numpy()

    def predict_proba(self, X):
        import torch
        self.net.eval()
        return self._forward(X, lambda t: torch.softmax(self.net(t), dim=1))

    def embed(self, X):
        self.net.eval()
        return self._forward(X, self.net.embed)

    def set_inference_threads(self, n: int) -> None:
        import torch
        torch.set_num_threads(int(n))

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    def save(self, directory: Path) -> Path:
        import torch
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pt"
        torch.save({"key": self.key, "family": self.family, "arch": self.arch, "c_in": self.c_in,
                    "n_samples": self.n_samples, "n_classes": self.n_classes, "hp": self.hp,
                    "state": self.net.state_dict()}, path)
        return path

    @classmethod
    def load(cls, directory: Path):
        import torch
        d = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=False)
        m = cls(d["key"], d["family"], d["arch"], d["c_in"], d["n_samples"], d["n_classes"], None, d["hp"])
        m.net.load_state_dict(d["state"])
        m.net.eval()
        return m


class HybridCNNRF(BaseModel):
    """CNN as a learned feature extractor, Random Forest as the classifier.

    1. A 1-D CNN is trained end-to-end on the raw windows (so its 64-d embedding is discriminative).
    2. The CNN is frozen; every window is mapped to its embedding.
    3. A Random Forest is trained on the embeddings and makes the final decision.
    Rationale: the CNN learns EMG-specific representations from raw signal, the forest adds a robust,
    non-linear decision rule and is insensitive to embedding scale.  Caveat: the forest sees embeddings of
    windows the CNN was fitted on (more separable than unseen data), so the forest cannot correct CNN overfitting.
    """

    kind_name = "hybrid_cnn_rf"
    input_kind = "raw"

    def __init__(self, key, family, cnn: TorchModel, rf, n_classes, n_jobs=4):
        super().__init__()
        self.key, self.family, self.cnn, self.rf, self.n_classes, self.n_jobs = key, family, cnn, rf, n_classes, n_jobs

    def fit(self, train, val=None):
        self.cnn.fit(train, val)
        emb = self.cnn.embed(train["X"])
        self.rf.fit(emb, train["y"])
        self.history = dict(self.cnn.history)
        return self

    def predict_proba(self, X):
        p = self.rf.predict_proba(self.cnn.embed(X))
        if p.shape[1] == self.n_classes:
            return p
        full = np.zeros((len(p), self.n_classes), dtype=p.dtype)
        full[:, self.rf.classes_.astype(int)] = p
        return full

    def set_inference_threads(self, n: int) -> None:
        self.cnn.set_inference_threads(n)
        self.rf.set_params(n_jobs=int(n))

    def n_params(self):
        return self.cnn.n_params()

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        self.cnn.save(directory)
        joblib.dump({"rf": self.rf, "n_classes": self.n_classes, "key": self.key, "family": self.family},
                    directory / "rf.joblib", compress=3)
        return directory / "model.pt"

    @classmethod
    def load(cls, directory: Path):
        d = joblib.load(Path(directory) / "rf.joblib")
        return cls(d["key"], d["family"], TorchModel.load(directory), d["rf"], d["n_classes"])
