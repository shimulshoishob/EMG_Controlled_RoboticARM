"""Vectorised time- and frequency-domain EMG features for classical ML models.

All windows of a batch are processed at once (numpy), which keeps training fast and gives a
single-window inference cost of well under a millisecond.  Input must already be preprocessed
(DC removed, band-passed): shape (N, T, C).
"""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig

_EPS = 1e-9
TIME_FEATURES = ["mav", "rms", "var", "iemg", "wl", "maxabs", "zc", "ssc", "wamp", "skew", "kurt",
                 "hjorth_mob", "hjorth_cplx"]


class FeatureExtractor:
    def __init__(self, fs: float, n_channels: int, n_samples: int, cfg: FeatureConfig):
        self.fs, self.C, self.T, self.cfg = float(fs), int(n_channels), int(n_samples), cfg
        nyq = self.fs / 2.0
        lo, hi = cfg.freq_range_hz
        self.freqs = np.fft.rfftfreq(self.T, d=1.0 / self.fs)
        self.fmask = (self.freqs >= lo) & (self.freqs <= min(hi, nyq))
        self.fv = self.freqs[self.fmask]
        self.win = np.hanning(self.T).astype(np.float32)
        self.band_masks = [(self.fv >= a) & (self.fv < b) for a, b in cfg.bands_hz]
        self.iu = np.triu_indices(self.C, k=1)
        self.names = self._names()

    # ------------------------------------------------------------------------------------------
    def _names(self):
        names = []
        per_ch = []
        if self.cfg.time_domain:
            per_ch += TIME_FEATURES
        if self.cfg.frequency_domain:
            per_ch += ["mean_hz", "median_hz", "peak_hz", "spec_entropy", "log_power"] + \
                      [f"bp{int(a)}_{int(b)}" for a, b in self.cfg.bands_hz]
        for f in per_ch:
            names += [f"{f}_ch{c + 1}" for c in range(self.C)]
        if self.cfg.cross_channel:
            names += [f"rms_ratio_ch{c + 1}" for c in range(self.C)]
            names += [f"corr_ch{a + 1}_ch{b + 1}" for a, b in zip(*self.iu)]
        return names

    @property
    def n_features(self) -> int:
        return len(self.names)

    def transform(self, Xp: np.ndarray) -> np.ndarray:
        A = np.ascontiguousarray(np.asarray(Xp, dtype=np.float32).transpose(0, 2, 1))     # (N, C, T)
        A = A - A.mean(axis=-1, keepdims=True)
        blocks = []
        std = A.std(axis=-1) + _EPS
        rms = np.sqrt(np.mean(A * A, axis=-1))
        if self.cfg.time_domain:
            absA = np.abs(A)
            dA = np.diff(A, axis=-1)
            absd = np.abs(dA)
            thr = (self.cfg.zc_ssc_rel_threshold * std)[..., None]
            wthr = (self.cfg.wamp_rel_threshold * std)[..., None]
            var = A.var(axis=-1)
            zc = (((A[..., :-1] * A[..., 1:]) < 0) & (absd >= thr)).sum(-1)
            s1, s2 = A[..., 1:-1] - A[..., :-2], A[..., 1:-1] - A[..., 2:]
            ssc = (((s1 * s2) > 0) & ((np.abs(s1) + np.abs(s2)) >= thr)).sum(-1)
            wamp = (absd >= wthr).sum(-1)
            skew = np.mean(A ** 3, axis=-1) / std ** 3
            kurt = np.mean(A ** 4, axis=-1) / std ** 4 - 3.0
            var_d = dA.var(axis=-1) + _EPS
            var_dd = np.diff(dA, axis=-1).var(axis=-1) + _EPS
            mob = np.sqrt(var_d / (var + _EPS))
            cplx = np.sqrt(var_dd / var_d) / (mob + _EPS)
            blocks += [absA.mean(-1), rms, var, absA.sum(-1), absd.sum(-1), absA.max(-1),
                       zc, ssc, wamp, skew, kurt, mob, cplx]
        if self.cfg.frequency_domain:
            S = np.abs(np.fft.rfft(A * self.win, axis=-1)) ** 2                  # (N, C, F)
            sv = S[..., self.fmask]
            total = sv.sum(-1) + _EPS
            mean_hz = (sv * self.fv).sum(-1) / total
            csum = np.cumsum(sv, axis=-1)
            median_hz = self.fv[np.argmax(csum >= 0.5 * total[..., None], axis=-1)]
            peak_hz = self.fv[np.argmax(sv, axis=-1)]
            p = sv / total[..., None]
            entropy = -(p * np.log2(p + 1e-12)).sum(-1) / np.log2(max(sv.shape[-1], 2))
            bands = [sv[..., m].sum(-1) / total if m.any() else np.zeros_like(total) for m in self.band_masks]
            blocks += [mean_hz, median_hz, peak_hz, entropy, np.log1p(total)] + bands
        feats = [b.reshape(len(A), -1) for b in blocks]
        if self.cfg.cross_channel:
            feats.append(rms / (rms.mean(axis=1, keepdims=True) + _EPS))
            Z = A / std[..., None]
            corr = np.einsum("nct,ndt->ncd", Z, Z) / A.shape[-1]
            feats.append(corr[:, self.iu[0], self.iu[1]])
        out = np.concatenate(feats, axis=1).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
