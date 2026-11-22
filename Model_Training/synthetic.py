"""Synthetic EMG generator - for smoke-testing the pipeline ONLY.

The data it writes mimics the BioWave raw-recording format (same CSV header, ADC-count scale, 500 Hz)
and file-naming convention, so the whole pipeline (discovery -> splits -> models -> report) can be exercised
without a labelled dataset.  Scores obtained on it say nothing about real gesture recognition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

GESTURES = ["left", "right", "up", "down", "closed_fist", "double_fist", "victory", "thumbs_up", "pinch", "rest"]
HEADER = "host_time_s,packet_seq,frame_id,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8,roll,pitch,yaw"


def _prototypes(n_ch: int):
    rng = np.random.default_rng(1234)                      # shared "anatomy": same for every subject
    base = rng.dirichlet(np.ones(n_ch) * 0.6, size=3)       # three muscle synergies
    amps, freqs = {}, {}
    for i, g in enumerate(GESTURES):
        mix = rng.dirichlet(np.ones(3) * 0.7)
        amps[g] = (mix @ base) * n_ch * (0.05 if g == "rest" else 1.0)
        freqs[g] = rng.uniform(55, 130)
    return amps, freqs


def _envelope(g: str, n: int, fs: float, rng) -> np.ndarray:
    t = np.arange(n) / fs
    dur = n / fs
    on, off = 0.25 * dur * rng.uniform(0.7, 1.3), 0.85 * dur * rng.uniform(0.9, 1.0)
    sm = lambda x: 1 / (1 + np.exp(-x))
    env = sm((t - on) * 12) * sm((off - t) * 12)
    if g == "double_fist":
        gap = 0.5 * (off - on)
        env = np.clip(sm((t - on) * 14) * sm((on + 0.4 * gap - t) * 14) + sm((t - on - 0.6 * gap) * 14) * sm((off - t) * 14), 0, 1)
    if g in ("left", "right", "up", "down"):
        env = env * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.8, 1.4) * t + rng.uniform(0, 6)))
    return env


def generate_dataset(out_dir: Path, n_subjects: int = 4, n_sessions: int = 2, seconds: float = 6.0,
                     fs: float = 500.0, n_ch: int = 8, seed: int = 7) -> int:
    """Write ``n_subjects x n_sessions x 10`` recordings; returns the number of files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    amps, freqs = _prototypes(n_ch)
    n = int(seconds * fs)
    count = 0
    for s in range(1, n_subjects + 1):
        subj_gain = rng.lognormal(0, 0.35, n_ch)
        shift = int(rng.integers(-1, 2))
        dc = 2048 + rng.normal(0, 40, n_ch)
        noise_sd = rng.uniform(4, 9)
        for sess in range(1, n_sessions + 1):
            sess_gain = rng.lognormal(0, 0.15, n_ch)
            for gi, g in enumerate(GESTURES):
                trial_gain = rng.lognormal(0, 0.25, n_ch)
                a = np.roll(amps[g], shift) * subj_gain * sess_gain * trial_gain * 140.0
                fc = freqs[g] * rng.uniform(0.85, 1.15)
                sos = butter(2, [max(fc * 0.6, 20), min(fc * 1.6, 230)], btype="band", fs=fs, output="sos")
                env = _envelope(g, n, fs, rng)
                sig = np.empty((n, n_ch))
                for c in range(n_ch):
                    carrier = sosfilt(sos, rng.normal(0, 1, n))
                    carrier /= carrier.std() + 1e-9
                    hum = 5.0 * np.sin(2 * np.pi * 50 * np.arange(n) / fs + rng.uniform(0, 6))
                    sig[:, c] = dc[c] + a[c] * env * carrier + rng.normal(0, noise_sd, n) + hum
                sig = np.clip(np.round(sig), 0, 4095)
                stamp = f"20261101_{9 + s:02d}{sess:02d}{gi:02d}"
                path = out_dir / f"emg_{stamp}_S{s:02d}_sess{sess}_{g}_trial1.csv"
                t = (np.arange(n) + 1) / fs
                with open(path, "w", newline="") as fh:
                    fh.write(HEADER + "\n")
                    for i in range(n):
                        fh.write(f"{t[i]:.4f},{i // 20},{i}," + ",".join(str(int(v)) for v in sig[i]) + ",0.00,0.00,0.00\n")
                count += 1
    return count


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data_demo"
    print(f"wrote {generate_dataset(target)} files to {target}")
