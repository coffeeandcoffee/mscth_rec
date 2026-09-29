#!/usr/bin/env python3
"""
posthoc_powerline_peakfreq_check.py — read-only post-hoc checks cited in the thesis limitations.

1. Powerline share of the raw signal: per window, the share of raw power (>= 1 Hz) at 47-53 Hz,
   median over all windows, per electrode.
2. AF8 delta peak frequency tail: share of windows above 10 peaks/s per class, and Cohen's d
   with and without those windows.
3. Noise baseline: mean Peak Frequency and Macro Frequency of pink noise, run through the same
   band filters, Hilbert envelope and 1.0 s windows, next to the recorded means; plus the
   expected rate of envelope maxima for band-passed noise, 0.641 x bandwidth (Rice 1945).

Reads runs/<RUN>/windows/primary and runs/<RUN>/features. Writes nothing.
Run: cd 05-12_data_analysis_and_results && venv/bin/python scripts/posthoc_powerline_peakfreq_check.py
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, hilbert

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

RUN = Path(__file__).parent.parent / 'runs' / 'run_20260611_220844'
FS = 256


def cohens_d(v, y):
    a, b = v[y == 1], v[y == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def macro_count(col):
    """Same algorithm as step05_feature_engineering.py."""
    peaks, _ = find_peaks(col)
    if len(peaks) == 0:
        return 0
    is_above = col >= np.mean(col[peaks])
    runs, cur = [], 0
    for val in is_above:
        if not val:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    gap = np.mean(runs) if runs else 0
    smooth, last = is_above.copy(), -1
    for i, val in enumerate(is_above):
        if val:
            if last != -1 and 0 < i - last - 1 <= gap:
                smooth[last + 1:i] = True
            last = i
    return int(smooth[0]) + int(np.sum(~smooth[:-1] & smooth[1:]))


# 1. Powerline share of the raw signal
shares = {ch: [] for ch in config.EEG_CHANNELS}
for f in sorted(glob.glob(str(RUN / 'windows' / 'primary' / 'P*.pkl'))):
    d = pickle.load(open(f, 'rb'))
    names = d['feature_names']
    for w in d['windows']:
        data = np.asarray(w['data'])
        for ch in shares:
            r = data[:, names.index(ch)]
            r = r - r.mean()
            power = np.abs(np.fft.rfft(r)) ** 2
            freqs = np.fft.rfftfreq(len(r), 1 / FS)
            shares[ch].append(power[(freqs >= 47) & (freqs <= 53)].sum() / power[freqs >= 1].sum())
print('1. Median share of raw power at 47-53 Hz per window')
for ch, s in shares.items():
    print(f'   {ch}: {np.median(s) * 100:.0f}%  (n={len(s)})')

# 2. AF8 delta peak frequency tail
rows, labels = [], []
for f in sorted(glob.glob(str(RUN / 'features' / 'P*.pkl'))):
    d = pickle.load(open(f, 'rb'))
    names = d['agg_names_full']
    rows.append(d['features_full'])
    labels.append(d['labels'])
X, y = np.vstack(rows), np.concatenate(labels)  # label 1 = STAY, 0 = SKIP
v = X[:, names.index('AF8_delta_peakfreq')]
tail = v > 10
print('\n2. AF8 delta peak frequency above 10 peaks/s')
print(f'   all windows {tail.mean() * 100:.1f}%, SKIP {tail[y == 0].mean() * 100:.1f}%, STAY {tail[y == 1].mean() * 100:.1f}%')
print(f'   |d| all windows {abs(cohens_d(v, y)):.3f}, without tail {abs(cohens_d(v[~tail], y[~tail])):.3f}')

# 3. Noise baseline
rng = np.random.default_rng(0)
n = FS * 600
spec = np.fft.rfft(rng.standard_normal(n))
freqs = np.fft.rfftfreq(n, 1 / FS)
freqs[0] = freqs[1]
pink = np.fft.irfft(spec / np.sqrt(freqs), n)
print('\n3. Band | Rice 0.641 x BW | pink noise PF, MF | recorded PF, MF')
for band, lo, hi in config.FREQUENCY_BANDS:
    env = np.abs(hilbert(config.extract_band_amplitude(pink, FS, lo, hi)))
    wins = env[:len(env) // FS * FS].reshape(-1, FS)
    pf_noise = np.mean([len(find_peaks(w)[0]) for w in wins])
    mf_noise = np.mean([macro_count(w) for w in wins])
    pf_rec = X[:, [i for i, nm in enumerate(names) if nm.endswith(f'_{band}_peakfreq')]].mean()
    mf_rec = X[:, [i for i, nm in enumerate(names) if nm.endswith(f'_{band}_macrofreq')]].mean()
    print(f'   {band:11s} {0.641 * (hi - lo):5.1f} | {pf_noise:5.1f} {mf_noise:4.1f} | {pf_rec:5.1f} {mf_rec:4.1f}')
