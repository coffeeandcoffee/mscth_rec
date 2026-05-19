# EEG Pipeline — Gotchas & Quick Start

For the next person touching `05-12_data_analysis_and_results/`. Read this before debugging anything.

---

## Environment

There are **two venvs** in the parent project folder. Only one works:

- ✅ `05-12_data_analysis_and_results/venv/` — use this
- ❌ `Data Recording and Quality Tests/mscth/` — broken, ignore

Activate before doing anything:

```bash
cd "05-12_data_analysis_and_results"
source venv/bin/activate
```

System Python is Homebrew 3.14 with PEP 668 — `pip install` refuses outside a venv. Don't fight it, just use the venv above.

---

## Running the pipeline

```bash
cd scripts
python3 run.py                              # fresh run, prompts to confirm parameters.json
python3 run.py --resume 20260518_153845     # resume from a specific run (last known good)
```

Each run creates `runs/run_YYYYMMDD_HHMMSS/` with `processed/`, `viz/`, manifests, and `parameters.json`. Always check `parameters.json` in your run dir to see what actually ran.

**Last run that completed steps 1–2:** `run_20260518_153845`
**Pipeline currently fails at:** Step 03 (see issues below)

---

## Pipeline architecture

Steps are numbered modules in `scripts/`. `run.py` calls each step's `run(run_dir, params)`. Constants and DSP utilities live in `config.py` — no step defines its own constants.

Two parallel signal branches throughout:
- `processed/nonotch/` — bandpass only, no notch
- `processed/notch/` — same + 50Hz notch applied

pkl schema (set by step01, all steps must respect):
```python
{
  'pid': int,
  'fs': float,
  'dfs': [DataFrame, ...],   # LIST of DataFrames, one per CSV — NOT a single 'df'
  'feature_names': [...],
  'baseline_stats': {'mean': {...}, 'std': {...}},
  'dropouts': [...],
}
```
**The key is `'dfs'` (list), never `'df'` (single).** This has already broken step05 once.

---

## Active blockers (fix these in order)

### 🔴 1. Step 03 crashes — KeyError on feature_names

**Symptom:**
```
File "step03_label_windows.py", line 170
    return df.iloc[indices][feature_names].values.astype(np.float32)
KeyError (via pandas _get_indexer_strict)
```

**Cause:** Step03 is indexing a DataFrame with `feature_names` columns that don't exist in it. Likely reading the wrong pkl key or operating on a merged DataFrame that lost the feature columns.

**Where to look:** `step03_label_windows.py` around line 170 and wherever it loads the pkl. Confirm it reads `data['dfs']` (list) and iterates over it, not `data['df']` (doesn't exist).

**Has not been fixed yet — next dev starts here.**

### 🔴 2. Step 05 crashes — KeyError: 'df'

**Symptom:**
```
File "step05_feature_engineering.py", line 147
    notch_df = notch_data['df']
KeyError: 'df'
```

**Cause:** Step05 reads `notch_data['df']` but the pkl stores `'dfs'` (list). Fix: change to `notch_data['dfs']` and handle the list. Check the rest of step05 for similar assumptions.

### 🟡 3. Step02 artifact detection doesn't actually use the notched signal

**Symptom:** EMG flags ~100% on both nonotch and notch branches — identical results.

**Cause:** Step02 reads raw channel columns (`df['TP9']`) from the pkl. These are **never notched** — notching only happens during feature computation in step01's `build_relative_power`. So both branches see identical raw EEG in step02.

**Fix needed:** Apply notch inside `_flag_ptp` in `step02_artifact_flag.py` before bandpassing, when operating on the notch branch. Or store notched raw channels in the notch pkl in step01.

**Context on the 100% EMG rate:** Even after fixing step02, expect high EMG flag rates on TP9/TP10 for many participants — Muse temporal electrodes have poor contact by design and genuinely show high broadband noise. This is a known hardware limitation, documented as a thesis caveat. The artifact flags are currently unused downstream (step03 uses `primary` manifest, not `artifact_exclude`), so this doesn't affect results.

---

## What's confirmed working

- ✅ Notch filter in `config.apply_notch_filter` works — verified by FFT
- ✅ Feature columns in notch pkl ARE notched (`TP9_high_gamma`: 1.10 nonotch → 0.85 notch for P6)
- ✅ Raw channel columns are intentionally NOT notched in the pkl (notch only applied during feature computation)
- ✅ Step01 pkl schema correct (`'dfs'` list, both branches)
- ✅ Steps 01–02 complete without crashing
- ✅ venv has all required packages

---

## Sanity checks

Run from `scripts/` with venv active. Edit the run timestamp to match your run.

**Verify notch works on features:**
```python
import pickle, numpy as np
with open('../runs/run_20260518_153845/processed/nonotch/P6.pkl','rb') as f:
    nn = pickle.load(f)
with open('../runs/run_20260518_153845/processed/notch/P6.pkl','rb') as f:
    n = pickle.load(f)
nn_hg = nn['dfs'][0]['TP9_high_gamma'].values
n_hg  = n['dfs'][0]['TP9_high_gamma'].values
print(f"nonotch high_gamma: {np.mean(np.abs(nn_hg)):.4f}")  # expect ~1.10
print(f"notch   high_gamma: {np.mean(np.abs(n_hg)):.4f}")   # expect ~0.85
# if both identical → notch broken
```

**Verify pkl schema before touching any step:**
```python
import pickle
with open('../runs/run_20260518_153845/processed/notch/P6.pkl','rb') as f:
    d = pickle.load(f)
print(d.keys())           # pid, fs, dfs, feature_names, baseline_stats, dropouts
print(type(d['dfs']))     # <class 'list'>
print(len(d['dfs']))      # number of CSVs for this participant
print(d['dfs'][0].columns.tolist())  # lsl_timestamp, class, TP9, AF7, AF8, TP10, + 28 feature cols
```

---

## Artifact detection — design rules (don't break these)

1. **No adaptive thresholds** — no SD floors, no z-scores. The param value IS the threshold.
2. **Unified mechanism** — both blink and EMG use peak-to-peak on bandpassed signal vs fixed µV.
3. **Viz reads from params** — `viz02.py` never hardcodes thresholds. Reads `params['step02']`, plots the bandpassed signal the detector evaluated, draws lines at `±thresh_uv/2`.
4. **Bandpass full signal first, then slice** — slicing then filtering causes edge ringing.

Current thresholds (`config.DEFAULT_PARAMS['step02']`): `blink_thresh_uv=800`, `emg_thresh_uv=500`.

---

## Participants

Defined in `config.py`:
- `ALL_PARTICIPANTS`: P4–P31
- `EXCLUDED_PARTICIPANTS`: [16, 19, 29]
- `INCLUDED_PARTICIPANTS`: 25 participants
- `SESSION_MAP`: PID → session folder. If a recording moves or is renamed, update here.

---

## When something breaks

1. Confirm venv is active: `which python3` should point inside `05-12_.../venv/`
2. Check `parameters.json` in your run dir — that's what actually ran
3. Resume from broken step: `python3 run.py --resume <timestamp>` — don't re-run steps 1–4 unnecessarily
4. Before touching any pkl: `pickle.load` it and print `.keys()` and `type(d['dfs'])`
5. **Never assume `'df'` exists** — the key is always `'dfs'` (list)