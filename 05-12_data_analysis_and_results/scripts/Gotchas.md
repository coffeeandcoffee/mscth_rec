# EEG Pipeline — Gotchas & Quick Start

For the next person touching `05-12_data_analysis_and_results/`. Read this before debugging anything.

---

## How to run

```bash
cd "05-12_data_analysis_and_results"
source venv/bin/activate
cd scripts
python3 run.py                              # fresh run, prompts to confirm parameters.json
python3 run.py --auto-approve               # skip confirmation prompt
python3 run.py --resume 20260520_065025     # resume from a specific run
python3 run.py --resume 20260520_065025 --step  # run only the next incomplete step
python3 run.py --resume 20260520_065025 --restep # repeat the last completed step and stop
```

Each run creates `runs/run_YYYYMMDD_HHMMSS/` with `processed/`, `windows/`, `viz/`, and `parameters.json`. Always check `parameters.json` in your run dir to see what actually ran.

---

## Environment

There are **two venvs** in the parent project folder. Only one works:

- ✅ `05-12_data_analysis_and_results/venv/` — use this
- ❌ `Data Recording and Quality Tests/mscth/` — broken, ignore

System Python is Homebrew 3.14 with PEP 668 — `pip install` refuses outside a venv. Don't fight it, just use the venv above.

---

## Pipeline architecture

Steps are numbered modules in `scripts/`. `run.py` calls each step's `run(run_dir, params)`. Constants and DSP utilities live in `config.py` — no step defines its own constants.

Two parallel signal branches throughout:
- `processed/nonotch/` — bandpass only, no notch
- `processed/notch/` — same + 50 Hz notch applied

### pkl schema (step01 output, all steps must respect)

```python
{
  'pid': int,
  'fs': float,
  'dfs': [DataFrame, ...],   # LIST of DataFrames, one per CSV — NEVER 'df'
  'feature_names': [...],
  'baseline_stats': {'mean': {...}, 'std': {...}},
  'dropouts': [...],
}
```

Each DataFrame in `dfs` also has `.attrs`:
```python
df.attrs['n_a_presses']  # int — count of A-keypresses in this CSV
df.attrs['a_press_times'] # list[float] — LSL timestamps of each A-keypress
```

**The key is `'dfs'` (list), never `'df'` (single).** This has already broken step05 once.

### Window pkl schema (step03 output)

```python
{
  'pid': int,
  'windows': [...],          # list of window dicts
  'feature_names': [...],
  'fs': float,
  'a_press_times': [...],    # all A-press timestamps for this participant
}
```

---

## Approved steps (manually validated)

| Step | Status | Notes |
|------|--------|-------|
| 01 — Preprocess | ✅ Approved | SKIP regions verified ~6s per isolated A-press |
| 02 — Artifact Flag | ✅ Approved | See viz02 caveat below |
| 03 — Window Labelling | ✅ Approved | Sliding window + burst detection working |
| 04 — Balance & Split | ✅ Approved | viz04 fixed: reads actual fold assignments + real STAY/SKIP counts |
| 05 — Feature Engineering | ✅ Approved | viz05 fixed: schematic reads stat count from params |
| 06+ | ⚠️ Not yet validated | Resume from here |

---

## Key parameters (config.py → parameters.json)

```
step01.skip_window_s    = 3.0   → labels ±3s around each A-press = 6s SKIP region
step03.half_window_s    = 3.0   → matches step01's ±3s (informational, not used in step03 logic)
step03.window_s         = 3.0   → extracted window duration (both SKIP and STAY)
step03.stride_s         = 0.6   → sliding window stride (both SKIP and STAY), 80% overlap
step03.burst_thresh_s   = 3.0   → used for burst block detection context
step02.blink_thresh_uv  = 800   → blink peak-to-peak threshold
step02.emg_thresh_uv    = 800   → EMG peak-to-peak threshold (raised from 500 to reduce false positives)
```

**Naming was cleaned up:** old `stay_stride_s` is now `stride_s` (applies to both SKIP and STAY).

---

## Step 01 — Preprocessing (detailed mathematical specification)

Step01 transforms raw Muse 2 CSV recordings into baseline-normalised spectral power features. Each participant's data is processed per-CSV (not merged) to preserve temporal boundaries.

### 1.1 Input format

Raw CSVs from Muse 2 via LSL. Columns: `lsl_timestamp`, `TP9`, `AF7`, `AF8`, `TP10` (µV), `keypress_A` (skip event), `keypress_B` (baseline marker). Native sampling ≈256 Hz but irregular (Bluetooth jitter).

### 1.2 STAY/SKIP labelling

Applied **before** interpolation, on raw timestamps:

```
For each sample s with timestamp t_s:
    class(s) = SKIP   if ∃ any A-press at time t_A where |t_s − t_A| ≤ skip_window_s
             = STAY   otherwise

    skip_window_s = 3.0   →   SKIP region = [t_A − 3, t_A + 3] = 6s per isolated A-press
```

When two A-presses are <6s apart, their SKIP regions overlap and merge naturally (union of intervals). No explicit merge logic — the per-sample rule handles it.

### 1.3 Temporal trimming

After labelling, each CSV is trimmed to `[t_firstA − 3s, t_lastA + 3s]`. CSVs with zero A-presses are discarded entirely.

### 1.4 Interpolation to uniform grid

The Muse 2's Bluetooth stream has irregular inter-sample intervals (~3.5–4.5 ms). All channels are resampled to a uniform 256 Hz grid:

```
t_uniform = [t_start, t_start + 1/256, t_start + 2/256, ..., t_end)
```

- **EEG channels** (TP9, AF7, AF8, TP10): linear interpolation via `scipy.interpolate.interp1d(kind='linear')`
- **Class labels** (STAY/SKIP): nearest-neighbour interpolation to preserve discrete labels without blending
- **DC offset removal**: after interpolation, each channel is zero-centred: `x_ch(t) ← x_ch(t) − mean(x_ch)`
- **Keypress columns**: dropped after interpolation (A-press timestamps preserved in `df.attrs['a_press_times']`)

### 1.5 Baseline extraction (90s reference window)

The baseline provides the reference statistics for z-score normalisation. Extracted once per participant across all CSVs:

```
t_baseline_start = t_firstB + baseline_offset_s        (= t_firstB + 10s)
t_baseline_end   = t_firstB + baseline_offset_s + baseline_duration_s   (= t_firstB + 100s)
```

Where `t_firstB` = timestamp of the first B-keypress (baseline marker). The 10s offset skips the initial settling period.

Baseline samples are interpolated to 256 Hz, then processed through the same filter bank as the main data (see §1.6).

### 1.6 Butterworth bandpass filtering (7-band decomposition)

Each EEG channel is decomposed into 7 frequency bands using 4th-order Butterworth bandpass filters applied via zero-phase filtering (`scipy.signal.filtfilt`):

| Band | f_low (Hz) | f_high (Hz) |
|------|-----------|-------------|
| delta | 1 | 4 |
| theta | 4 | 8 |
| alpha | 8 | 13 |
| beta | 13 | 30 |
| low_gamma | 30 | 40 |
| high_gamma | 40 | 60 |
| very_high | 60 | 100 |

**Butterworth transfer function** (4th order bandpass):

```
H(s) = 1 / (1 + (s/ωc)^2n)    where n = 4 (order)

Implementation:
    ω_lo = f_low / (fs/2)       # normalised low cutoff
    ω_hi = f_high / (fs/2)      # normalised high cutoff
    b, a = scipy.signal.butter(4, [ω_lo, ω_hi], btype='band')
    x_band(t) = filtfilt(b, a, x_ch(t))    # zero-phase (no group delay)
```

`filtfilt` applies the filter forward then backward, doubling the effective order to 8 and ensuring zero phase distortion.

**Band amplitude** is the raw output of the bandpass filter (not envelope-extracted):

```
amp_band(t) = filtfilt(b, a, x_ch(t))
```

### 1.7 Instantaneous power

Band power is the squared amplitude at each time point:

```
P_band(t) = amp_band(t)²
```

### 1.8 Baseline statistics

For each of the 28 features (4 channels × 7 bands), compute mean and standard deviation over the 90s baseline window:

```
μ_baseline = mean(P_band(t))    for t ∈ baseline window
σ_baseline = std(P_band(t))     for t ∈ baseline window

Clamped: if μ < 10⁻¹², set μ = 10⁻¹²
         if σ < 10⁻¹², set σ = 10⁻¹²
```

### 1.9 Z-score normalisation

Each feature column in the output is the z-scored instantaneous power relative to baseline:

```
z(t) = (P_band(t) − μ_baseline) / σ_baseline
```

This produces 28 feature columns per DataFrame: `{channel}_{band}` (e.g. `TP9_alpha`, `AF8_high_gamma`).

**Interpretation:** z = 0 means baseline-typical power; z > 0 means above-baseline; z < 0 means below-baseline. Values are NOT constrained to be positive (unlike the old ratio normalisation `P/μ`).

### 1.10 Notch filter (ablation branch only)

Applied to the `processed/notch/` branch only, before bandpass filtering:

```
H_notch(z) = IIR notch at f = 50 Hz, Q = 30
b, a = scipy.signal.iirnotch(50.0, 30.0, fs=256)
x_notched(t) = filtfilt(b, a, x_ch(t))
```

The notch branch uses its own baseline statistics (computed from notch-filtered baseline data), so z-scores are self-consistent within each branch.

### 1.11 Output feature space

```
4 channels × 7 bands = 28 features per sample
Each feature = z-scored instantaneous band power

Total output per CSV: DataFrame with columns:
    lsl_timestamp, class, TP9, AF7, AF8, TP10,        ← raw (DC-removed)
    TP9_delta, TP9_theta, ..., TP10_very_high          ← 28 z-score features
```

---

## Step 02 — Artifact flagging (detailed)

### 2.1 Detection mechanism

Both blink and EMG artifacts use the same approach: bandpass the full signal, compute peak-to-peak amplitude in a sliding window, flag windows exceeding a fixed threshold.

```
For artifact type A (blink or EMG):
    1. x_bp(t) = filtfilt(b_A, a_A, x_ch(t))        # bandpass entire signal
    2. ptp(t) = max(x_bp[t−w:t+w]) − min(x_bp[t−w:t+w])   # local peak-to-peak
    3. artifact(t) = 1 if ptp(t) > thresh_µV, else 0
```

| Type | Channels | Bandpass | Threshold |
|------|----------|----------|-----------|
| Blink | AF7, AF8 | 1–40 Hz | 800 µV ptp |
| EMG | TP9, TP10 | 30–100 Hz | 800 µV ptp |

### 2.2 Design constraints

- **No adaptive thresholds**: param value IS the threshold. No z-scores, no SD floors.
- **Bandpass full signal first, then slice**: prevents edge ringing artefacts.
- **Viz reads from params**: `viz02.py` draws threshold lines at `±thresh_µV/2`.

### 2.3 Downstream impact

Artifact flags are stored but **not used** in the primary analysis manifest. The `primary` manifest includes all windows regardless of artifacts. The `artifact_exclude` manifest exists for sensitivity analysis only.

---

## Step 03 — Window labelling (detailed mathematical specification)

Step03 extracts fixed-duration analysis windows from the labelled time series, applies burst detection, and produces 4 sensitivity manifests.

### 3.1 Region detection

Contiguous SKIP and STAY regions are identified by scanning the `class` column of each interpolated DataFrame:

```
Let C(i) = class of sample i ∈ {SKIP, STAY}

A contiguous region R_k of label L is a maximal run:
    R_k = {i_start, i_start+1, ..., i_end}   where C(i) = L for all i ∈ R_k
                                               and C(i_start − 1) ≠ L, C(i_end + 1) ≠ L

Region time span:
    t_start(R_k) = lsl_timestamp[i_start]
    t_end(R_k)   = lsl_timestamp[i_end]
    duration(R_k) = t_end − t_start + 1/fs     (includes last sample's period)
```

For a single isolated A-press, `duration(R_SKIP) ≈ 6.0s` (±3.9ms due to 256 Hz quantisation).

### 3.2 Sliding window extraction

From each region, windows of `window_s = 3.0s` are extracted with `stride_s = 0.6s` stride (80% overlap):

```
Given region R_k with span S = duration(R_k):

    n_windows = round((S − window_s) / stride_s) + 1
    n_windows = max(1, n_windows)

    For wi = 0, 1, ..., n_windows − 1:
        t_wi = t_start(R_k) + wi × stride_s
        Window_wi = all samples in [t_wi, t_wi + window_s) that carry label L
```

The `round()` ensures deterministic window counts despite ±1-sample jitter from 256 Hz quantisation (see Resolved Issues below).

**Window start times are computed as `t_start + wi × stride_s`** (integer-multiplied offset), NOT by repeated `t += stride_s`, to avoid floating-point accumulation errors.

### 3.3 Window count for standard regions

For a single isolated 6s SKIP region:

```
S ≈ 6.0s
n = round((6.0 − 3.0) / 0.6) + 1 = round(5.0) + 1 = 6

Windows at offsets:  0.0, 0.6, 1.2, 1.8, 2.4, 3.0
Covering:           [0, 3), [0.6, 3.6), [1.2, 4.2), [1.8, 4.8), [2.4, 5.4), [3.0, 6.0)
```

For STAY regions, the same formula applies — typically producing many more windows since STAY regions are much longer.

### 3.4 Sample count validation

A window is only emitted if it contains ≥80% of the expected sample count:

```
expected_samples = fs × window_s = 256 × 3.0 = 768
threshold = int(expected_samples × 0.8) = 614

Window emitted iff |{samples in [t_wi, t_wi + window_s) with label L}| ≥ 614
```

This rejects edge windows where the region boundary cuts through the window.

### 3.5 Burst detection

When A-presses are close together (<6s apart), their ±3s SKIP regions merge into a single longer block. Step03 detects these merged blocks and counts actual A-presses per block.

```
For each SKIP block R_k:
    A_in_block = {t_A ∈ a_press_times : t_start(R_k) − 0.1 ≤ t_A ≤ t_end(R_k) + 0.1}
    n_A = |A_in_block|

    Expected single-press duration: D_single = 2 × half_window_s = 6.0s

    is_burst(R_k) = True   if n_A ≥ 2   OR   duration(R_k) > D_single + 1.0s
                  = False  otherwise
```

All windows from a burst block are flagged `is_burst_skip = True`.

**Example:** Two A-presses 4s apart at t=10 and t=14:
- SKIP regions: [7, 13] and [11, 17] → merged block [7, 17] = 10s
- Block contains 2 A-presses → burst
- All windows extracted from [7, 17] are flagged

### 3.6 Block statistics (for viz03)

For each SKIP block, step03 records:

```
block_stats = {
    region_idx, start_time, end_time, duration_s,
    n_a_presses,    ← actual count from a_press_times
    is_burst        ← True if merged block
}
```

Saved to `windows/primary/block_stats.csv` for use by viz03 (inter-A-press histogram, burst distribution).

### 3.7 Sensitivity manifests (4 variants)

Step03 runs four times with different inclusion/exclusion criteria:

| Manifest | Artifacts | Bursts | Purpose |
|----------|-----------|--------|---------|
| `primary` | included | included | Main analysis |
| `artifact_exclude` | excluded (>50% artifact) | included | Sensitivity |
| `burst_exclude` | included | excluded | Sensitivity |
| `artifact_exclude_burst_exclude` | both excluded | both excluded | Sensitivity |

Artifact fraction per window:

```
artifact_frac(W) = mean(artifact_any[i])   for i ∈ W
Window excluded if artifact_frac > 0.5
```

### 3.8 Output schema

Each manifest produces one pkl per participant:

```python
{
  'pid': int,
  'windows': [
    {
      'window_id': int,          # sequential index
      'start_time': float,       # LSL timestamp
      'end_time': float,         # start_time + window_s
      'label': int,              # 0 = SKIP, 1 = STAY
      'is_burst_skip': bool,     # True if from merged SKIP block
      'artifact_frac': float,    # fraction of samples with artifact flag
      'n_samples': int,          # actual sample count in window
      'data': np.float32 array,  # shape (n_samples, 28) — z-score features
    },
    ...
  ],
  'feature_names': [...],        # 28 feature column names
  'fs': float,                   # 256.0
  'a_press_times': [...],        # all A-press timestamps (for viz03)
}
```

---

## Step 04 — Balance & Split (Smart Stratified Temporal Split)

Step 04 is responsible for creating Cross-Validation (CV) splits that are both perfectly balanced (50/50 STAY/SKIP) and methodologically pure regarding data leakage.

### 4.1 The Data Leakage Problem (Why we need a Temporal Firewall)

A "lazy" random K-Fold split places Training windows and Testing windows milliseconds apart. Since EEG background noise is highly correlated over short timeframes, the model can "cheat" by memorizing the background noise of the training data to predict the adjacent test data. 

To prevent this, we must draw a hard **Temporal Firewall** across the entire recording (e.g., "Nothing before minute 10 touches anything after minute 13"). This firewall must apply universally to *all* classes simultaneously. If we were to separate STAY and SKIP into independent pools and split them separately, a Fold 1 Test SKIP window might occur at the exact same physical time as a Fold 1 Train STAY window. This would destroy the firewall and reintroduce severe data leakage.

### 4.2 The "Smart Stratified Temporal Split"

While a blanket temporal split prevents leakage, a "lazy" split (dividing the total session time by 5) fails if the participant's SKIP behaviors are concentrated at the end of the session—resulting in empty folds. 

The **Smart Stratified Temporal Split** solves both problems:
1. **Cumulative Stratification**: Instead of dividing the *time* by 5, it divides the *number of SKIP behaviors* by 5. It calculates the cumulative sum of chronologically ordered SKIP windows.
2. **Split Markers**: It places 4 temporal split markers exactly at the timestamps where 20%, 40%, 60%, and 80% of the SKIP windows have occurred. This guarantees every fold gets an equal proportion of the minority class.
3. **Impartial Firewall**: At each split marker, it enforces a strict `gap_s` (3.0 seconds) deletion zone. Any window (STAY or SKIP) touching this gap is permanently discarded. 

### 4.3 Mathematical Validation Mechanism

To guarantee mathematical purity ("Sauberkeit"), step04 runs an absolute validation assertion on the generated splits before saving them:
1. **Absolute Independence**: `Intersection(Train_IDs, Test_IDs) == 0`.
2. **Temporal Distance Proof**: It physically measures `min_distance = min(abs(train_time - test_time))` for every single train/test window pair in a fold.
3. It `assert min_distance >= gap_s - 0.1` (accounting for float precision). If any Train window is within 3 seconds of any Test window, the pipeline crashes.

---

## Resolved issues

### ✅ Floating-point window count jitter (step03)

**Problem:** At 256 Hz, a "6s" SKIP region is actually 5.992–6.004s depending on sample alignment. The old `while t + window_s <= region_end + 1/fs` check was sensitive to this — sometimes producing 5 windows, sometimes 6 for identical-length regions.

**Fix:** Replaced with deterministic `n_windows = round((region_span - window_s) / stride_s) + 1`. The `round()` absorbs ±1-sample jitter. Window start times are computed as `region_start + wi * stride_s` (integer-multiplied) to avoid float accumulation from repeated addition.

### ✅ Step05 KeyError: 'df'

**Cause:** Step05 read `notch_data['df']` but pkl stores `'dfs'` (list). Fixed to `pd.concat(notch_data['dfs'])`.

### ✅ Step03 SKIP windows were only 3s instead of extracting from full 6s region

**Cause:** Old code used `t += window_s` (3s non-overlapping stride) for SKIP windows while STAY used 0.6s stride. Fixed to use `stride_s` for both.

### ✅ Burst detection was broken (0 bursts detected)

**Cause:** Old `flag_burst_skips()` tried to read `keypress_A` from interpolated DataFrames, but step01 drops keypress columns during interpolation. Fixed by storing `a_press_times` in `df.attrs` in step01 and using those in step03.

### ✅ viz04 Panel 1 re-derived fold boundaries instead of reading actual splits

**Cause:** viz04 duplicated step04's fold-boundary math (`block_dur = ...`) instead of reading the produced `splits/P{pid}_splits.pkl`. If step04 hit the `block_dur < 1.0` fallback, the viz wouldn't reflect it. Fixed to load the actual split pkl and plot real window ranges per fold.

### ✅ viz04 Panel 2 assumed perfect 50/50 balance

**Cause:** `half = n_balanced_test_total / 2` assumed undersampling always achieved exact 50/50. When `n_min == 0` in `undersample_balance()`, the original unbalanced ids are returned, making the `/2` split fictional. Fixed to read actual `test_n_stay`/`test_n_skip` from the split pkl.

### ✅ viz05 Panel 1 hardcoded stat count

**Cause:** Schematic grid displayed literal `'4'` and title used `4 stats × 7 bands × 4 ch` instead of reading from `params['step05']['stats']`. Fixed to derive all counts from params and config.

---

## Viz02 artifact rates — thesis caveat

**What you see:** viz02 shows ~100% EMG artifact rates on nearly all participants (orange bars reaching 1.0). The right panel shows TP9 broadband signal routinely exceeding ±800µV.

**Why it looks alarming:** The Muse 2 temporal electrodes (TP9/TP10) sit behind the ears with minimal pressure. They produce high broadband noise that legitimately exceeds any reasonable EMG threshold. This is a **hardware limitation**, not a signal processing bug.

**What actually happens downstream:** The `primary` manifest (used for all main results) **includes** artifact-flagged windows. The `artifact_exclude` manifest exists for sensitivity analysis only. So the high artifact rates don't affect the main results pipeline — they're informational.

**Thesis framing recommendation:** Don't include the raw viz02 bar chart in the thesis — it invites "why did you proceed?" questions without adding analytical value. Instead:
- Mention in the Methods section that artifact detection was performed using peak-to-peak thresholds on bandpassed signals (blink: 1–40 Hz on AF7/AF8; EMG: 30–100 Hz on TP9/TP10)
- Note that temporal electrode artifact rates were high due to Muse 2 hardware contact limitations
- State that a sensitivity analysis was conducted with and without artifact exclusion (the 4 manifest variants)
- Report whether results differ meaningfully between `primary` and `artifact_exclude` manifests — if they don't, that's a strength ("results robust to artifact inclusion")

---

## Thesis mentions (must be disclosed)

Things encountered during development that require scientific disclosure:

### 1. Discrete sampling causes non-exact region boundaries

Step01 labels ±3.0s around each A-press, but at 256 Hz the actual SKIP region boundaries snap to the nearest sample (1/256 = 3.9ms resolution). This means a "6s region" is actually 5.992–6.004s. The window extraction uses `round()` to ensure deterministic window counts despite this jitter.

**Mention as:** "SKIP regions were defined as ±3.0s around each A-keypress. Due to the 256 Hz sampling rate, actual region boundaries were quantized to the nearest sample (±3.9ms)."

### 2. Burst-skip detection is structural, not parametric

Bursts are detected by whether SKIP regions merged (i.e., a contiguous SKIP block contains ≥2 A-presses), not by a fixed inter-press interval threshold. This is a natural consequence of the ±3s labelling: if two A-presses are <6s apart, their SKIP regions overlap and merge.

**Mention as:** "Burst-skip events were identified when consecutive A-keypresses occurred within 6s, causing their ±3s SKIP regions to merge into a single contiguous block."

### 3. Consumer-grade EEG limitations

The Muse 2 has 4 dry electrodes. Temporal channels (TP9/TP10) have known poor contact. High artifact rates on these channels are expected and documented in Muse literature. The pipeline includes sensitivity analysis across artifact inclusion/exclusion to quantify impact.

### 4. Overlapping windows introduce statistical dependence

With 80% overlap (0.6s stride on 3s windows), adjacent windows share 80% of their data. This is standard in EEG spectral analysis but must be disclosed. The temporal blocked cross-validation in step04 uses gap periods to mitigate data leakage between train/test splits.

**Mention as:** "Windows were extracted with 80% overlap (0.6s stride). Temporal block cross-validation with 3s gap periods was used to prevent data leakage from overlapping windows."

---

## Artifact detection — design rules (don't break these)

1. **No adaptive thresholds** — no SD floors, no z-scores. The param value IS the threshold.
2. **Unified mechanism** — both blink and EMG use peak-to-peak on bandpassed signal vs fixed µV.
3. **Viz reads from params** — `viz02.py` never hardcodes thresholds. Reads `params['step02']`, plots the bandpassed signal the detector evaluated, draws lines at `±thresh_uv/2`.
4. **Bandpass full signal first, then slice** — slicing then filtering causes edge ringing.

Current thresholds (`config.DEFAULT_PARAMS['step02']`): `blink_thresh_uv=800`, `emg_thresh_uv=800`.

---

## Participants

Defined in `config.py`:
- `ALL_PARTICIPANTS`: P4–P31
- `EXCLUDED_PARTICIPANTS`: [16, 19, 29]
- `INCLUDED_PARTICIPANTS`: 25 participants
- `SESSION_MAP`: PID → session folder. If a recording moves or is renamed, update here.

---

## Sanity checks

Run from `scripts/` with venv active. Edit the run timestamp to match your run.

**Verify SKIP region durations (should be ~6s for isolated A-presses):**
```python
import pickle, numpy as np
with open('../runs/run_20260520_065025/processed/nonotch/P4.pkl','rb') as f:
    d = pickle.load(f)
df = d['dfs'][0]
ts = df['lsl_timestamp'].values
classes = df['class'].values
skip_mask = np.array([c == 'SKIP' for c in classes])
skip_idx = np.where(skip_mask)[0]
breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
for block in np.split(skip_idx, breaks)[:5]:
    print(f"  dur={ts[block[-1]]-ts[block[0]]:.3f}s  samples={len(block)}")
# expect: ~5.996s, ~1537 samples per isolated block
```

**Verify window counts per SKIP block (should be uniform 6 for isolated blocks):**
```python
import pickle
with open('../runs/run_20260520_065025/windows/primary/P5.pkl','rb') as f:
    d = pickle.load(f)
skip_wins = sorted([w for w in d['windows'] if w['label'] == 0], key=lambda w: w['start_time'])
# group by region (windows within 0.6s stride of each other)
regions = [[skip_wins[0]]]
for w in skip_wins[1:]:
    if w['start_time'] - regions[-1][-1]['start_time'] <= 0.7:
        regions[-1].append(w)
    else:
        regions.append([w])
for i, r in enumerate(regions[:10]):
    dur = r[-1]['end_time'] - r[0]['start_time']
    print(f"  region {i}: {len(r)} windows, span={dur:.1f}s")
# expect: 6 windows per isolated region
```

**Verify pkl schema:**
```python
import pickle
with open('../runs/run_20260520_065025/processed/nonotch/P6.pkl','rb') as f:
    d = pickle.load(f)
print(d.keys())           # pid, fs, dfs, feature_names, baseline_stats, dropouts
print(type(d['dfs']))     # <class 'list'>
print(d['dfs'][0].attrs)  # n_a_presses, a_press_times
```

---

## When something breaks

1. Confirm venv is active: `which python3` should point inside `05-12_.../venv/`
2. Check `parameters.json` in your run dir — that's what actually ran
3. Resume from broken step: `python3 run.py --resume <timestamp>` — don't re-run steps 1–4 unnecessarily
4. Repeat the last completed step if you changed its script: `python3 run.py --resume <timestamp> --restep`
5. Before touching any pkl: `pickle.load` it and print `.keys()` and `type(d['dfs'])`
6. **Never assume `'df'` exists** — the key is always `'dfs'` (list)
7. Check `df.attrs` for `a_press_times` — step03 depends on this from step01