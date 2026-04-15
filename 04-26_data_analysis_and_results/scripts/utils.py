import numpy as np
from scipy import signal

# Strict Constants from legacy pipeline
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
FREQUENCY_BANDS = [
    ('delta', 1, 4),
    ('theta', 4, 8),
    ('alpha', 8, 13),
    ('beta', 13, 30),
    ('low_gamma', 30, 40),
    ('high_gamma', 40, 60),
    ('very_high', 60, 100),
]

def apply_notch_filter(data, fs, notch_freq=50.0, quality_factor=30.0):
    """Apply notch filter to remove power line interference."""
    if len(data) < 20:
        return data
    try:
        b, a = signal.iirnotch(notch_freq, quality_factor, fs)
        return signal.filtfilt(b, a, data)
    except Exception:
        return data

def get_bandpass_filter(fs, low_freq, high_freq):
    """Return coefficients for a 4th-order Butterworth bandpass filter."""
    nyquist = fs / 2
    low = max(low_freq / nyquist, 0.01)
    high = min(high_freq / nyquist, 0.99)
    if low >= high:
        return None, None
    return signal.butter(4, [low, high], btype='band')

def extract_band_power(data, fs, low_freq, high_freq):
    """Legacy bandpass extraction returning scalar analytical amplitude."""
    b, a = get_bandpass_filter(fs, low_freq, high_freq)
    if b is None or len(data) < 20:
        return np.zeros_like(data)
    try:
        return signal.filtfilt(b, a, data)
    except Exception:
        return np.zeros_like(data)

def compute_100s_baseline_power(df_raw, notch_freq=None):
    """
    Computes absolute baseline Power means per band for Baseline Normalization.
    Mathematical derivation constraint: P(t) = Amplitude(t)^2. P_rel = P / mean(P_baseline)
    
    Returns standard dictionary mapping Feature_Name -> mean scalar squared Power
    """
    time_diffs = np.diff(df_raw['lsl_timestamp'].values)
    fs = 1.0 / np.median(time_diffs)
    
    # Filter explicitly to the 100s baseline
    df_base = df_raw[df_raw['class'].isin(['baseline_1', 'baseline_2'])].copy()
    if len(df_base) < 100:
        # Failsafe if baseline is corrupted
        return None, None
        
    baseline_power_means = {}
    
    for ch in EEG_CHANNELS:
        raw_signal = df_base[ch].values
        if notch_freq:
            raw_signal = apply_notch_filter(raw_signal, fs, notch_freq)
            
        for band_name, low_freq, high_freq in FREQUENCY_BANDS:
            # 1. Isolate the exact frequency band amplitude over the resting state
            amp_baseline = extract_band_power(raw_signal, fs, low_freq, high_freq)
            # 2. Derive raw Electrical Power natively squaring scalar voltage bounds
            power_baseline = amp_baseline ** 2
            # 3. Calculate absolute geometric arithmetic mean bounds resolving Baseline 
            mean_base_power = np.mean(power_baseline)
            
            # Anti-zero bounds failsafe
            if mean_base_power < 1e-12:
                mean_base_power = 1e-12
                
            feature_name = f"{ch}_{band_name}"
            baseline_power_means[feature_name] = mean_base_power
            
    return baseline_power_means, fs

def build_normalized_features(df_raw, baseline_power_means, notch_freq=None):
    """
    Computes Baseline Normalized continuous Relative Power vectors over full session topology.
    Returns: DataFrame natively identical with 28 newly mapped normalized channels appended.
    """
    time_diffs = np.diff(df_raw['lsl_timestamp'].values)
    fs = 1.0 / np.median(time_diffs)
    
    df_features = df_raw.copy()
    feature_names = []
    
    for ch in EEG_CHANNELS:
        raw_signal = df_raw[ch].values
        if notch_freq:
            raw_signal = apply_notch_filter(raw_signal, fs, notch_freq)
            
        for band_name, low_freq, high_freq in FREQUENCY_BANDS:
            feature_name = f"{ch}_{band_name}"
            feature_names.append(feature_name)
            
            if baseline_power_means:
                # 1. Exact raw topological band separation
                amp_vector = extract_band_power(raw_signal, fs, low_freq, high_freq)
                # 2. Square absolute raw topological vector values mapped into Power dimension
                power_vector = amp_vector ** 2
                # 3. Globally normalize by scalar absolute 100s Resting Means resolving directly to Relative Power Change
                p_rel_vector = power_vector / baseline_power_means[feature_name]
                df_features[feature_name] = p_rel_vector
            else:
                # Failsafe backward compatibility strictly keeping unnormalized amplitudes
                df_features[feature_name] = extract_band_power(raw_signal, fs, low_freq, high_freq)
                
    return df_features, feature_names
