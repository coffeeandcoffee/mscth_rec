import pickle, numpy as np
from scipy import signal
with open('../runs/run_20260518_145744/processed/notch/P6.pkl', 'rb') as f:
    d = pickle.load(f)
sig = d['dfs'][0]['TP9'].values
f, Pxx = signal.welch(sig, fs=d['fs'], nperseg=1024)
# print power at 49-51Hz
mask = (f > 49) & (f < 51)
print(f"Power at 50Hz: {Pxx[mask].mean():.2f}")
print(f"Power at 30Hz: {Pxx[(f>29)&(f<31)].mean():.2f}")
print(f"Power at 70Hz: {Pxx[(f>69)&(f<71)].mean():.2f}")

# 1. Confirm the raw channel is identical between branches
import pickle, numpy as np
with open('../runs/run_20260518_145744/processed/nonotch/P6.pkl','rb') as f:
    nn = pickle.load(f)
with open('../runs/run_20260518_145744/processed/notch/P6.pkl','rb') as f:
    n = pickle.load(f)
print("TP9 channel identical?", np.allclose(nn['dfs'][0]['TP9'], n['dfs'][0]['TP9']))

# 2. Check 50Hz power in a band FEATURE (which IS notched)
# low_gamma is 30-40Hz so it shouldn't show much 50Hz, but compare them
import numpy as np
nn_hg = nn['dfs'][0]['TP9_high_gamma'].values
n_hg  = n['dfs'][0]['TP9_high_gamma'].values
print(f"nonotch high_gamma mean abs: {np.mean(np.abs(nn_hg)):.4f}")
print(f"notch   high_gamma mean abs: {np.mean(np.abs(n_hg)):.4f}")