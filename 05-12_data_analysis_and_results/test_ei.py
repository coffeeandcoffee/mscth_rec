import pickle, glob, numpy as np
all_ei = []
all_labels = []
for pkl_file in glob.glob("runs/run_20260611_220844/features/P*.pkl"):
    data = pickle.load(open(pkl_file, 'rb'))
    all_ei.extend(data['ei_values'])
    all_labels.extend(data['labels'])

all_ei = np.array(all_ei)
all_labels = np.array(all_labels)

unique_labels = np.unique(all_labels)
ei_0 = all_ei[all_labels == unique_labels[0]]
ei_1 = all_ei[all_labels == unique_labels[1]]

mean0 = np.mean(ei_0)
mean1 = np.mean(ei_1)
var0 = np.var(ei_0)
var1 = np.var(ei_1)
pooled_std = np.sqrt((var0 + var1) / 2)
ei_d = np.abs(mean1 - mean0) / (pooled_std + 1e-9)
print(f"Mean0: {mean0:.4f}, Mean1: {mean1:.4f}")
print(f"Var0: {var0:.4f}, Var1: {var1:.4f}")
print(f"Pooled STD: {pooled_std:.4f}")
print(f"Cohen's d: {ei_d:.5f}")
