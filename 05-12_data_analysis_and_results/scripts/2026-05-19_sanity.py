import pickle
with open('../runs/run_20260519_064108/windows/primary/P4.pkl', 'rb') as f:
    d = pickle.load(f)

skip_wins = [w for w in d['windows'] if w['label'] == 0]
print("First 5 SKIP windows:")
for w in skip_wins[:5]:
    dur = w['end_time'] - w['start_time']
    print(f"  start={w['start_time']:.3f}  end={w['end_time']:.3f}  dur={dur:.3f}s")

# -----------

with open('../runs/run_20260519_064108/windows/primary/P4.pkl', 'rb') as f:
    d = pickle.load(f)

wins = sorted(d['windows'], key=lambda w: w['start_time'])

print("First 10 windows in time order:")
for w in wins[:10]:
    print(f"  {'SKIP' if w['label']==0 else 'STAY'}  "
          f"start={w['start_time']:.3f}  end={w['end_time']:.3f}  "
          f"dur={w['end_time']-w['start_time']:.3f}s")

print("\nGaps between consecutive windows:")
for i in range(1, min(len(wins), 10)):
    gap = wins[i]['start_time'] - wins[i-1]['end_time']
    print(f"  [{wins[i-1]['label']}→{wins[i]['label']}]  gap={gap:.3f}s")

# ----------

with open('../runs/run_20260519_064108/windows/primary/P4.pkl', 'rb') as f:
    d = pickle.load(f)

wins = sorted(d['windows'], key=lambda w: w['start_time'])

print("First 10 windows:")
for w in wins[:10]:
    dur = w['end_time'] - w['start_time']
    label = 'SKIP' if w['label'] == 0 else 'STAY'
    print(f"  {label}  start={w['start_time']:.3f}  end={w['end_time']:.3f}  dur={dur:.3f}s")