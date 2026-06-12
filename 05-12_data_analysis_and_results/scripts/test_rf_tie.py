import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.random.randn(100, 5)
y = np.array([0]*50 + [1]*50)
clf = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42)
clf.fit(X, y)
preds = clf.predict(X)
print("Unique predictions:", np.unique(preds, return_counts=True))
