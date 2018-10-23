from __future__ import print_function

import numpy as np

from utils import f1_scores_threshold


y_true = np.array([1, 1, 0, 0])
y_prab = np.array([0.8, 0.5, 0.2, 0.1])
thresholds = np.linspace(0, 1, 100)

f1_scores = f1_scores_threshold(y_true, y_prab, thresholds)

print(thresholds)
print(f1_scores)

idx = np.argmax(f1_scores)
print(idx)
print(thresholds[idx])