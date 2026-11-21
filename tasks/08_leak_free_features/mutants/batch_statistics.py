# Learns nothing in fit and recomputes statistics from each transform batch:
# the classic leakage / train-serve skew bug.
import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        self.categories = [np.unique(np.asarray(cat)[:, j]) for j in range(np.asarray(cat).shape[1])]
        return self

    def transform(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        med, mean, std = np.nanmedian(num, 0), np.nanmean(num, 0), np.nanstd(num, 0)
        std = np.where(std == 0, 1.0, std)
        cols = [(np.where(np.isnan(num), med, num) - mean) / std]
        for j, cats in enumerate(self.categories):
            cols.append((cat[:, j][:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(cols)
