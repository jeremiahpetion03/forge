# Correct features, but fills NaN in the caller's array in place, silently
# erasing which values were missing.
import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        self.median = np.nanmedian(num, axis=0)
        self.mean = np.nanmean(num, axis=0)
        std = np.nanstd(num, axis=0)
        self.std = np.where(std == 0, 1.0, std)
        self.categories = [np.unique(cat[:, j]) for j in range(cat.shape[1])]
        return self

    def transform(self, num, cat):
        num = np.asarray(num, dtype=np.float64)  # same object for float64 input
        cat = np.asarray(cat)
        rows, cols_ = np.nonzero(np.isnan(num))
        num[rows, cols_] = self.median[cols_]
        cols = [(num - self.mean) / self.std]
        for j, cats in enumerate(self.categories):
            cols.append((cat[:, j][:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(cols)
