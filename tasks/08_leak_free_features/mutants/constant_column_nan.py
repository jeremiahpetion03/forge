# Forgets the std == 0 guard: a constant column becomes 0/0 = NaN.
import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        self.median = np.nanmedian(num, axis=0)
        self.mean = np.nanmean(num, axis=0)
        self.std = np.nanstd(num, axis=0)
        self.categories = [np.unique(cat[:, j]) for j in range(cat.shape[1])]
        return self

    def transform(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols = [(np.where(np.isnan(num), self.median, num) - self.mean) / self.std]
        for j, cats in enumerate(self.categories):
            cols.append((cat[:, j][:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(cols)
