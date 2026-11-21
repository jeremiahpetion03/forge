import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        return self

    def transform(self, num, cat):
        # Statistics come from the batch being transformed: leakage, and a row's
        # features depend on which other rows are in the batch.
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        med = np.nanmedian(num, axis=0)
        filled = np.where(np.isnan(num), med, num)
        mean, std = np.nanmean(num, axis=0), np.nanstd(num, axis=0)
        std = np.where(std == 0, 1.0, std)
        cols = [(filled - mean) / std]
        for j in range(cat.shape[1]):
            cats = np.unique(cat[:, j])
            cols.append((cat[:, j][:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(cols)
