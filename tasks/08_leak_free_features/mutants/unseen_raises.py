# Encodes categories by dictionary lookup, so a category first seen in
# production raises KeyError instead of encoding as all zeros.
import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        self.median = np.nanmedian(num, axis=0)
        self.mean = np.nanmean(num, axis=0)
        std = np.nanstd(num, axis=0)
        self.std = np.where(std == 0, 1.0, std)
        self.index = [{c: i for i, c in enumerate(np.unique(cat[:, j]))} for j in range(cat.shape[1])]
        return self

    def transform(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        cols = [(np.where(np.isnan(num), self.median, num) - self.mean) / self.std]
        for j, idx in enumerate(self.index):
            oh = np.zeros((len(cat), len(idx)))
            oh[np.arange(len(cat)), [idx[v] for v in cat[:, j]]] = 1.0
            cols.append(oh)
        return np.hstack(cols)
