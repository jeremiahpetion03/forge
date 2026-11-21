import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        # Statistics over observed values only. Computing them AFTER imputing
        # would pull every column toward its median and shrink the std.
        self.median = np.nanmedian(num, axis=0)
        self.mean = np.nanmean(num, axis=0)
        std = np.nanstd(num, axis=0)
        self.std = np.where(std == 0, 1.0, std)          # constant column: no 0/0
        self.categories = [np.unique(cat[:, j]) for j in range(cat.shape[1])]  # sorted
        return self

    def transform(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        filled = np.where(np.isnan(num), self.median, num)  # new array; input untouched
        cols = [(filled - self.mean) / self.std]
        for j, cats in enumerate(self.categories):
            # Unseen values match no training category, so their row is all zeros.
            cols.append((cat[:, j][:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(cols)
