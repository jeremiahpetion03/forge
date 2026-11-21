# Imputes the training data first, then computes mean and std on the filled
# column. The std shrinks toward zero as missingness grows.
import numpy as np


class FeaturePipeline:
    def fit(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        self.median = np.nanmedian(num, axis=0)
        filled = np.where(np.isnan(num), self.median, num)
        self.mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)
        self.categories = [np.unique(cat[:, j]) for j in range(cat.shape[1])]
        return self

    def transform(self, num, cat):
        num = np.asarray(num, dtype=np.float64)
        cat = np.asarray(cat)
        cols = [(np.where(np.isnan(num), self.median, num) - self.mean) / self.std]
        for j, cats in enumerate(self.categories):
            cols.append((cat[:, j][:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(cols)
