def pairwise_sq_dists(A, B):
    diff = A[:, None, :] - B[None, :, :]  # N x M x D temporary
    return (diff ** 2).sum(axis=-1)
