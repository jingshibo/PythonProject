import numpy as np


def _to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got {a.shape}")
    return a


def _align_shapes(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Your arrays might be (T,C) or (C,T). Try transpose if needed.
    """
    X = _to_2d(X)
    Y = _to_2d(Y)
    if X.shape == Y.shape:
        return X, Y
    if X.T.shape == Y.shape:
        return X.T, Y
    if X.shape == Y.T.shape:
        return X, Y.T
    if X.T.shape == Y.T.shape:
        return X.T, Y.T
    raise ValueError(f"Cannot align shapes X{X.shape} vs Y{Y.shape}")


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Spatiotemporal cosine similarity:
      <X,Y>_F / (||X||_F ||Y||_F)
    """
    X, Y = _align_shapes(X, Y)
    x = X.reshape(-1)
    y = Y.reshape(-1)
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + eps
    return float(np.dot(x, y) / denom)


def pairwise_mean_similarity(listA, listB) -> float:
    """
    Mean cosine similarity over ALL pairs (i,j).
    """
    if len(listA) == 0 or len(listB) == 0:
        return float("nan")

    scores = []
    for a in listA:
        for b in listB:
            scores.append(cosine_similarity_matrix(a, b))
    return float(np.mean(scores))


def compare_datasets_pairwise_mean(dsA: dict, dsB: dict, keys):
    """
    For each key:
      S_k = mean_{i,j} CosSim(A_i, B_j)
    Overall:
      mean over keys.
    """
    per_key = {}
    for k in keys:
        if k not in dsA or k not in dsB:
            per_key[k] = float("nan")
            continue
        per_key[k] = pairwise_mean_similarity(dsA[k], dsB[k])

    # average over keys that are not nan
    vals = [v for v in per_key.values() if np.isfinite(v)]
    overall = float(np.mean(vals)) if vals else float("nan")
    return overall, per_key


