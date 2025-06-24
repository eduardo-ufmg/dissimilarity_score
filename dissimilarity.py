import numpy as np

def dissimilarity(Q: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a multi-class dissimilarity measure from a similarity matrix Q and labels y.
    
    Parameters
    ----------
    Q : np.ndarray, shape (N, C)
        Similarity matrix: Q[i, j] is the similarity of sample i to class j.
    y : np.ndarray, shape (N,)
        Integer labels in {0, 1, ..., C-1}. Each class index must appear at least once.
    
    Returns
    -------
    S : float
        Aggregated dissimilarity: sum over all 0 <= k < l < C of
          ||V_k - V_l|| * (V_k · V_l) / (||V_k|| * ||V_l||),
        where V_k = mean of Q[i,:] over i with y[i] == k.
    """
    # Ensure inputs have compatible shapes
    if Q.ndim != 2:
        raise ValueError("Q must be a 2D array of shape (N, C).")
    N, C = Q.shape
    if y.ndim != 1 or y.shape[0] != N:
        raise ValueError("y must be a 1D array of length N = Q.shape[0].")
    
    # Convert y to integer array
    y = np.asarray(y, dtype=np.int64)
    # Check label range
    if np.any(y < 0) or np.any(y >= C):
        raise ValueError("Labels y must be integers in [0, C-1], matching Q's columns.")
    
    # Count samples per class
    counts = np.bincount(y, minlength=C)
    if np.any(counts == 0):
        # Some class has no samples
        missing = np.where(counts == 0)[0]
        raise ValueError(f"Each class 0..{C-1} must appear at least once; missing: {missing.tolist()}")
    
    # Compute sum of Q rows per class, shape (C, C)
    # sum_Q[k] = sum of Q[i] over i with y[i] == k
    sum_Q = np.zeros((C, C), dtype=Q.dtype)
    # Using np.add.at for efficiency in C loops
    np.add.at(sum_Q, y, Q)
    # Compute class-mean vectors V: shape (C, C)
    # V[k] = (1 / counts[k]) * sum_Q[k]
    # Broadcasting counts[:,None]
    V = sum_Q / counts[:, None]
    
    # Compute Gram matrix of V: G[k, l] = V[k]·V[l]
    G = V @ V.T  # shape (C, C)
    
    # Compute squared norms of each V[k]
    norm_sq = np.diag(G)  # shape (C,)
    # Compute norms, guard against zero
    # Since counts>0 and similarities nonnegative, norms should be >0; but guard anyway:
    norms = np.sqrt(norm_sq)
    
    # Prepare outer product of norms for cosine
    # To avoid division by zero, we will mask later; create outer:
    norm_outer = norms[:, None] * norms[None, :]
    
    # Compute pairwise cosines, safe: set cos=0 where norm_outer == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cosines = np.divide(G, norm_outer, out=np.zeros_like(G), where=(norm_outer > 0))
    
    # Compute pairwise squared distances: ||V_k - V_l||^2 = norm_sq[k] + norm_sq[l] - 2*G[k,l]
    # Use broadcasting:
    D2 = norm_sq[:, None] + norm_sq[None, :] - 2 * G
    # Numerical errors might make tiny negatives; clip to >=0
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    
    # Compute matrix of d_{k,l} = D[k,l] * cosines[k,l]
    # We only need sum over k<l
    M = D * cosines
    
    # Sum over upper triangle k<l
    # Using np.triu_indices
    iu = np.triu_indices(C, k=1)
    S = np.sum(M[iu])
    
    return float(S)
