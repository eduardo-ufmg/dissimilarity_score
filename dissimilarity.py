import numpy as np
from itertools import combinations

def dissimilarity(Q: np.ndarray, y: np.ndarray, h: float, k: int) -> float:
    """
    Calculates a dissimilarity metric based on a generalization of the method
    proposed in "Width optimization of RBF kernels for binary classiﬁcation of support
    vector machines: A density estimation-based approach" by Menezes et al. (2019).

    The function measures the separability between classes in a given similarity space.
    A lower value indicates a pair of classes that are less separable (more similar),
    representing the "weakest link" in the multi-class separation.

    Parameters:
        Q (np.ndarray): A 2D numpy array of shape (n_samples, n_classes). Q[i, j]
                        represents the similarity of sample `i` to class `j`.
        y (np.ndarray): A 1D numpy array of shape (n_samples,) containing the true
                        class labels for each sample. Labels are expected to be
                        integers from 0 to n_classes-1.
        h (float): The bandwidth parameter for the RBF kernel. Used as regularization
                   to control the smoothness of the similarity space.
        k (int): The number of nearest neighbors to consider in the sparse RBF kernel.
                 Used as regularization to control the sparsity of the similarity space.

    Raises:
        TypeError: If Q or y cannot be converted to numpy arrays.
        ValueError: If Q is not a 2D array, y is not a 1D array, or if the number of samples in Q and y do not match.
        ValueError: If the number of columns in Q does not match the number of unique classes in y.

    Returns:
        float: The minimum pairwise dissimilarity score among all unique pairs of
               classes. Returns 0.0 if there are fewer than 2 classes. A higher
               score implies better separability.
    """
    # Ensure inputs are numpy arrays for optimized operations
    try:
        Q = np.asanyarray(Q, dtype=np.float64)
        y = np.asanyarray(y, dtype=int)
    except (ValueError, TypeError):
        raise TypeError("Inputs Q and y must be convertible to numpy arrays.")

    if Q.ndim != 2:
        raise ValueError("Input Q must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("Input y must be a 1D array.")
    if Q.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in Q and y must be the same.")

    # Find unique classes present in the data
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    # If there's only one class or no classes, dissimilarity is not applicable.
    if n_classes < 2:
        return 0.0

    # The number of columns in Q should match the number of classes.
    if Q.shape[1] != n_classes:
         raise ValueError(f"The number of columns in Q ({Q.shape[1]}) must match the number of unique classes in y ({n_classes}).")


    # --- Step 1: Compute the Class Similarity Matrix S ---
    # S[i, j] will be the average similarity of samples from class `i` to class `j`.
    # This is a vectorized implementation that avoids explicit Python loops over samples.
    S = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, label in enumerate(unique_labels):
        # Create a boolean mask to select samples belonging to the current class
        mask = (y == label)
        # For class i, calculate its average similarity to all other classes j.
        # This computes a single row of the S matrix.
        S[i, :] = np.mean(Q[mask], axis=0)

    # Each row of S is now a vector Vi, representing the similarity profile of class i.
    V_vectors = S

    # --- Step 2: Calculate pairwise dissimilarity for all unique class pairs ---
    min_dissimilarity = np.inf

    # Use itertools.combinations to efficiently get all unique pairs of class indices.
    for i, j in combinations(range(n_classes), 2):
        Vi = V_vectors[i]
        Vj = V_vectors[j]

        # Calculate the components of the dissimilarity function from the paper.
        # D(Vi, Vj) = ||Vi - Vj|| * cos(Vi, Vj)
        euclidean_dist = np.linalg.norm(Vi - Vj)
        norm_vi = np.linalg.norm(Vi)
        norm_vj = np.linalg.norm(Vj)

        # Handle the edge case where a class similarity vector has zero magnitude.
        # This can happen if a class has no similarity to any other class,
        # which is unlikely but possible. In this case, cosine is undefined.
        if norm_vi == 0 or norm_vj == 0:
            pairwise_dissim = 0.0
        else:
            dot_product = np.dot(Vi, Vj)
            # The cosine of the angle between the two class-similarity vectors
            cosine_similarity = dot_product / (norm_vi * norm_vj)
            pairwise_dissim = euclidean_dist * cosine_similarity

        # Update the minimum dissimilarity found so far.
        if pairwise_dissim < min_dissimilarity:
            min_dissimilarity = pairwise_dissim

    # If no pairs were evaluated (which shouldn't happen with n_classes >= 2), return 0.
    return min_dissimilarity if min_dissimilarity != np.inf else 0.0

