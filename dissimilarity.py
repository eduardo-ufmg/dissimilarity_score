from itertools import combinations

import numpy as np


def dissimilarity(
    Q: np.ndarray, y: np.ndarray, factor_h: float, factor_k: int
) -> float:
    """
    Calculates a dissimilarity metric based on a generalization of the method
    proposed in "Width optimization of RBF kernels for binary classiﬁcation of support
    vector machines: A density estimation-based approach" by Menezes et al. (2019).

    The function measures the separability between classes in a given similarity space.

    Parameters:
        Q (np.ndarray): A 2D numpy array of shape (n_samples, n_classes). Q[i, j]
                        represents the similarity of sample `i` to class `j`.
        y (np.ndarray): A 1D numpy array of shape (n_samples,) containing the true
                        class labels for each sample. Labels are expected to be
                        integers from 0 to n_classes-1.
        factor_h (float): A scaled factor from the RBF kernel bandwidth parameter.
        factor_k (int): A scaled factor from the number of nearest neighbors used in
                        the sparse RBF kernel.

    Returns:
        float: The dissimilarity score.

    Raises:
        TypeError: If Q or y cannot be converted to numpy arrays.
        ValueError: If Q is not a 2D array, y is not a 1D array, or if the number of
                    samples in Q and y do not match.
        ValueError: If the number of columns in Q does not match the number of
                    unique classes in y.
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
        raise ValueError(
            f"The number of columns in Q ({Q.shape[1]}) must match the "
            f"number of unique classes in y ({n_classes})."
        )

    # --- Step 1: Compute the Class Similarity Matrix S ---
    # S[i, j] will be the average similarity of samples from class `i` to class `j`.
    S = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, label in enumerate(unique_labels):
        mask = y == label
        S[i, :] = np.mean(Q[mask], axis=0)

    # Each row of S is now a vector Vi, representing the similarity profile of class i.
    V_vectors = S

    # --- Step 2: Calculate pairwise dissimilarity for all unique class pairs ---
    all_dissimilarities = []

    # Use itertools.combinations to efficiently get all unique pairs of class indices.
    for i, j in combinations(range(n_classes), 2):
        Vi = V_vectors[i]
        Vj = V_vectors[j]

        # Calculate the components of the dissimilarity function from the paper.
        euclidean_dist = np.linalg.norm(Vi - Vj)
        norm_vi = np.linalg.norm(Vi)
        norm_vj = np.linalg.norm(Vj)

        # Handle the edge case where a class similarity vector has zero magnitude.
        if norm_vi == 0 or norm_vj == 0:
            pairwise_dissim = 0.0
        else:
            dot_product = np.dot(Vi, Vj)
            # The cosine of the angle between the two class-similarity vectors
            cosine_similarity = dot_product / (norm_vi * norm_vj)
            pairwise_dissim = euclidean_dist * cosine_similarity

        # Add the calculated dissimilarity to our list
        all_dissimilarities.append(pairwise_dissim)

    # --- Step 3: Calculate the final score ---
    # Convert the list to a NumPy array for efficient vectorized calculations.
    dissim_array = np.array(all_dissimilarities)

    # Calculate the mean and standard deviation of all pairwise dissimilarities.
    mean_dissim = np.mean(dissim_array)
    std_dissim = np.std(dissim_array)

    # Return the mean minus the standard deviation.
    return float(mean_dissim - std_dissim) * factor_h * factor_k
