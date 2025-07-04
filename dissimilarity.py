from itertools import combinations

from paramhandling.paramhandler import parcheck, get_nparrays, get_classes

import numpy as np


def dissimilarity(
    Q: np.ndarray, y: np.ndarray, factor_h: float, factor_k: float, classes: np.ndarray | None = None
) -> float:
    """
    Calculates a dissimilarity metric based on a generalization of the method
    proposed in "Width optimization of RBF kernels for binary classiﬁcation of support
    vector machines: A density estimation-based approach" by Menezes et al. (2019).

    The function measures the separability between classes in a given similarity space.

    Parameters:
        Q (np.ndarray): An (M, N) similarity matrix where M is the number of samples
                        and N is the number of classes. Q[i, c] is the similarity
                        of sample i to class c. These rows are treated as points
                        in an N-dimensional space.
        y (np.ndarray): An (M,) array of labels, where y[i] is the integer class
                        label for sample i.
        factor_h (float): A scaled factor from the RBF kernel bandwidth parameter.
        factor_k (float): A scaled factor from the number of nearest neighbors used in
                          the sparse RBF kernel.
        classes (np.ndarray | None): The complete list of unique class labels. If provided,
                                     it's used to define the class space. If None,
                                     classes are inferred from y.

    Returns:
        float: The dissimilarity score.

    Raises:
        TypeError: If Q or y cannot be converted to numpy arrays.
        ValueError: If Q is not a 2D array, y is not a 1D array, or if the number of
                    samples in Q and y do not match.
        ValueError: If the number of columns in Q does not match the number of
                    unique classes in y.
    """

    parcheck(Q, y, factor_h, factor_k, classes)
    Q, y = get_nparrays(Q, y)
    unique_labels, n_classes = get_classes(y, classes)

    # --- Step 1: Compute the Class Similarity Matrix S ---
    # S[i, j] will be the average similarity of samples from class `i` to class `j`.
    S = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, label in enumerate(unique_labels):
        mask = y == label
        # Only compute the mean if there are samples for the class.
        if np.any(mask):
            S[i, :] = np.mean(Q[mask], axis=0)
        # Otherwise, S[i, :] remains a vector of zeros.

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
    # Those factors are consistently yielding good results. Please don't change!
    return float(mean_dissim - std_dissim) * (1 - factor_h) * (1 - factor_k)
