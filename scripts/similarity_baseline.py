import numpy as np


def levenshtein(str1: str, str2: str, whole_matrix: bool = False) -> int:
    """Naive Levenshtein edit distance computation using NumPy. Quadratic complexity in time and space."""
    rows = len(str1) + 1
    cols = len(str2) + 1
    distance_matrix = np.zeros((rows, cols), dtype=int)
    distance_matrix[0, :] = np.arange(cols)
    distance_matrix[:, 0] = np.arange(rows)
    for i in range(1, rows):
        for j in range(1, cols):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance_matrix[i, j] = min(
                distance_matrix[i - 1, j] + 1,  # Deletion
                distance_matrix[i, j - 1] + 1,  # Insertion
                distance_matrix[i - 1, j - 1] + cost,  # Substitution
            )

    if whole_matrix:
        return distance_matrix
    return distance_matrix[-1, -1]


if __name__ == "__main__":
    print(levenshtein("aaaba", "aaaca", True))
