import numpy as np

def is_asset_correlation_matrix(C, tol=1e-8):
    """
    Validates whether a matrix C is a valid asset correlation matrix based on:
    1. Symmetry
    2. Unit diagonal
    3. Positive semi-definiteness

    Parameters:
    C (list or numpy.ndarray): The matrix to evaluate.
    tol (float): Tolerance for floating-point comparisons.

    Returns:
    bool: True if it is a valid correlation matrix, False otherwise.
    dict: A dictionary containing the results of each specific check.
    """
    # Convert input to a numpy array for vector operations
    C = np.asarray(C, dtype=float)

    # Check if the matrix is 2D and square
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        return False, {"Error": "Matrix is not square."}

    # 1. Check if symmetric: C.T == C
    # np.allclose compares arrays element-wise within the given tolerance
    is_symmetric = np.allclose(C, C.T, atol=tol)

    # 2. Check if unit diagonal: C[i,i] == 1
    # np.diag extracts the diagonal, we compare it against an array of 1s
    is_unit_diag = np.allclose(np.diag(C), 1.0, atol=tol)

    # 3. Check if positive semi-definite (PSD)
    # A symmetric matrix is PSD if all its eigenvalues are >= 0.
    # We use np.linalg.eigvalsh as it is optimized for symmetric/Hermitian matrices.
    if is_symmetric:
        eigenvalues = np.linalg.eigvalsh(C)
        # We check >= -tol instead of >= 0 to account for numerical precision errors
        # that might make a 0 eigenvalue appear as a very tiny negative number.
        is_psd = np.all(eigenvalues >= -tol)
    else:
        # If it's not symmetric, eigvalsh might not be appropriate, and it's
        # inherently not a valid correlation matrix anyway.
        is_psd = False
        eigenvalues = None

    # Compile the detailed results
    details = {
        "is_square": True,
        "is_symmetric": bool(is_symmetric),
        "is_unit_diagonal": bool(is_unit_diag),
        "is_positive_semi_definite": bool(is_psd)
    }

    # Matrix is valid only if all three conditions are met
    is_valid = is_symmetric and is_unit_diag and is_psd

    return is_valid, details

# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # Example 1: A valid 3x3 correlation matrix
    valid_matrix = [
        [ 1.0,  0.8, -0.2],
        [ 0.8,  1.0,  0.1],
        [-0.2,  0.1,  1.0]
    ]

    is_valid, details = is_asset_correlation_matrix(valid_matrix)
    print("Example 1 (Valid Matrix):")
    print(f"Is valid? {is_valid}")
    print(f"Details: {details}\n")

    # Example 2: Invalid matrix (Not Positive Semi-Definite)
    # Correlation between A/B is 0.9, B/C is 0.9, but A/C is -0.9.
    # This is mathematically impossible, making the matrix not PSD.
    invalid_psd_matrix = [
        [ 1.0,  0.9, -0.9],
        [ 0.9,  1.0,  0.9],
        [-0.9,  0.9,  1.0]
    ]

    is_valid, details = is_asset_correlation_matrix(invalid_psd_matrix)
    print("Example 2 (Invalid Matrix - Impossible correlations):")
    print(f"Is valid? {is_valid}")
    print(f"Details: {details}")

    #example 3:
    matrix_3 = [
    [
      1,
      -0.00035
    ],
    [
      -0.00035,
      1
    ]
              ]

    is_valid, details = is_asset_correlation_matrix(matrix_3)
    print("Example 3:")
    print(f"Is valid? {is_valid}")
    print(f"Details: {details}")
