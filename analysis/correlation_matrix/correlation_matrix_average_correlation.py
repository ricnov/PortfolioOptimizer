import numpy as np

def average_correlation_formula(C):
    """
    Computes the average correlation using standard nested loops.
    This directly maps to the mathematical formula provided.
    
    Parameters:
    C (list of lists or numpy.ndarray): An n x n asset correlation matrix.
    
    Returns:
    float: The average correlation.
    """
    n = len(C)
    
    if n <= 1:
        raise ValueError("The matrix must be at least 2x2.")
        
    total_sum = 0.0
    
    # Outer loop corresponding to sum_{i=1}^{n-1}
    # Note: Python uses 0-based indexing, so we go from 0 to n-2
    for i in range(n - 1):
        # Inner loop corresponding to sum_{j=i+1}^{n}
        # Python goes from i+1 to n-1
        for j in range(i + 1, n):
            total_sum += C[i][j]
            
    # Calculate the average by multiplying by 2 / (n * (n - 1))
    average_corr = (2 * total_sum) / (n * (n - 1))
    
    return average_corr

def average_correlation_numpy(C):
    """
    Computes the average correlation efficiently using NumPy.
    
    Parameters:
    C (numpy.ndarray): An n x n asset correlation matrix.
    
    Returns:
    float: The average correlation.
    """
    C = np.asarray(C)
    n = C.shape[0]
    
    if n <= 1:
        raise ValueError("The matrix must be at least 2x2.")
        
    # np.triu_indices gets the indices for the upper triangle.
    # Setting k=1 explicitly excludes the main diagonal.
    upper_triangle_indices = np.triu_indices(n, k=1)
    
    # Extract those off-diagonal elements
    off_diagonal_elements = C[upper_triangle_indices]
    
    # Compute the arithmetic mean of these elements
    average_corr = np.mean(off_diagonal_elements)
    
    return average_corr

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Create an example 3x3 correlation matrix
    # Diagonal elements are 1.0 (an asset perfectly correlates with itself)
    # The matrix is symmetric.
    C_matrix = np.array([
        [
      1,
      0.5,
      0.9
    ],
    [
      0.5,
      1,
      0.7
    ],
    [
      0.9,
      0.7,
      1
    ]
    ])
    
    print("Asset Correlation Matrix (C):")
    print(C_matrix)
    
    # The off-diagonal elements are 0.50, 0.90, and 0.70.
    # The expected average is: (0.50 + 0.90 + 0.70) / 3 = 0.70 
    
    avg_formula = average_correlation_formula(C_matrix)
    avg_numpy = average_correlation_numpy(C_matrix)
    
    print(f"\nAverage Correlation (using formula/loops): {avg_formula:.4f}")
    print(f"Average Correlation (using NumPy): {avg_numpy:.4f}")
