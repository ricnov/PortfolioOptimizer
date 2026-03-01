import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_mri(covariance_matrix, n_components):
    """
    Calculates the Market Rank Indicator (MRI) for a given covariance matrix.
    
    Formula:
        MRI = lambda_n / ( Product(lambda_1...lambda_N) )^(1/N)
        
    Where:
        lambda_n: The largest eigenvalue.
        lambda_1...lambda_N: The N smallest eigenvalues.
        
    Parameters:
    -----------
    covariance_matrix : np.ndarray or pd.DataFrame
        The covariance matrix (Sigma) of the asset returns (n x n).
    n_components : int
        The number of smallest eigenvalues (N) to retain in the denominator.
        
    Returns:
    --------
    float
        The Market Rank Indicator (MRI) value.
    """
    # 1. Prepare Input
    if isinstance(covariance_matrix, pd.DataFrame):
        cov_mat = covariance_matrix.values
    else:
        cov_mat = np.array(covariance_matrix)
        
    # 2. Compute Eigenvalues
    # np.linalg.eigh is optimized for symmetric/Hermitian matrices (like covariance).
    # It returns eigenvalues in ascending order: lambda_1 <= lambda_2 <= ... <= lambda_n
    eigenvalues, _ = np.linalg.eigh(cov_mat)
    
    # 3. Extract Specific Eigenvalues
    # lambda_n (The largest eigenvalue) is the last element in the sorted array
    lambda_n = eigenvalues[-1]
    
    # lambda_1 ... lambda_N (The N smallest eigenvalues) are the first N elements
    smallest_eigenvalues = eigenvalues[:n_components]
    
    # 4. Handle Singularity / Non-Positive Definite Matrices
    # The logic requires the matrix to be invertible (positive definite), i.e., lambda > 0.
    # If the smallest eigenvalues are <= 0, the geometric mean is undefined or 0.
    if np.any(smallest_eigenvalues <= 0):
        # In financial contexts, a 0 eigenvalue implies perfect collinearity (singularity).
        # This usually represents infinite/undefined MRI based on the standard formula.
        return np.inf

    # 5. Compute Denominator: Geometric Mean of the N smallest eigenvalues
    # Geometric Mean = (Product(values))^(1/N)
    
    # Using prod() directly:
    product_smallest = np.prod(smallest_eigenvalues)
    denominator = np.power(product_smallest, 1 / n_components)
    
    # Alternative (Log-Sum-Exp) for numerical stability with very large matrices/numbers:
    # denominator = np.exp(np.mean(np.log(smallest_eigenvalues)))
    
    # 6. Compute MRI
    mri = lambda_n / denominator
    
    return mri

def rolling_mri(returns, window_size, n_components=1):
    """
    Calculates the rolling Market Rank Indicator over a time series.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns time series.
    window_size : int
        Rolling window size.
    n_components : int
        Number of smallest eigenvalues to retain (N).
        
    Returns:
    --------
    pd.Series
        Time series of MRI values.
    """
    mri_series = pd.Series(index=returns.index, dtype=float)
    mri_series[:] = np.nan
    
    print(f"Calculating Rolling MRI (Window={window_size}, N={n_components})...")
    
    for t in range(window_size, len(returns)):
        # Get window
        subset = returns.iloc[t-window_size:t]
        
        # Calculate Covariance
        cov_matrix = subset.cov()
        
        # Calculate MRI
        try:
            val = calculate_mri(cov_matrix, n_components)
            mri_series.iloc[t] = val
        except Exception as e:
            mri_series.iloc[t] = np.nan
            
    return mri_series

# ==========================================
# Example Usage / Test Main
# ==========================================

def main():
    # 1. Define inputs based on the previous simple example
    # A 2-asset system with positive covariance
    input_data = {
        "assets": 2,
        "assetsCovarianceMatrix": [
            [9, 1],
            [1, 5]
        ]
    }
    
    cov_matrix = np.array(input_data["assetsCovarianceMatrix"])
    N = 1  # We retain the 1 smallest eigenvalue for the denominator
    
    # 2. Run Calculation
    mri_value = calculate_mri(cov_matrix, N)
    
    # 3. Verify Logic Manually
    # Eigenvalues of [[9, 1], [1, 5]] are approx: 9.236 (lambda_2) and 4.764 (lambda_1)
    # Sorted: [4.764, 9.236]
    # lambda_n (Largest) = 9.236
    # N=1, so we take just the smallest: 4.764
    # Geometric mean of [4.764] is 4.764.
    # MRI = 9.236 / 4.764 = 1.938
    
    print("--- Market Rank Indicator (MRI) Test ---")
    print(f"Input Matrix:\n{cov_matrix}")
    
    evals, _ = np.linalg.eigh(cov_matrix)
    print(f"\nEigenvalues (Ascending): {evals}")
    print(f"Lambda_n (Largest, Numerator): {evals[-1]:.4f}")
    
    # Calculate Denominator details
    smallest = evals[:N]
    geo_mean = np.prod(smallest)**(1/N)
    print(f"Smallest {N} Eigenvalues: {smallest}")
    print(f"Geometric Mean (Denominator): {geo_mean:.4f}")
    
    print(f"\nCalculated MRI: {mri_value:.6f}")
    
    # 4. Check 'Spectral Condition Number' equivalence
    # When N=1, MRI is exactly the Condition Number (Lambda_max / Lambda_min)
    cond_number = evals[-1] / evals[0]
    print(f"Verification (Cond. Num): {cond_number:.6f}")

if __name__ == "__main__":
    main()
