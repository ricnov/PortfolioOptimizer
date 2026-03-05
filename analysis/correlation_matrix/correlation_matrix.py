import numpy as np
import pandas as pd

def corr_from_covariance(cov_matrix):
    """
    Extracts the asset correlation matrix from the asset covariance matrix.
    Formula: C_{i,j} = Sigma_{i,j} / (sigma_i * sigma_j)
    
    Parameters:
    cov_matrix (numpy.ndarray or pandas.DataFrame): The n x n covariance matrix.
    
    Returns:
    numpy.ndarray: The n x n correlation matrix.
    """
    # Convert to numpy array in case a pandas DataFrame is passed
    cov_matrix = np.asarray(cov_matrix)
    
    # Extract standard deviations (sigma): square root of the diagonal elements (variances)
    vols = np.sqrt(np.diag(cov_matrix))
    
    # Compute the denominator matrix (sigma_i * sigma_j) using the outer product
    vols_matrix = np.outer(vols, vols)
    
    # Compute the correlation matrix
    corr_matrix = cov_matrix / vols_matrix
    
    # Fix potential floating point precision issues by enforcing 1.0 on the diagonal
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix

def corr_from_returns(returns):
    """
    Computes the Pearson asset correlation matrix directly from asset returns.
    
    Parameters:
    returns (pandas.DataFrame or numpy.ndarray): T x n matrix of asset returns 
                                                 (T = time periods, n = assets)
    
    Returns:
    pandas.DataFrame or numpy.ndarray: The n x n correlation matrix.
    """
    if isinstance(returns, pd.DataFrame):
        # Pandas has a built-in method for Pearson correlation
        return returns.corr(method='pearson')
    else:
        # If it's a numpy array, use numpy's corrcoef 
        # (rowvar=False implies columns represent different assets)
        return np.corrcoef(returns, rowvar=False)
