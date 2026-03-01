import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_absorption_ratio(covariance_matrix, n_components):
    """
    Calculates the Absorption Ratio (AR) for a given covariance matrix.
    
    Formula:
        AR = Sum(Variance of Top N Eigenvectors) / Sum(Variance of All Eigenvectors)
        
    Parameters:
    -----------
    covariance_matrix : np.ndarray or pd.DataFrame
        The covariance matrix (Sigma) of the asset returns (n x n).
    n_components : int
        The number of eigenvectors (N) to retain in the numerator.
        
    Returns:
    --------
    float
        The absorption ratio value (between 0 and 1).
    """
    # Ensure inputs are numpy arrays
    if isinstance(covariance_matrix, pd.DataFrame):
        cov_mat = covariance_matrix.values
    else:
        cov_mat = covariance_matrix

    # Compute Eigenvalues and Eigenvectors
    # Since Covariance matrices are symmetric, we use eigh (Hermitian) 
    # which is faster and more numerically stable for symmetric matrices.
    eigenvalues, _ = np.linalg.eigh(cov_mat)
    
    # Sort eigenvalues in descending order (eigh returns them in ascending order)
    # sigma_E1^2 >= ... >= sigma_En^2
    sorted_eigenvalues = eigenvalues[::-1]
    
    # Ensure eigenvalues are positive (numerical noise can cause tiny negatives close to 0)
    sorted_eigenvalues = np.maximum(sorted_eigenvalues, 0)
    
    # Calculate numerator: Sum of variances of top N eigenvectors
    numerator = np.sum(sorted_eigenvalues[:n_components])
    
    # Calculate denominator: Sum of variances of all eigenvectors (Total Variance)
    # This is mathematically equivalent to the trace of the covariance matrix
    denominator = np.sum(sorted_eigenvalues)
    
    # Compute Absorption Ratio
    ar = numerator / denominator
    
    return ar

def rolling_absorption_ratio(returns, window_size, n_components=None, fraction_components=0.2):
    """
    Calculates the rolling Absorption Ratio over a time series of asset returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Time series of asset returns (rows=time, cols=assets).
    window_size : int
        The size of the rolling window (e.g., 252 for 1 year of daily data).
    n_components : int, optional
        Fixed number of components (N) to retain.
    fraction_components : float, optional
        Fraction of total assets to retain as N (default is 0.2, i.e., 20%).
        Used only if n_components is None.
        
    Returns:
    --------
    pd.Series
        Time series of Absorption Ratios.
    """
    num_assets = returns.shape[1]
    
    # Determine N (number of components)
    if n_components is None:
        n_components = int(np.round(num_assets * fraction_components))
        # Ensure at least 1 component is selected
        n_components = max(1, n_components)
    
    print(f"Calculating Absorption Ratio using top {n_components} components out of {num_assets} assets...")
    
    # Container for results
    ar_series = pd.Series(index=returns.index, dtype=float)
    ar_series[:] = np.nan
    
    # Loop through the data with a rolling window
    # Note: efficient rolling covariance in pandas can be done via .rolling().cov()
    # However, for PCA we usually need step-by-step processing.
    
    # We start the loop from 'window_size' to the end
    for t in range(window_size, len(returns)):
        # Extract the window of returns
        subset_returns = returns.iloc[t-window_size:t]
        
        # Calculate Covariance Matrix for this window
        # The paper typically assumes centralizing returns (subtract mean) 
        # is handled by the cov function.
        cov_matrix = subset_returns.cov()
        
        # Calculate AR for this window
        ar = calculate_absorption_ratio(cov_matrix, n_components)
        
        # Store result (aligned to the end of the window)
        ar_series.iloc[t] = ar
        
    return ar_series

# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    # Simulate 50 assets over 1000 days
    np.random.seed(42)
    n_assets = 50
    n_days = 1000
    
    # Create two regimes:
    # Regime A: Low correlation (Random noise)
    data_A = np.random.normal(0, 0.01, (500, n_assets))
    
    # Regime B: High correlation (Systemic shock)
    # Introduce a common factor affecting all assets
    market_factor = np.random.normal(0, 0.02, (500, 1))
    noise = np.random.normal(0, 0.01, (500, n_assets))
    data_B = market_factor + noise # Assets move together
    
    # Combine data
    data = np.vstack([data_A, data_B])
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    
    asset_names = [f"Asset_{i}" for i in range(n_assets)]
    returns_df = pd.DataFrame(data, index=dates, columns=asset_names)
    
    # 2. Parameters
    # Kritzman paper often suggests a window of ~500 days for long term or smaller for shifts.
    # We will use a smaller window here to see the transition quickly.
    WINDOW_SIZE = 60 
    
    # Retain roughly 1/5th of eigenvectors (approx 20%)
    FRACTION_N = 0.2 
    
    # 3. Calculate Rolling Absorption Ratio
    ar_results = rolling_absorption_ratio(
        returns_df, 
        window_size=WINDOW_SIZE, 
        fraction_components=FRACTION_N
    )
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(ar_results, label=f'Absorption Ratio (Window={WINDOW_SIZE})', color='darkblue')
    
    # Mark the transition point
    transition_date = dates[500]
    plt.axvline(x=transition_date, color='red', linestyle='--', label='Regime Change (High Correlation)')
    
    plt.title("Systemic Risk Indicator: Absorption Ratio")
    plt.ylabel("Absorption Ratio")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output detailed check for a specific point
    print("\nDetailed Check for the final window:")
    final_cov = returns_df.iloc[-WINDOW_SIZE:].cov()
    N = int(n_assets * FRACTION_N)
    
    # Manual PCA check
    evals, _ = np.linalg.eigh(final_cov)
    evals = np.sort(evals)[::-1]
    numerator = sum(evals[:N])
    denominator = sum(evals)
    calculated_ar = numerator / denominator
    
    print(f"Total Variance (Denominator): {denominator:.6f}")
    print(f"Variance of Top {N} PC (Numerator): {numerator:.6f}")
    print(f"Calculated AR: {calculated_ar:.4f}")
