import numpy as np
import pandas as pd
from numpy.linalg import inv, pinv

def calculate_turbulence_index(current_returns, reference_mean, reference_cov):
    """
    Calculates the Turbulence Index (d) for a single period.
    
    Formula:
        d = (1/n) * (r - mu).T * Sigma^-1 * (r - mu)
        
    Parameters:
    -----------
    current_returns (r) : np.array
        The vector of asset returns for the current period T (shape: n,).
    reference_mean (mu) : np.array
        The vector of mean asset returns over the reference period T' (shape: n,).
    reference_cov (Sigma) : np.ndarray
        The covariance matrix of asset returns over the reference period T' (shape: n x n).
        
    Returns:
    --------
    float
        The turbulence index d.
    """
    # 1. Ensure inputs are numpy arrays
    r = np.array(current_returns)
    mu = np.array(reference_mean)
    sigma = np.array(reference_cov)
    
    # 2. Get number of assets (n)
    n = len(r)
    
    # 3. Compute the deviation vector (r - mu)
    deviation = r - mu
    
    # 4. Invert the Covariance Matrix (Sigma^-1)
    # Note: The prompt assumes Sigma is invertible (positive definite).
    # For robustness, we can check for singularity, but we will use standard inv()
    # as per the theoretical definition.
    try:
        sigma_inv = inv(sigma)
    except np.linalg.LinAlgError:
        # Fallback for singular matrices (if assumption is violated)
        print("Warning: Covariance matrix is singular. Using Pseudo-Inverse.")
        sigma_inv = pinv(sigma)

    # 5. Compute the Mahalanobis Distance (Squared)
    # calculation: (1 x n) vector @ (n x n) matrix @ (n x 1) vector
    # In numpy 1D arrays, @ handles the transposes automatically.
    distance_squared = deviation.T @ sigma_inv @ deviation
    
    # 6. Normalize by n (as per Kinlaw & Turkington 2013)
    d = distance_squared / n
    
    return d

def compute_historical_turbulence(returns, window_size=None, constant_reference=False):
    """
    Computes the turbulence index over a time series.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Time series of asset returns.
    window_size : int, optional
        If provided, defines the rolling window T' to calculate mu and Sigma.
        If None and constant_reference=False, uses the entire history up to t (expanding window).
    constant_reference : bool, optional
        If True, uses the FIRST 'window_size' observations as the fixed reference (T')
        and calculates turbulence for all subsequent periods T.
        
    Returns:
    --------
    pd.Series
        Time series of Turbulence Index values.
    """
    n_assets = returns.shape[1]
    turbulence_series = pd.Series(index=returns.index, dtype=float)
    turbulence_series[:] = np.nan
    
    # Mode A: Fixed Reference Period (Typical for stress testing)
    # e.g., "How turbulent is today compared to the 2010-2020 average?"
    if constant_reference and window_size:
        # Define T' (Reference Period)
        reference_data = returns.iloc[:window_size]
        mu = reference_data.mean().values
        sigma = reference_data.cov().values
        
        print(f"Reference Period established (First {window_size} obs). Calculating turbulence for remaining data...")
        
        # Calculate T (Current Period) for the rest of the data
        for t in range(window_size, len(returns)):
            r = returns.iloc[t].values
            d = calculate_turbulence_index(r, mu, sigma)
            turbulence_series.iloc[t] = d
            
    # Mode B: Rolling Reference Period (Typical for monitoring regime shifts)
    # e.g., "How turbulent is today compared to the last 1 year?"
    elif window_size:
        print(f"Calculating Rolling Turbulence (Window={window_size})...")
        for t in range(window_size, len(returns)):
            # Reference window T'
            ref_window = returns.iloc[t-window_size:t]
            mu = ref_window.mean().values
            sigma = ref_window.cov().values
            
            # Current vector r (at time t)
            r = returns.iloc[t].values
            
            d = calculate_turbulence_index(r, mu, sigma)
            turbulence_series.iloc[t] = d
            
    return turbulence_series

# ==========================================
# Example Usage / Main
# ==========================================

if __name__ == "__main__":
    # 1. Setup Input: 2 Assets
    # Let's assume a reference period T' where markets are calm.
    # Mean returns are ~0, Covariance is standard.
    
    reference_mu = [1.0, 1.0]  # Expected return 0%
    reference_cov = [
        [9.0, 1.0],  # Variance 1, Correlation 0.5
        [1.0, 5.0]
    ]
    
    print("--- Reference Parameters (T') ---")
    print(f"Means: {reference_mu}")
    print(f"Covariance:\n{np.array(reference_cov)}")
    
    # 2. Test Case A: "Normal" day
    # Returns are close to the mean (0,0)
    r_normal = [1.0, 0.0]
    d_normal = calculate_turbulence_index(r_normal, reference_mu, reference_cov)
    
    # 3. Test Case B: "Turbulent" day
    # Returns are large (3 std devs) and move in opposite direction to correlation
    # (i.e., Correlation surprise: Asset 1 up, Asset 2 down significantly)
    r_shock = [3.0, -3.0] 
    d_shock = calculate_turbulence_index(r_shock, reference_mu, reference_cov)
    
    print("\n--- Turbulence Calculation ---")
    print(f"Case A (Quiet Day) returns {r_normal}: d = {d_normal:.12f}")
    print(f"Case B (Shock Day) returns {r_shock}: d = {d_shock:.12f}")
    
    print("\n--- Verification Logic ---")
    print("Expected Value of d is approx 1.0 for normal data.")
    print("Values significantly > 1.0 indicate financial turbulence.")

    # 4. Generate Synthetic Time Series Example
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    
    # Generate 100 days of data
    # First 90 days: Normal distribution matching reference cov
    data_calm = np.random.multivariate_normal(reference_mu, reference_cov, 90)
    
    # Last 10 days: High Volatility & Broken Correlation (Turbulence)
    shock_cov = [[5.0, -0.9], [-0.9, 5.0]] # Higher vol, negative correlation
    data_shock = np.random.multivariate_normal(reference_mu, shock_cov, 10)
    
    data_all = np.vstack([data_calm, data_shock])
    df_returns = pd.DataFrame(data_all, index=dates, columns=['Asset_1', 'Asset_2'])
    
    # Calculate Turbulence using the first 50 days as the "Reference Baseline"
    turb_values = compute_historical_turbulence(df_returns, window_size=50, constant_reference=True)
    
    # Plotting (text based output here)
    print("\n--- Time Series Analysis (Last 15 days) ---")
    print(turb_values.tail(15))
