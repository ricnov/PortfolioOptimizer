import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def empirical_bootstrap_simulation(returns, T_prime, seed=None):
    """
    Performs an empirical bootstrap simulation of asset returns based on 
    Efron (1979) and the logic of resampling cross-sectional vectors.
    
    Parameters:
    -----------
    returns : np.ndarray or pd.DataFrame
        The original dataset of returns. 
        Shape should be (T, n) where:
        - T is the number of time periods in the original history.
        - n is the number of assets.
    T_prime : int
        The number of time periods to simulate in the bootstrap sample.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    np.ndarray or pd.DataFrame
        The bootstrapped returns matrix of shape (T_prime, n).
    """
    
    # Handle random state for reproducibility
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    # Convert input to numpy array if it is a pandas DataFrame
    is_dataframe = isinstance(returns, pd.DataFrame)
    if is_dataframe:
        data_values = returns.values
        columns = returns.columns
        index = returns.index
    else:
        data_values = np.array(returns)
        
    T, n = data_values.shape
    
    # ---------------------------------------------------------
    # CORE LOGIC: Efron (1979) Empirical Bootstrap
    # ---------------------------------------------------------
    # 1. We treat each row (time period t) as a single observation unit.
    # 2. We sample T' indices from the range [0, T-1] with replacement.
    #    This ensures we pick cross-sectional vectors r_{t,:} 
    #    preserving the correlation structure between assets i=1..n.
    
    random_indices = rng.integers(low=0, high=T, size=T_prime)
    
    # 3. Construct the bootstrap sample using these indices
    bootstrapped_data = data_values[random_indices, :]
    
    # Return in the same format as input
    if is_dataframe:
        # Create a new DataFrame. Note: The index is reset because 
        # time order is scrambled/randomized in i.i.d. bootstrap.
        return pd.DataFrame(bootstrapped_data, columns=columns)
    else:
        return bootstrapped_data

# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # 1. Setup Synthetic Data
    # -----------------------------------------------------
    np.random.seed(42)
    n_assets = 3
    n_periods_original = 1000  # T
    
    # Generate correlated returns (Multivariate Normal)
    # Asset 0 and 1 are highly correlated (0.9), Asset 2 is distinct
    means = [0.001, 0.001, 0.001]
    cov_matrix = [
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.2],
        [0.2, 0.2, 1.0]
    ]
    
    # Original Returns Matrix (T x n)
    original_returns = np.random.multivariate_normal(means, cov_matrix, n_periods_original)
    original_df = pd.DataFrame(original_returns, columns=['Asset_A', 'Asset_B', 'Asset_C'])
    
    print(f"Original Data Shape: {original_df.shape}")
    print("Original Correlation Matrix:")
    print(original_df.corr().round(2))
    print("-" * 30)

    # 2. Perform Empirical Bootstrap
    # -----------------------------------------------------
    # We want to simulate a new timeline of length T' = 5000
    T_prime = 5000
    
    bootstrapped_df = empirical_bootstrap_simulation(original_df, T_prime, seed=99)
    
    print(f"Bootstrapped Data Shape: {bootstrapped_df.shape}")
    print("Bootstrapped Correlation Matrix:")
    print(bootstrapped_df.corr().round(2))
    
    # 3. Validation / Visualization
    # -----------------------------------------------------
    # The key property of this method is that it PRESERVES cross-sectional correlation.
    # Asset A and Asset B should still be ~0.9 correlated.
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(original_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Original Correlation Structure")
    
    sns.heatmap(bootstrapped_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title(f"Bootstrapped Correlation Structure (T'={T_prime})")
    
    plt.tight_layout()
    plt.show()

    print("\nCheck: Notice how the correlation between Asset_A and Asset_B")
    print("is preserved in the simulation, confirming the row-wise resampling logic.")
