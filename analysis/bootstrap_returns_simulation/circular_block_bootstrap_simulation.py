import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# 1. HELPER: Automatic Block Length Selection (Politis & White, 2004)
# =============================================================================

def optimal_block_length(returns):
    """
    Calculates the optimal block length for the Circular Block Bootstrap 
    using the method described in Politis & White (2004) and corrected in 
    Patton, Politis & White (2009).

    This implementation uses the parametric approximation (fitting an AR(1) model)
    to estimate the spectral density constants, which is the standard practical 
    approach for 'automatic' selection on financial time series.

    Formula (Circular Block): b_opt = ( (2 * rho^2) / (1 - rho^2)^2 )^(1/3) * T^(1/3)
    """
    # Ensure numpy array
    if isinstance(returns, pd.DataFrame):
        data = returns.values
    else:
        data = np.array(returns)
    
    T, n = data.shape
    optimal_blocks = []

    for i in range(n):
        series = data[:, i]
        
        # 1. Estimate AR(1) coefficient (rho)
        # We use Pearson correlation at lag 1 as a robust estimator for rho
        if np.std(series) == 0:
            rho = 0
        else:
            rho = np.corrcoef(series[:-1], series[1:])[0, 1]
            
        # Bound rho to avoid division by zero or explosive values
        # (Financial returns typically have low autocorrelation, e.g., < 0.2)
        rho = np.clip(rho, -0.99, 0.99)

        # 2. Apply Politis & White (2004) Formula for Circular Bootstrap
        # The constant differs slightly between Moving Block and Circular Block.
        # For Circular Block, the standard approximation is:
        term1 = (2 * (rho ** 2)) / ((1 - (rho ** 2)) ** 2)
        b_val = (term1 ** (1/3)) * (T ** (1/3))
        
        optimal_blocks.append(b_val)

    # 3. Aggregation
    # Since we must apply ONE block length to the whole system to preserve
    # cross-sectional correlations, we take the median (or ceiling of median).
    avg_b = np.median(optimal_blocks)
    
    # Round to nearest integer and ensure minimum is 1
    final_b = max(1, int(np.round(avg_b)))
    
    return final_b

# =============================================================================
# 2. CORE: Empirical Bootstrap (Your provided code)
# =============================================================================

def empirical_bootstrap_simulation(returns, T_prime, seed=None):
    """
    Performs an empirical bootstrap simulation (i.i.d. resampling).
    Used as a fallback if block_length == 1.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    is_dataframe = isinstance(returns, pd.DataFrame)
    if is_dataframe:
        data_values = returns.values
        columns = returns.columns
    else:
        data_values = np.array(returns)

    T, n = data_values.shape
    random_indices = rng.integers(low=0, high=T, size=T_prime)
    bootstrapped_data = data_values[random_indices, :]

    if is_dataframe:
        return pd.DataFrame(bootstrapped_data, columns=columns)
    else:
        return bootstrapped_data

# =============================================================================
# 3. CORE: Circular Block Bootstrap
# =============================================================================

def circular_block_bootstrap(returns, T_prime, block_length=None, seed=None):
    """
    Performs a Circular Block Bootstrap simulation of asset returns.

    Parameters:
    -----------
    returns : np.ndarray or pd.DataFrame
        The original dataset of returns (T x n).
    T_prime : int
        The number of time periods to simulate.
    block_length : int, optional
        The integer block length 'b'. If None, it is calculated automatically
        using the Politis & White (2004) method.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    np.ndarray or pd.DataFrame
        The bootstrapped returns matrix of shape (T_prime, n).
    """
    
    # 1. Setup Data
    if isinstance(returns, pd.DataFrame):
        data = returns.values
        columns = returns.columns
        is_df = True
    else:
        data = np.array(returns)
        is_df = False
        
    T, n = data.shape

    # 2. Determine Block Length
    if block_length is None:
        b = optimal_block_length(data)
        print(f"DEBUG: Automatically selected block length b={b}")
    else:
        b = int(block_length)

    # 3. Fallback Condition
    # "in case that value is one, the circular bootstrap is automatically 
    # replaced by an empirical bootstrap."
    if b < 2:
        print("DEBUG: Block length is 1. Reverting to Empirical Bootstrap.")
        return empirical_bootstrap_simulation(returns, T_prime, seed)

    # 4. Circular Block Bootstrap Logic
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Calculate how many blocks we need to cover T_prime
    # We use ceiling division
    k = int(np.ceil(T_prime / b))

    # We sample k starting indices uniformly from [0, T-1]
    # In Circular bootstrap, any index is valid because we wrap around.
    start_indices = rng.integers(low=0, high=T, size=k)

    bootstrapped_blocks = []

    for start_idx in start_indices:
        # Create indices for the block: [start, start+1, ..., start+b-1]
        # Use modulo T to handle the "Circular" wrapping
        block_indices = (np.arange(start_idx, start_idx + b)) % T
        
        # Extract block
        block_data = data[block_indices, :]
        bootstrapped_blocks.append(block_data)

    # 5. Concatenate and Truncate
    # Stack all blocks vertically
    simulated_path = np.vstack(bootstrapped_blocks)
    
    # Truncate to exact T_prime length (since k*b >= T_prime)
    simulated_path = simulated_path[:T_prime, :]

    # Return Result
    if is_df:
        return pd.DataFrame(simulated_path, columns=columns)
    else:
        return simulated_path

# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Settings for demonstration
    np.random.seed(42)
    n_assets = 3
    n_periods_original = 1000
    
    # 1. Generate Synthetic Data with Autocorrelation
    # We add a slight AR(1) component to justify using Block Bootstrap over Empirical
    # X_t = 0.5 * X_{t-1} + noise
    noise = np.random.multivariate_normal(
        [0, 0, 0], 
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.2], [0.2, 0.2, 1.0]], 
        n_periods_original
    )
    
    returns_data = np.zeros_like(noise)
    for t in range(1, n_periods_original):
        returns_data[t] = 0.3 * returns_data[t-1] + noise[t]
        
    original_df = pd.DataFrame(returns_data, columns=['Asset_A', 'Asset_B', 'Asset_C'])

    print(f"Original Data Shape: {original_df.shape}")
    
    # 2. Run Simulations
    T_sim = 2000
    
    # A. Empirical (No memory)
    emp_sim = empirical_bootstrap_simulation(original_df, T_sim, seed=42)
    
    # B. Circular Block (Auto block length)
    cbb_sim = circular_block_bootstrap(original_df, T_sim, block_length=None, seed=42)
    
    # 3. Comparison / Validation
    # We calculate the lag-1 autocorrelation of Asset A to see if CBB preserved it
    orig_acf = original_df['Asset_A'].autocorr(lag=1)
    emp_acf = emp_sim['Asset_A'].autocorr(lag=1)
    cbb_acf = cbb_sim['Asset_A'].autocorr(lag=1)
    
    print("-" * 40)
    print(f"Original Lag-1 Autocorr (Asset A):  {orig_acf:.4f}")
    print(f"Empirical BS Lag-1 Autocorr:        {emp_acf:.4f}  (Should be ~0)")
    print(f"Circular Block BS Lag-1 Autocorr:   {cbb_acf:.4f}  (Should be closer to Original)")
    print("-" * 40)
    
    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Cross-Sectional Correlation (Should be preserved in both)
    sns.heatmap(cbb_sim.corr(), annot=True, cmap='coolwarm', ax=axes[0])
    axes[0].set_title("CBB Cross-Sectional Correlation")
    
    # Plot 2: Time Series snippet
    axes[1].plot(original_df['Asset_A'].iloc[:100], label='Original', alpha=0.7)
    axes[1].plot(cbb_sim['Asset_A'].iloc[:100], label='CBB Sim', alpha=0.7, linestyle='--')
    axes[1].set_title("First 100 periods (Asset A)")
    axes[1].legend()
    
    # Plot 3: Autocorrelation Structure (The key difference)
    # Visualizing how CBB preserves the 'clumps' of data compared to IID noise
    pd.plotting.autocorrelation_plot(original_df['Asset_A'], ax=axes[2], label='Original')
    pd.plotting.autocorrelation_plot(cbb_sim['Asset_A'], ax=axes[2], label='CBB', linestyle='--')
    pd.plotting.autocorrelation_plot(emp_sim['Asset_A'], ax=axes[2], label='Empirical', linestyle=':')
    axes[2].set_xlim(0, 20)
    axes[2].set_title("Autocorrelation Function")
    
    plt.tight_layout()
    plt.show()
