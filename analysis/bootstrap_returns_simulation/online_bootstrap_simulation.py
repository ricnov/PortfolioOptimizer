import numpy as np
import pandas as pd

class AutoregressiveOnlineBootstrap:
    """
    Implements the Autoregressive Online Bootstrap for Time Series 
    (Palm & Nagler, 2024).
    
    This method generates bootstrap weights using a time-dependent 
    autoregressive process to account for serial dependence in the data.
    """
    
    def __init__(self, beta=None, random_state=None):
        """
        Initialize the bootstrap generator.

        Parameters:
        -----------
        beta : float, optional
            The smoothing hyperparameter. 
            Default is the optimal bias-variance value: sqrt(2) - 1 (~0.414).
        random_state : int, optional
            Seed for reproducibility.
        """
        # Default beta = 2^(1/2) - 1
        if beta is None:
            self.beta = np.sqrt(2) - 1
        else:
            self.beta = beta
            
        self.rng = np.random.default_rng(random_state)

    def generate_weights(self, T, n_paths=1):
        """
        Generates the autoregressive weight matrix V of shape (T, n_paths).
        
        The process is defined as:
        V_t = 1 + rho_t * (V_{t-1} - 1) + sqrt(1 - rho_t^2) * zeta_t
        where rho_t = 1 - t^(-beta) and zeta_t ~ N(0, 1).
        """
        # V will store the weights. shape: (T, n_paths)
        V = np.zeros((T, n_paths))
        
        # Initialize V_0. 
        # The stationary distribution of the process is N(1, 1).
        # We start V[0] (t=1 in paper) based on the process definition.
        # At t=1, rho_1 = 1 - 1^(-beta) = 0.
        # So V_1 = 1 + 0 + 1 * zeta_1 = 1 + zeta_1.
        
        zeta = self.rng.standard_normal((T, n_paths))
        
        # Iterate to generate the AR process
        # We use a loop because rho changes with t and V_t depends on V_{t-1}
        
        # t=0 (corresponding to time period 1)
        # rho = 0, so V[0] is just i.i.d N(1,1)
        V[0, :] = 1.0 + zeta[0, :]
        
        for t_idx in range(1, T):
            t = t_idx + 1 # Time period 1..T
            
            # Calculate time-dependent correlation coefficient
            # rho_t = 1 - t^(-beta)
            rho_t = 1.0 - (t ** (-self.beta))
            
            # Update rule: V_t = 1 + rho*(V_{t-1}-1) + sqrt(1-rho^2)*zeta
            # Note: We use t_idx-1 for the previous weight
            prev_centered = V[t_idx-1, :] - 1.0
            innovation = np.sqrt(1.0 - rho_t**2) * zeta[t_idx, :]
            
            V[t_idx, :] = 1.0 + (rho_t * prev_centered) + innovation
            
        return V

    def simulate(self, returns, n_simulations=1, T_prime=None):
        """
        Simulate asset returns using the bootstrap method.

        Parameters:
        -----------
        returns : np.ndarray or pd.DataFrame
            Original asset returns of shape (T, n_assets).
        n_simulations : int
            Number of bootstrap paths to generate.
        T_prime : int, optional
            Number of time periods to simulate (T' <= T).
            If None, defaults to T.

        Returns:
        --------
        simulated_paths : np.ndarray
            A 3D array of shape (n_simulations, T_prime, n_assets).
            (Or (T_prime, n_assets) if n_simulations=1 and user prefers squeeze,
             but standard is 3D).
        """
        # Input handling
        is_pandas = isinstance(returns, (pd.DataFrame, pd.Series))
        if is_pandas:
            data = returns.values
        else:
            data = np.asarray(returns)
            
        # Ensure 2D (T, n)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        T, n_assets = data.shape
        
        if T_prime is None:
            T_prime = T
        
        if T_prime > T:
            raise ValueError(f"T_prime ({T_prime}) cannot be larger than observed time periods T ({T}) for this implementation.")

        # 1. Calculate Empirical Mean (Cross-sectional center)
        # We assume stationarity for the bootstrap window
        mu = np.mean(data, axis=0)
        
        # 2. Center the returns
        # shape: (T, n_assets)
        centered_returns = data - mu
        
        # Slice to T_prime if necessary (bootstrap the first T_prime observations)
        # Or bootstrap the whole T and slice? 
        # Standard practice: Use data corresponding to the simulation horizon.
        r_c_subset = centered_returns[:T_prime]
        
        # 3. Generate Weights
        # shape: (T_prime, n_simulations)
        V = self.generate_weights(T_prime, n_paths=n_simulations)
        
        # 4. Apply Weights (Multiplier Bootstrap)
        # We need to broadcast:
        # returns: (T_prime, n_assets)
        # weights: (T_prime, n_simulations)
        # Output:  (n_simulations, T_prime, n_assets)
        
        simulated_paths = np.zeros((n_simulations, T_prime, n_assets))
        
        for i in range(n_simulations):
            # Extract weights for simulation i, center them at 0
            # V_i shape: (T_prime,)
            # w_i = V_i - 1  (Standard Normal Multiplier)
            w_i = (V[:, i] - 1.0).reshape(-1, 1)
            
            # Apply multiplier: r* = mu + (r - mu) * w
            # This preserves the variance structure (E[w^2]=1)
            sim_path = mu + (r_c_subset * w_i)
            simulated_paths[i] = sim_path
            
        return simulated_paths

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Generate Dummy Data (2 Assets, 100 Time periods)
    np.random.seed(42)
    T_periods = 100
    n_assets = 2
    
    # Correlated assets
    cov_matrix = [[0.001, 0.0005], [0.0005, 0.001]]
    returns_data = np.random.multivariate_normal([0.005, 0.008], cov_matrix, T_periods)
    
    # 2. Instantiate the Bootstrap
    # Beta defaults to optimal value (sqrt(2) - 1)
    aob = AutoregressiveOnlineBootstrap(random_state=999)
    
    # 3. Simulate
    T_prime = 50 # Simulate over a subset T' <= T
    n_sims = 5
    
    simulations = aob.simulate(returns_data, n_simulations=n_sims, T_prime=T_prime)
    
    print(f"Original Data Shape: {returns_data.shape}")
    print(f"Simulation Shape: {simulations.shape}  (Simulations, Time, Assets)")
    
    # Preview first simulation path for Asset 0
    print("\nFirst 5 returns of Simulation 1, Asset 0:")
    print(simulations[0, :5, 0])
