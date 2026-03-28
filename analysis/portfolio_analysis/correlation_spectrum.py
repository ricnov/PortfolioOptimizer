import numpy as np
import json

# ==========================================
# Method 1: Parametric (From Covariance Matrix)
# ==========================================
def compute_spectrum_from_cov(cov_matrix_data, weights_data):
    """
    Computes the ex-ante correlation spectrum of a portfolio using the asset 
    covariance matrix and portfolio weights.

    This parametric approach calculates the expected correlation between the 
    portfolio and its underlying assets, assuming the covariance structure 
    is known and stable.

    Parameters
    ----------
    cov_matrix_data : array_like of shape (n, n)
        The asset covariance matrix (Sigma), where n is the number of assets.
        Must be a symmetric, positive semi-definite matrix.
    weights_data : array_like of shape (n,)
        The vector of portfolio asset weights (w). The weights typically 
        sum to 1, and for long-only portfolios, w_i in [0, 1].

    Returns
    -------
    rho_w : ndarray of shape (n,)
        The correlation spectrum of the portfolio. Each element rho_i 
        represents the correlation between the portfolio and asset i, 
        bounded within [-1, 1].

    Notes
    -----
    The correlation spectrum vector rho(w) is defined component-wise as:
    
        rho(w)_i = (Sigma * w)_i / (sigma_p * sigma_i)
        
    Where:
        - Sigma is the n x n covariance matrix.
        - w is the n x 1 weight vector.
        - sigma_p is the portfolio standard deviation: sqrt(w^T * Sigma * w).
        - sigma_i is the standard deviation of asset i: sqrt(Sigma_ii).

    References
    ----------
    .. [1] T. Froidure, K. Jalalzai, Y. Choueifaty (2019). "Portfolio 
       Rho-Representativity". Int. J. Theor. Appl. Finance, 22(07).
    """
    Sigma = np.array(cov_matrix_data)
    w = np.array(weights_data)
    
    # 1. Compute asset standard deviations: sigma_i = sqrt(Sigma_ii)
    sigma_i = np.sqrt(np.diag(Sigma))
    
    # 2. Compute portfolio variance and standard deviation: sigma_p = sqrt(w^T * Sigma * w)
    port_variance = w.T @ Sigma @ w
    sigma_p = np.sqrt(port_variance)
    
    # 3. Compute marginal volatility contribution: (Sigma * w)
    marginal_contrib = Sigma @ w
    
    # 4. Compute the correlation spectrum: rho(w)
    rho_w = marginal_contrib / (sigma_p * sigma_i)
    
    return rho_w

# ==========================================
# Method 2: Historical (From Price Series)
# ==========================================
def compute_arithmetic_returns(prices):
    """Computes arithmetic returns from a price series: R_t = (P_t - P_{t-1}) / P_{t-1}"""
    p = np.array(prices)
    return (p[1:] - p[:-1]) / p[:-1]

def compute_spectrum_from_prices(assets_data, portfolio_data):
    """
    Computes the ex-post correlation spectrum of a portfolio using historical 
    price series.

    This historical approach calculates the realized Pearson correlation 
    between the arithmetic returns of the portfolio and the arithmetic returns 
    of each individual asset over a set number of time periods (T).

    Parameters
    ----------
    assets_data : list of dict
        A list of dictionaries containing historical asset prices. 
        Format expected: [{'assetPrices': [P_0, P_1, ..., P_T]}, ...]
        where P_t > 0 is the price of the asset at time t.
    portfolio_data : dict
        A dictionary containing historical portfolio values.
        Format expected: {'portfolioValues': [V_0, V_1, ..., V_T]}
        where V_t > 0 is the value of the portfolio at time t.

    Returns
    -------
    rho_w : ndarray of shape (n,)
        The historical correlation spectrum of the portfolio. Each element 
        rho_i is the Pearson correlation coefficient between the portfolio's 
        returns and asset i's returns, bounded within [-1, 1].

    Notes
    -----
    Arithmetic returns for an asset/portfolio from time t-1 to t are defined as:
    
        R_t = (P_t - P_{t-1}) / P_{t-1}
        
    The correlation spectrum vector rho(w) is defined component-wise as:
    
        rho(w)_i = rho_{p, i}
        
    Where rho_{p, i} is the sample Pearson correlation coefficient between 
    the series of portfolio returns and the series of returns for asset i.

    References
    ----------
    .. [1] T. Froidure, K. Jalalzai, Y. Choueifaty (2019). "Portfolio 
       Rho-Representativity". Int. J. Theor. Appl. Finance, 22(07).
    """
    # 1. Compute portfolio returns
    port_prices = portfolio_data["portfolioValues"]
    port_returns = compute_arithmetic_returns(port_prices)
    
    rho_w = []
    
    # 2. Compute returns for each asset and find the correlation with the portfolio
    for asset in assets_data:
        asset_prices = asset["assetPrices"]
        asset_returns = compute_arithmetic_returns(asset_prices)
        
        # Calculate Pearson correlation coefficient
        # np.corrcoef returns a 2x2 matrix, we take the off-diagonal element [0, 1]
        correlation = np.corrcoef(port_returns, asset_returns)[0, 1]
        rho_w.append(correlation)
        
    return np.array(rho_w)


# ==========================================
# Execution with Provided JSON Data
# ==========================================
if __name__ == "__main__":
    
    # --- Data 1: Covariance Matrix Setup ---
    json_cov_data = """{
      "assets": 2,
      "assetsCovarianceMatrix": [
        [0.0025, 0.0005],
        [0.0005, 0.01]
      ],
      "portfolios": [
        {"assetsWeights": [0.5, 0.5]}
      ]
    }"""
    
    data_cov = json.loads(json_cov_data)
    cov_matrix = data_cov["assetsCovarianceMatrix"]
    weights = data_cov["portfolios"][0]["assetsWeights"]
    
    spectrum_cov = compute_spectrum_from_cov(cov_matrix, weights)
    
    print("--- Method 1: Correlation Spectrum from Covariance Matrix ---")
    print(f"Asset 1 Rho: {spectrum_cov[0]:.5f}")
    print(f"Asset 2 Rho: {spectrum_cov[1]:.5f}\n")


    # --- Data 2: Historical Prices Setup ---
    json_price_data = """{
      "assets": [
        {"assetPrices": [100, 101, 105]},
        {"assetPrices": [100, 99, 101]}
      ],
      "portfolios": [
        {"portfolioValues": [100, 100.5, 101]}
      ]
    }"""
    
    data_prices = json.loads(json_price_data)
    assets_info = data_prices["assets"]
    portfolio_info = data_prices["portfolios"][0]
    
    spectrum_prices = compute_spectrum_from_prices(assets_info, portfolio_info)
    
    print("--- Method 2: Correlation Spectrum from Historical Prices ---")
    print(f"Asset 1 Rho: {spectrum_prices[0]:.5f}")
    print(f"Asset 2 Rho: {spectrum_prices[1]:.5f}")
