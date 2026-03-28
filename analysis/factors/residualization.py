import numpy as np

def compute_residualized_factor(X, target_index):
    """
    Computes the residuals of a factor against a set of other factors.
    
    Parameters:
    -----------
    X : numpy.ndarray
        A factor returns matrix of shape (m, T), where m is the number 
        of factors and T is the number of time periods.
    target_index : int
        The row index (0 to m-1) of the factor to be residualized (Xi).
        
    Returns:
    --------
    R_res_i : numpy.ndarray
        The residualized returns of the target factor, shape (T,).
    alpha : float
        The intercept of the regression.
    beta : numpy.ndarray
        The regression coefficients, shape (m-1,).
    """
    m, T = X.shape
    
    # 1. Extract the target factor Xi (shape: T,)
    X_i = X[target_index, :]
    
    # 2. Extract the remaining factors X_{-i} (shape: m-1, T)
    X_minus_i = np.delete(X, target_index, axis=0)
    
    # 3. Prepare the design matrix for regression
    # Transpose X_{-i} to shape (T, m-1) so columns are features (factors) and rows are observations (time)
    X_indep = X_minus_i.T
    
    # Add a column of ones to account for the intercept (alpha)
    # The design matrix shape becomes (T, m)
    X_design = np.column_stack([np.ones(T), X_indep])
    
    # 4. Solve the linear least squares problem
    # np.linalg.lstsq with rcond=None guarantees the minimum Euclidean norm solution 
    # for the coefficients in case of multicollinearity/rank-deficiency.
    # theta contains [alpha, beta_1, beta_2, ..., beta_{m-1}]
    theta, _, _, _ = np.linalg.lstsq(X_design, X_i, rcond=None)
    
    # Extract alpha and beta
    alpha = theta[0]
    beta = theta[1:]
    
    # 5. Compute the predicted values and the residuals
    # R_{res, i} = X_i - (alpha + beta^T X_{-i})
    predicted_X_i = X_design.dot(theta)
    R_res_i = X_i - predicted_X_i
    
    return R_res_i, alpha, beta

# ==========================================
# Example Usage with Synthetic Data
# ==========================================
if __name__ == "__main__":
    # Define dimensions
    m = 4    # 4 factors (e.g., Market, Size, Value, Momentum)
    T = 100  # 100 time periods (e.g., daily returns)
    
    # Generate a synthetic matrix of factor returns X \in R^{m \times T}
    # Setting a random seed for reproducibility
    np.random.seed(42)
    X = np.random.normal(0.001, 0.02, (m, T)) 
    
    # Let's introduce some artificial collinearity (Factor 0 drives Factor 1)
    X[1, :] = 1.5 * X[0, :] + np.random.normal(0, 0.005, T)
    
    # Target factor to residualize (e.g., Factor 1)
    target_factor_idx = 1
    
    # Compute residuals
    residuals, alpha, beta = compute_residualized_factor(X, target_factor_idx)
    
    print(f"Original Variance of Factor {target_factor_idx}: {np.var(X[target_factor_idx, :]):.6f}")
    print(f"Residualized Variance of Factor {target_factor_idx}: {np.var(residuals):.6f}")
    print(f"Alpha: {alpha:.6f}")
    print(f"Betas (against other factors): {beta}")
