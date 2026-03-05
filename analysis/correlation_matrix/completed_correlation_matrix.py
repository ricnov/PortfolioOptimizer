import numpy as np
import cvxpy as cp
import warnings

class CorrelationMatrixCompleter:
    def __init__(self, tol=1e-6, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def complete_exact(self, A):
        """
        Computes the maximum determinant completion using strict constraints.
        Uses CLARABEL for superior precision in Semi-Definite Programming.
        Returns None if the feasible set is empty.
        """
        n = A.shape[0]
        C = cp.Variable((n, n), symmetric=True)
        mask = ~np.isnan(A)
        
        # Constraints: Positive Semi-Definite (>> 0) and Unit Diagonal
        constraints = [C >> 0, cp.diag(C) == 1]
        
        # Match known correlations precisely (Hard Constraints)
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    constraints.append(C[i, j] == A[i, j])
                    
        # Objective: Maximize log(det(C))
        prob = cp.Problem(cp.Maximize(cp.log_det(C)), constraints)
        
        try:
            # CLARABEL is an interior-point solver highly optimized for log_det and SDP
            prob.solve(solver=cp.CLARABEL)
            if prob.status in ["optimal", "optimal_inaccurate"] and C.value is not None:
                return C.value
        except cp.error.SolverError:
            pass
            
        return None

    def _get_minimally_altered_matrix(self, A):
        """
        Finds the nearest valid correlation matrix to the specified entries of A
        minimizing the Frobenius distance for known elements.
        """
        n = A.shape[0]
        X = cp.Variable((n, n), symmetric=True)
        mask = ~np.isnan(A)
        
        constraints = [X >> 0, cp.diag(X) == 1]
        
        # Objective: Minimize sum of squared differences for known elements
        obj_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    obj_terms.append(X[i, j] - A[i, j])
                    
        if not obj_terms:
            return np.eye(n)
            
        prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.vstack(obj_terms))), constraints)
        prob.solve(solver=cp.CLARABEL)
        
        if X.value is not None:
            # Reconstruct A' using the closest valid values
            A_prime = np.full((n, n), np.nan)
            A_prime[mask] = X.value[mask]
            np.fill_diagonal(A_prime, 1.0)
            return A_prime
            
        raise ValueError("Could not compute a minimally altered valid matrix.")

    def complete_proprietary(self, A):
        """
        The 'Proprietary' method logic:
        1. Try exact completion.
        2. If infeasible, find a minimally altered copy A' and complete that.
        """
        C_star = self.complete_exact(A)
        
        if C_star is not None:
            return C_star
            
        # Fallback for empty feasible set
        A_prime = self._get_minimally_altered_matrix(A)
        return self.complete_exact(A_prime)

    def complete_heuristic(self, A):
        """
        The 'Approximate' method based on alternating projections 
        (van der Schans and Boer, 2013). Scales extremely well with n.
        """
        n = A.shape[0]
        mask = ~np.isnan(A)
        
        # Initialize missing values to 0
        C = np.copy(A)
        C[~mask] = 0.0
        np.fill_diagonal(C, 1.0)
        
        for iteration in range(self.max_iter):
            # 1. Eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            
            # 2. Project onto Positive Semi-Definite cone (clip negatives)
            eigvals_clipped = np.maximum(eigvals, 1e-8)
            C_new = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            
            # 3. Project onto Unit Diagonal and Known Values
            d = np.sqrt(np.diag(C_new))
            C_new = (C_new / d[:, None]) / d[None, :] # Re-normalize to unit diagonal
            C_new[mask] = A[mask]                     # Re-apply known correlations
            np.fill_diagonal(C_new, 1.0)              # Strictly enforce diagonal
            
            # 4. Check convergence
            if np.linalg.norm(C_new - C, ord='fro') < self.tol:
                return C_new
                
            C = C_new
            
        warnings.warn("Heuristic did not fully converge within max_iter.")
        return C

    def is_positive_semidefinite(self, C, tol=1e-8):
        """
        Verifies if a completed matrix is positive semi-definite by checking
        if its smallest eigenvalue is greater than or equal to -tol.
        """
        eigvals = np.linalg.eigvalsh(C)
        min_eigval = np.min(eigvals)
        is_psd = min_eigval >= -tol
        return is_psd, min_eigval


# --- Testing the Implementation ---
if __name__ == "__main__":
    # The test case from your previous prompt
    A_symmetric = np.array([
        [1.0,  0.95, 0.95],
        [0.95, 1.0,  np.nan],
        [0.95, np.nan, 1.0]
    ])

    completer = CorrelationMatrixCompleter()
    
    print("--- 1. Enhanced Proprietary Method (CLARABEL Solver) ---")
    C_exact = completer.complete_proprietary(A_symmetric)
    print("Completed Matrix:\n", np.round(C_exact, 4))
    
    is_psd, min_eig = completer.is_positive_semidefinite(C_exact)
    print(f"Is PSD? {is_psd} (Minimum Eigenvalue: {min_eig:.2e})\n")

    print("--- 2. Heuristic Method (van der Schans & Boer) ---")
    C_approx = completer.complete_heuristic(A_symmetric)
    print("Completed Matrix:\n", np.round(C_approx, 4))
    
    is_psd_approx, min_eig_approx = completer.is_positive_semidefinite(C_approx)
    print(f"Is PSD? {is_psd_approx} (Minimum Eigenvalue: {min_eig_approx:.2e})")
