import numpy as np
from scipy.linalg import eigh, fractional_matrix_power
from sklearn.cluster import KMeans
import warnings

class SpectralAssetClusterer:
    def __init__(self, method='symmetric_sponge', k=None, tau_p=1.0, tau_m=1.0, max_auto_k=15):
        """
        Initializes the Spectral Asset Clusterer.
        
        Parameters:
        - method: 'blockbuster', 'sponge', or 'symmetric_sponge' (default).
        - k: Number of clusters. If None, automatically computes k using the eigengap heuristic.
        - tau_p, tau_m: Regularization parameters for the SPONGE methods.
        - max_auto_k: Maximum number of clusters to consider if k is None.
        """
        valid_methods = ['blockbuster', 'sponge', 'symmetric_sponge']
        if method.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        self.method = method.lower()
        self.k = k
        self.tau_p = tau_p
        self.tau_m = tau_m
        self.max_auto_k = max_auto_k

    def _auto_compute_k(self, evals):
        """Proprietary heuristic to compute k using the maximum eigengap."""
        # Calculate the absolute differences between consecutive eigenvalues
        gaps = np.abs(np.diff(evals))
        # Find the index of the largest gap within the allowed range
        search_range = min(self.max_auto_k, len(gaps))
        optimal_k = np.argmax(gaps[:search_range]) + 1
        return max(2, optimal_k) # Enforce at least 2 clusters

    def fit_predict(self, C):
        """
        Computes the partition of the asset universe into clusters.
        
        Parameters:
        - C: n x n numpy array representing the asset correlation matrix.
        
        Returns:
        - labels: 1D numpy array of cluster assignments.
        """
        n = C.shape[0]
        
        # Copy matrix to avoid modifying the original and remove self-loops
        A = np.copy(C)
        np.fill_diagonal(A, 0)
        
        if self.method == 'blockbuster':
            # Blockbuster standard: rely on the dominant eigenvectors of the correlation matrix
            evals, evecs = eigh(A)
            # Sort descending for largest eigenvalues
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            if self.k is None:
                self.k = self._auto_compute_k(evals)
                
            Y = evecs[:, :self.k]
            
        elif self.method in ['sponge', 'symmetric_sponge']:
            # Decompose into positive and negative adjacency matrices
            A_p = np.maximum(A, 0)
            A_m = -np.minimum(A, 0)
            
            D_p = np.diag(np.sum(A_p, axis=1))
            D_m = np.diag(np.sum(A_m, axis=1))
            
            L_p = D_p - A_p
            L_m = D_m - A_m
            
            if self.method == 'sponge':
                Matrix_num = L_p + self.tau_m * D_m
                Matrix_den = L_m + self.tau_p * D_p
            else:
                # Symmetric SPONGE
                D_bar = D_p + D_m
                diag_D = np.diag(D_bar)
                
                # Safe inverse square root
                d_inv_sqrt = np.zeros(n)
                mask = diag_D > 1e-12
                d_inv_sqrt[mask] = 1.0 / np.sqrt(diag_D[mask])
                D_inv_sqrt = np.diag(d_inv_sqrt)
                
                L_sym_p = D_inv_sqrt @ L_p @ D_inv_sqrt
                L_sym_m = D_inv_sqrt @ L_m @ D_inv_sqrt
                D_sym_p = D_inv_sqrt @ D_p @ D_inv_sqrt
                D_sym_m = D_inv_sqrt @ D_m @ D_inv_sqrt
                
                Matrix_num = L_sym_p + self.tau_m * D_sym_m
                Matrix_den = L_sym_m + self.tau_p * D_sym_p
            
            # Solve the generalized eigenvalue problem by transforming it to standard form
            # (Den^-0.5 * Num * Den^-0.5)
            Matrix_den_reg = Matrix_den + np.eye(n) * 1e-8 # Add small regularization
            den_inv_sqrt = np.real(fractional_matrix_power(Matrix_den_reg, -0.5))
            
            M = den_inv_sqrt @ Matrix_num @ den_inv_sqrt
            M = (M + M.T) / 2 # Ensure strict symmetry to avoid complex numerical artifacts
            
            evals, evecs = eigh(M)
            # Sort ascending for smallest eigenvalues (minimization problem)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            if self.k is None:
                self.k = self._auto_compute_k(evals)
                
            Y = evecs[:, :self.k]

        # Row-normalize the eigenvectors (Standard step in spectral clustering)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        # Avoid division by zero
        Y_norm = np.divide(Y, norms, out=np.zeros_like(Y), where=(norms != 0))
        
        # Final K-Means clustering in the embedded space
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Y_norm)
        
        return labels

# Example Usage:
# np.random.seed(42)
# C_dummy = np.corrcoef(np.random.randn(100, 50)) # 100 assets, 50 observations
# clusterer = SpectralAssetClusterer(method='symmetric_sponge', k=None)
# clusters = clusterer.fit_predict(C_dummy)
# print(f"Detected {clusterer.k} clusters.")
