import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.stats import random_correlation

class HierarchicalAssetClustering:
    """
    Implements the clustering logic for Hierarchical-Based Asset Allocation
    as described by Thomas Raffinot.
    """
    
    def __init__(self, linkage_method='ward', n_clusters=None, max_clusters=10, n_refs=10):
        """
        Args:
            linkage_method (str): 'single', 'average', 'complete', or 'ward' (default).
            n_clusters (int): Number of clusters. If None, computes using Gap Statistic.
            max_clusters (int): Max number of clusters to test for Gap Statistic.
            n_refs (int): Number of reference datasets for Gap Statistic.
        """
        valid_methods = ['single', 'average', 'complete', 'ward']
        if linkage_method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.n_refs = n_refs

    def _correlation_to_distance(self, C):
        """Converts correlation matrix to a distance matrix."""
        # Ensure values are within [-1, 1] to avoid float precision errors
        C = np.clip(C, -1.0, 1.0)
        # Calculate distance D = sqrt(0.5 * (1 - C))
        D = np.sqrt(0.5 * (1.0 - C))
        # Fill diagonal with 0s strictly
        np.fill_diagonal(D, 0.0)
        return D

    def _get_linkage(self, C):
        """Computes the hierarchical clustering linkage matrix."""
        D = self._correlation_to_distance(C)
        # Scipy's linkage requires a condensed distance matrix
        condensed_D = ssd.squareform(D, checks=False)
        Z = sch.linkage(condensed_D, method=self.linkage_method)
        return Z

    def _compute_dispersion(self, C, labels):
        """
        Computes the within-cluster dispersion (W_k) based on the distance matrix.
        """
        D = self._correlation_to_distance(C)
        dispersion = 0.0
        n_clusters = len(np.unique(labels))
        
        for k in range(1, n_clusters + 1):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) > 1:
                # Extract the sub-distance matrix for this cluster
                sub_D = D[np.ix_(cluster_indices, cluster_indices)]
                # Sum of pairwise distances divided by 2 * size of cluster
                dispersion += np.sum(sub_D) / (2.0 * len(cluster_indices))
        
        # If dispersion is 0 (e.g., each point is its own cluster), return a tiny float
        return dispersion if dispersion > 0 else 1e-10

    def _generate_null_correlation(self, n_assets):
        """
        Generates a random positive-definite correlation matrix.
        Uses the Dirichlet distribution to generate eigenvalues that sum to n_assets,
        then uses scipy's random_correlation to generate the matrix.
        This mathematically approximates a uniform distribution of correlation matrices.
        """
        # Generate random eigenvalues that sum to n_assets
        # Alpha=1.0 for uniform distribution on the simplex
        eigenvalues = np.random.dirichlet(np.ones(n_assets)) * n_assets
        
        # random_correlation requires eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return random_correlation.rvs(eigs=eigenvalues)

    def compute_gap_statistic(self, C, Z):
        """
        Computes the Gap Statistic using the 1-standard-error rule
        to rigidly find the optimal number of clusters, preventing overfitting.
        """
        n_assets = C.shape[0]
        # max_clusters cannot exceed the number of assets
        max_k = min(self.max_clusters, n_assets)
        
        gaps = np.zeros(max_k)
        s_k = np.zeros(max_k) # Standard deviation of the null reference
        
        for k in range(1, max_k + 1):
            # 1. Compute dispersion of the actual data
            labels = sch.fcluster(Z, k, criterion='maxclust')
            W_k = self._compute_dispersion(C, labels)
            
            # 2. Compute dispersion of the null reference distribution
            ref_dispersions = np.zeros(self.n_refs)
            for i in range(self.n_refs):
                C_null = self._generate_null_correlation(n_assets)
                Z_null = self._get_linkage(C_null)
                labels_null = sch.fcluster(Z_null, k, criterion='maxclust')
                ref_dispersions[i] = self._compute_dispersion(C_null, labels_null)
            
            # 3. Calculate the gap and the standard error
            log_ref_disp = np.log(ref_dispersions)
            expected_log_W_k = np.mean(log_ref_disp)
            
            gaps[k-1] = expected_log_W_k - np.log(W_k)
            
            # Compute standard deviation scaled by sqrt(1 + 1/n_refs)
            sd = np.std(log_ref_disp, ddof=1) if self.n_refs > 1 else 0.0
            s_k[k-1] = sd * np.sqrt(1.0 + 1.0 / self.n_refs)
            
        # 4. Apply the 1-standard-error rule from Tibshirani et al.
        # Choose the smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
        optimal_k = 1  # Default fallback to 1 cluster
        for k in range(max_k - 1):
            if gaps[k] >= gaps[k+1] - s_k[k+1]:
                optimal_k = k + 1
                break
        else:
            # If the condition is never met, fallback to the absolute maximum gap
            optimal_k = np.argmax(gaps) + 1
            
        return optimal_k, gaps

    def cluster_assets(self, C, asset_names=None):
        """
        Main method to execute the clustering logic.
        """
        # Ensure C is a numpy array
        if isinstance(C, pd.DataFrame):
            if asset_names is None:
                asset_names = C.columns.tolist()
            C = C.values
            
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(C.shape[0])]

        # Step 1: Compute Linkage
        Z = self._get_linkage(C)

        # Step 2: Determine number of clusters
        if self.n_clusters is None:
            optimal_k, gaps = self.compute_gap_statistic(C, Z)
            k = optimal_k
            print(f"Optimal number of clusters computed via Gap Statistic: {k}")
        else:
            k = self.n_clusters
            print(f"Using user-provided number of clusters: {k}")

        # Step 3: Cut the hierarchical tree
        cluster_labels = sch.fcluster(Z, k, criterion='maxclust')
        
        # Format the output into a readable dictionary/dataframe mapping
        allocation_groups = pd.DataFrame({
            'Asset': asset_names,
            'Cluster': cluster_labels
        }).sort_values(by='Cluster').reset_index(drop=True)
        
        return allocation_groups, Z


if __name__ == "__main__":
  # Create a dummy correlation matrix for 5 assets
  C_dummy = np.array([
    [1,
      0.7606306078350177,
      0.15733356650676536
    ],
    [
      0.7606306078350177,
      1,
      0.7606306078350177
    ],
    [
      0.15733356650676536,
      0.7606306078350177,
      1]
  ])
  assets = ['Stock_A', 'Stock_B', 'Stock_C']

  # Initialize the allocator using default Ward linkage and Gap Statistic
  allocator = HierarchicalAssetClustering(linkage_method='ward', n_clusters=None)

  # Group the assets
  clusters, linkage_matrix = allocator.cluster_assets(C_dummy, asset_names=assets)
  print(clusters)
