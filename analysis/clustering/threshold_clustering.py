import numpy as np
import pandas as pd

def fast_threshold_clustering(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> list:
    """
    Fast Threshold Clustering Algorithm (FTCA)
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        An n x n asset correlation matrix. The matrix is not required to be 
        positive semi-definite.
    threshold : float, default 0.5
        The correlation threshold to group similar assets. Higher thresholds 
        result in more clusters, while lower thresholds result in fewer.
        
    Returns:
    --------
    clusters : list of lists
        A list where each element is a list of asset names belonging to the same cluster.
    """
    # Ensure the input is a dataframe to easily track asset names
    if not isinstance(corr_matrix, pd.DataFrame):
        corr_matrix = pd.DataFrame(corr_matrix)
        
    assets = list(corr_matrix.columns)
    C = corr_matrix.values
    
    # Keep track of assets that haven't been assigned to a cluster yet
    unassigned = set(range(len(assets)))
    clusters = []
    
    while unassigned:
        # Condition 1: If only one asset remains, it forms its own cluster
        if len(unassigned) == 1:
            remaining_asset = unassigned.pop()
            clusters.append([assets[remaining_asset]])
            break
            
        unassigned_list = list(unassigned)
        
        # Calculate the Average Correlation of each unassigned asset to all OTHER unassigned assets
        avg_corrs = {}
        for i in unassigned_list:
            other_unassigned = [x for x in unassigned_list if x != i]
            # Mean correlation to the rest of the unassigned universe
            avg_corrs[i] = np.mean([C[i, j] for j in other_unassigned])
            
        # Find the asset with the Highest Average Correlation (HC) and Lowest (LC)
        hc_idx = max(avg_corrs, key=avg_corrs.get)
        lc_idx = min(avg_corrs, key=avg_corrs.get)
        
        # Condition 2: Check if HC and LC are highly correlated
        if C[hc_idx, lc_idx] > threshold:
            # Add a new cluster made of both HC and LC
            new_cluster = [hc_idx, lc_idx]
            unassigned.remove(hc_idx)
            unassigned.remove(lc_idx)
            
            # Find all other unassigned assets that have an average correlation to HC and LC > threshold
            to_add = []
            for i in list(unassigned):
                avg_corr_hc_lc = (C[i, hc_idx] + C[i, lc_idx]) / 2.0
                if avg_corr_hc_lc > threshold:
                    to_add.append(i)
                    
            # Add them to the cluster and remove from unassigned pool
            for i in to_add:
                new_cluster.append(i)
                unassigned.remove(i)
                
            clusters.append([assets[idx] for idx in new_cluster])
            
        # Condition 3: HC and LC are NOT highly correlated
        else:
            # 3a. Create a cluster made of HC
            hc_cluster = [hc_idx]
            unassigned.remove(hc_idx)
            
            to_add_hc = []
            for i in list(unassigned):
                if C[i, hc_idx] > threshold:
                    to_add_hc.append(i)
                    
            for i in to_add_hc:
                hc_cluster.append(i)
                unassigned.remove(i)
                
            clusters.append([assets[idx] for idx in hc_cluster])
            
            # 3b. Create a cluster made of LC (if LC hasn't been swept up)
            # Note: LC is guaranteed to still be in `unassigned` because C[hc, lc] <= threshold
            if lc_idx in unassigned:
                lc_cluster = [lc_idx]
                unassigned.remove(lc_idx)
                
                to_add_lc = []
                for i in list(unassigned):
                    if C[i, lc_idx] > threshold:
                        to_add_lc.append(i)
                        
                for i in to_add_lc:
                    lc_cluster.append(i)
                    unassigned.remove(i)
                    
                clusters.append([assets[idx] for idx in lc_cluster])

    return clusters
