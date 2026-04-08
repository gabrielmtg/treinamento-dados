import numpy as np

import math

def augment_window_dynamic(buffer_rows):
    """
    Transforms W raw PMU records into EXACTLY 25 dimensions ensuring invariant dimensionality.
    buffer_rows is a sequence of identical dimensional observations [branch, cache, instr, cycles, delta].
    """
    # Exclude label if present in row
    pure_features = [row[:-1] if len(row) == 6 else row for row in buffer_rows]
    channels = list(zip(*pure_features))  # Transpose to 5 tuples of length W
    
    features = []
    for ch in channels:
        W = len(ch)
        mean_val = sum(ch) / W
        variance = sum((x - mean_val) ** 2 for x in ch) / W
        std_val = math.sqrt(variance)
        min_val = min(ch)
        max_val = max(ch)
        delta_val = ch[-1] - ch[0]
        
        features.extend([mean_val, std_val, min_val, max_val, delta_val])
        
    return features

def get_contamination_ratio(buffer_rows):
    """
    Computes the contamination ratio: proportion of 'attack' samples (label == 1.0)
    inside the current sliding window W.
    """
    W = len(buffer_rows)
    if W == 0: return 0.0
    
    attacks = sum(1 for row in buffer_rows if row[-1] >= 1.0)
    return attacks / W

