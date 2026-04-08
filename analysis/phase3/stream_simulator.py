import pandas as pd
import numpy as np
import math

def time_to_ms(time_str):
    # Same standard string time extraction
    parts = str(time_str).split(':')
    if len(parts) >= 3:
        h = int(parts[0])
        m = int(parts[1])
        s_parts = parts[2].split('.')
        if len(s_parts) == 2:
            s, ms = int(s_parts[0]), int(s_parts[1])
        else:
            s = int(parts[2])
            ms = int(parts[3]) if len(parts) > 3 else 0
        return (h * 3600 + m * 60 + s) * 1000 + ms
    return 0

def compute_baseline_stats(benign_data):
    """
    Computes \mu and \sigma for each feature using only a benign segment.
    """
    if len(benign_data) == 0:
        raise ValueError("Benign data segment is empty.")
    
    features_only = np.array([row[:-1] for row in benign_data])
    mu = np.mean(features_only, axis=0)
    sigma = np.std(features_only, axis=0)
    
    sigma[sigma == 0] = 1e-6
    return mu, sigma

def stream_fold_5(file_path):
    """
    Generator yielding RAW PMU samples sequentially from Fold 5.
    We compute delta_ms natively here to output the 5 standard channels required by Option A:
    [branch, cache, instr, cycles, delta_ms, label]
    """
    df = pd.read_csv(file_path)
    
    prev_ms = 0
    for _, row in df.iterrows():
        # CORE_ID,TIMESTAMP,CPU_CYCLES,INSTRUCTIONS,CACHE_MISSES,BRANCH_MISSES,LABEL
        t_str = row['TIMESTAMP']
        cycles = float(row['CPU_CYCLES'])
        instr = float(row['INSTRUCTIONS'])
        cache = float(row['CACHE_MISSES'])
        branch = float(row['BRANCH_MISSES'])
        label = float(row['LABEL'])
        
        curr_ms = time_to_ms(t_str)
        delta_ms = curr_ms - prev_ms if prev_ms > 0 else 0.0
        prev_ms = curr_ms
        
        # Consistent channel order with training phase 2
        yield [branch, cache, instr, cycles, delta_ms, label]
