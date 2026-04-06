import math
import os

WINDOW_SIZE = 5

def time_to_ms(time_str):
    parts = time_str.split(':')
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

def augment_window(raw_rows):
    channels = list(zip(*raw_rows)) # 5 tuples, each of length 5
    features = []
    # Drop raw values entirely to reduce topology complexity to 25.
    
    # Add temporal summaries (25 features)
    for ch in channels:
        mean_val = sum(ch) / len(ch)
        variance = sum((x - mean_val) ** 2 for x in ch) / len(ch)
        std_val = math.sqrt(variance)
        min_val = min(ch)
        max_val = max(ch)
        delta_val = ch[-1] - ch[0]
        
        features.extend([mean_val, std_val, min_val, max_val, delta_val])
    return features

def extract_raw_windows_from_file(arquivo):
    amostras = []
    try:
        dados = []
        with open(arquivo, 'r') as f:
            prev_time_ms = 0
            for linha in f:
                l = linha.replace(',', ' ').strip()
                if l and l[0].isdigit():
                    partes = l.split()
                    if len(partes) >= 6:
                        tempo_str = partes[0]
                        cycles = float(partes[1])
                        instr = float(partes[2])
                        cache = float(partes[3])
                        branch = float(partes[4])
                        label = float(partes[5])
                        
                        curr_ms = time_to_ms(tempo_str)
                        delta_ms = curr_ms - prev_time_ms if prev_time_ms > 0 else 0.0
                        prev_time_ms = curr_ms
                        
                        dados.append([branch, cache, instr, cycles, delta_ms, label])

        if len(dados) < WINDOW_SIZE: return amostras

        for i in range(len(dados) - WINDOW_SIZE + 1):
            window_slice = dados[i:i+WINDOW_SIZE]
            raw_rows = [x[:5] for x in window_slice]
            features = augment_window(raw_rows)
            label_real = window_slice[-1][5]
            amostras.append((features, label_real))

    except Exception as e:
        print(f"Erro em {os.path.basename(arquivo)}: {e}")
    return amostras

def compute_benign_bounds(train_samples):
    """Computes pure benign bounds (clamp min/max and z-score mu/sigma)."""
    benign_samples = [s for s in train_samples if s[1] == 0.0]
    if not benign_samples: return [], [], [], []
    num_features = len(benign_samples[0][0])
    
    c_mins = []
    c_maxs = []
    means = []
    stds = []
    
    for c in range(num_features):
        col = [s[0][c] for s in benign_samples]
        raw_min = min(col)
        raw_max = max(col)
        
        envelope = (raw_max - raw_min) * 0.10
        clamp_min = raw_min - envelope
        clamp_max = raw_max + envelope
        
        # Apply strict clamping to simulate the transform correctly.
        clamped_col = [max(clamp_min, min(x, clamp_max)) for x in col]
        
        mu = sum(clamped_col) / len(clamped_col)
        variance = sum((x - mu) ** 2 for x in clamped_col) / len(clamped_col)
        sigma = math.sqrt(variance)
        if sigma == 0: sigma = 1e-9
        
        c_mins.append(clamp_min)
        c_maxs.append(clamp_max)
        means.append(mu)
        stds.append(sigma)
        
    return c_mins, c_maxs, means, stds

def apply_safeguard_transforms(samples, c_mins, c_maxs, means, stds):
    transformed_samples = []
    num_features = len(means)
    for feats, label in samples:
        norm_feats = []
        for c in range(num_features):
            # 1. CLAMP
            val = max(c_mins[c], min(feats[c], c_maxs[c]))
            # 2. Z-SCORE
            norm_val = (val - means[c]) / stds[c]
            norm_feats.append(norm_val)
        transformed_samples.append((norm_feats, label))
    return transformed_samples

def export_deployment_params_c(c_mins, c_maxs, means, stds, threshold, filepath):
    with open(filepath, "w") as f:
        f.write("#ifndef ZSCORE_PARAMS_H\n#define ZSCORE_PARAMS_H\n\n")
        f.write(f"#define NUM_FEATURES {len(means)}\n")
        f.write(f"#define DECISION_THRESHOLD {threshold:.6f}f\n\n")
        
        f.write("const float clamp_mins[NUM_FEATURES] = {\n")
        for m in c_mins: f.write(f"    {m}f,\n")
        f.write("};\n\n")
        
        f.write("const float clamp_maxs[NUM_FEATURES] = {\n")
        for m in c_maxs: f.write(f"    {m}f,\n")
        f.write("};\n\n")

        f.write("const float feature_means[NUM_FEATURES] = {\n")
        for m in means: f.write(f"    {m}f,\n")
        f.write("};\n\n")
        
        f.write("const float feature_stds[NUM_FEATURES] = {\n")
        for s in stds: f.write(f"    {s}f,\n")
        f.write("};\n\n")
        
        f.write("#endif\n")
