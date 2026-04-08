import os
import sys
import numpy as np
import subprocess
import re
import math

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_path, "..", "phase2")))

from stream_simulator import stream_fold_5
from deployment_harness import generate_python_inferencer

# Monkey-patch or re-parse weights locally
def parse_weights_float32(filepath, num_inputs=25, h1=16, h2=8):
    with open(filepath, 'r') as f: content = f.read()
    matches = re.findall(r'([-+]?\d*\.\d+)f', content)
    pesos = [np.float64(np.float32(float(x))) for x in matches]
    p = 0
    W1 = np.zeros((num_inputs, h1), dtype=np.float64); b1 = np.zeros(h1, dtype=np.float64)
    for i in range(h1):
        if p >= len(pesos): break
        b1[i] = pesos[p]; p += 1
        for j in range(num_inputs):
            if p >= len(pesos): break
            W1[j, i] = pesos[p]; p += 1

    W2 = np.zeros((h1, h2), dtype=np.float64); b2 = np.zeros(h2, dtype=np.float64)
    for i in range(h2):
        if p >= len(pesos): break
        b2[i] = pesos[p]; p += 1
        for j in range(h1):
            if p >= len(pesos): break
            W2[j, i] = pesos[p]; p += 1

    W3 = np.zeros((h2, 1), dtype=np.float64); b3 = np.zeros(1, dtype=np.float64)
    for i in range(1):
        if p >= len(pesos): break
        b3[i] = pesos[p]; p += 1
        for j in range(h2):
            if p >= len(pesos): break
            W3[j, i] = pesos[p]; p += 1
    return W1, b1, W2, b2, W3, b3

def augment_window_dynamic_baseline(buffer_rows):
    pure_features = [row[:-1] if len(row) == 6 else row for row in buffer_rows]
    channels = list(zip(*pure_features))
    
    features = []
    for ch in channels:
        W = len(ch)
        mean_val = sum(ch) / W
        variance = sum((x - mean_val) ** 2 for x in ch) / W
        
        if variance < 0.0: variance = 0.0
        
        std_val = math.sqrt(variance)
        min_val = min(ch)
        max_val = max(ch)
        delta_val = ch[-1] - ch[0]
        
        features.extend([mean_val, std_val, min_val, max_val, delta_val])
        
    return features

def get_static_params():
    file_p = os.path.join(base_path, "..", "phase2", "zscore_params.h")
    with open(file_p, 'r') as f:
        content = f.read()

    def ext(arr_name):
        match = re.search(f"const float {arr_name}\\[.*?\\] = {{(.*?)}};", content, re.DOTALL)
        if not match: return np.zeros(25)
        num_strs = match.group(1).split(',')
        return np.array([np.float64(np.float32(float(x.replace('f','').strip()))) for x in num_strs if x.strip() != ''], dtype=np.float64)

    match_t = re.search(r"#define DEPLOYMENT_THRESHOLD ([-+]?\d*\.\d+)f", content)
    thresh = float(match_t.group(1)) if match_t else 0.5

    return ext('clamp_mins'), ext('clamp_maxs'), ext('feature_means'), ext('feature_stds'), thresh

def main():
    print("Extracting first 5000 rows...", flush=True)
    stream = stream_fold_5(os.path.join(base_path, "..", "..", "data", "validacao_20_porcento.csv"))
    
    samples = []
    for _ in range(5000):
        try:
            samples.append(next(stream))
        except StopIteration:
            break

    c_mins, c_maxs, means, stds, thresh = get_static_params()
    parser, predictor = generate_python_inferencer(25, 16, 8)
    weights = parse_weights_float32(os.path.join(base_path, "..", "phase2", "pesos_treinados.h"))

    print("Computing pure Python inference...", flush=True)
    py_preds = []
    buffer = []
    W = 20
    for row in samples:
        buffer.append(row)
        if len(buffer) > W:
            buffer.pop(0)
        
        feats = augment_window_dynamic_baseline(buffer)
        
        normed = []
        for i, val in enumerate(feats):
            v = val
            if v < c_mins[i]: v = c_mins[i]
            if v > c_maxs[i]: v = c_maxs[i]
            v = (v - means[i]) / stds[i]
            normed.append(v)

        prob = predictor([normed], weights)[0]
        if len(py_preds) == 0:
            print("--- First Sample Debug (Py) ---")
            print(f"Feats: {feats}")
            print(f"Normed: {normed}")
            print(f"Prob: {prob:.12e}")

        py_preds.append(float(prob))

    print("Compiling C binary...", flush=True)
    c_source_path = os.path.join(base_path, "consistency_checker_full.c")
    bin_path = os.path.join(base_path, "consistency_checker_full")
    subprocess.run(["gcc", "-O3", "-Wall", "-Wextra", "-march=native", c_source_path, "-o", bin_path, "-lm"], check=True)

    print("Running C pipeline...", flush=True)
    input_str = ""
    for r in samples:
        input_str += f"{r[0]:.9f} {r[1]:.9f} {r[2]:.9f} {r[3]:.9f} {r[4]:.9f}\n"

    res = subprocess.run([bin_path], input=input_str, text=True, capture_output=True, check=True)
    c_out = res.stdout.strip().split('\n')
    c_preds = []
    for x in c_out:
        if not x.strip(): continue
        if x.startswith("---"):
            print(x)
            continue
        if " " in x:
            print(f"C Normed: [{x.strip()}]")
            continue
        c_preds.append(float(x))

    if len(c_preds) != len(py_preds):
        print(f"Length mismatch: C={len(c_preds)}, Py={len(py_preds)}")
        return

    max_drift = 0.0
    sum_drift = 0.0
    mismatches = 0
    t = 0
    
    for c_p, py_p in zip(c_preds, py_preds):
        drift = abs(c_p - py_p)
        max_drift = max(max_drift, drift)
        sum_drift += drift
        
        c_class = 1 if c_p >= thresh else 0
        py_class = 1 if py_p >= thresh else 0
        
        if c_class != py_class:
            mismatches += 1
            print(f"Mismatch at sample {t}! C_Prob={c_p:.12e}, Py_Prob={py_p:.12e}, C_Class={c_class}, Py_Class={py_class}")
        t += 1

    mean_drift = sum_drift / len(c_preds)
    
    print("\n=== Validation Results ===")
    print(f"Maximum absolute drift: {max_drift:.8e}")
    print(f"Mean drift: {mean_drift:.8e}")
    print(f"Mismatches (Threshold={thresh}): {mismatches}")

    if max_drift < 1e-5 and mismatches == 0:
        print("SUCCESS! C Implementation perfectly matches Python theory.")
    else:
        print("FAILED validity constraints.")

if __name__ == '__main__':
    main()
