import os
import subprocess
import numpy as np

from feature_engineering import extract_raw_windows_from_file, compute_benign_bounds, apply_safeguard_transforms
from deployment_harness import generate_python_inferencer

def run_subprocess(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}")

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("Building consistency_checker.c ...")
    subprocess.run(["gcc", "consistency_checker.c", "-o", "ccheck", "-lm"], cwd=base_path)
    
    print("Loading data for edge-cases...")
    data_path = os.path.join(base_path, "..", "..", "data", "data_final_clean.csv")
    main_raw = extract_raw_windows_from_file(data_path)
    
    if not main_raw:
        print("No data found to test.")
        return
        
    # Get arbitrary cases (1st, last, and a few middle)
    test_cases = [main_raw[0], main_raw[-1]] + main_raw[len(main_raw)//2 : len(main_raw)//2 + 8]
    
    # We calculate Z-Score parameters based on training, let's just pass raw features.
    # The C code automatically does limits + Z-Score + Inference.
    # So we just need to load Python predictor parser with current weights.
    
    NUM_INPUTS = 25
    BEST_H1 = 16
    parser, predictor = generate_python_inferencer(NUM_INPUTS, BEST_H1)
    
    try:
        weights = parser(os.path.join(base_path, "pesos_treinados.h"))
    except FileNotFoundError:
        print("Error: pesos_treinados.h not found. Run full_validation_pipeline.py first.")
        return
    
    # We must manually replicate the normalization locally for Python because `predictor` takes normalized features.
    # We will read `zscore_params.h` to make sure we parse the EXACT float representations C uses (avoid float round drift).
    
    print("\n--- Running Python vs Native C Consistency Evaluations ---")
    
    with open(os.path.join(base_path, "zscore_params.h"), 'r') as f:
        h_str = f.read()
        
    import re
    c_m = [float(x.replace('f', '').strip()) for x in re.findall(r'clamp_mins.*?{(.*?)}', h_str, re.DOTALL)[0].split(',') if 'f' in x]
    c_M = [float(x.replace('f', '').strip()) for x in re.findall(r'clamp_maxs.*?{(.*?)}', h_str, re.DOTALL)[0].split(',') if 'f' in x]
    mu = [float(x.replace('f', '').strip()) for x in re.findall(r'feature_means.*?{(.*?)}', h_str, re.DOTALL)[0].split(',') if 'f' in x]
    stds = [float(x.replace('f', '').strip()) for x in re.findall(r'feature_stds.*?{(.*?)}', h_str, re.DOTALL)[0].split(',') if 'f' in x]
    
    matches = 0
    for idx, (raw_feat, label) in enumerate(test_cases):
        # 1. Evaluate Python side
        norm_py = []
        for i in range(NUM_INPUTS):
            v = max(c_m[i], min(raw_feat[i], c_M[i]))
            norm_py.append((v - mu[i]) / stds[i])
            
        py_prob = predictor([norm_py], weights)[0]
        
        # 2. Evaluate C side
        cmd = ["./ccheck"] + [str(v) for v in raw_feat]
        res = subprocess.run(cmd, cwd=base_path, text=True, capture_output=True)
        try:
            c_prob = float(res.stdout.strip())
        except ValueError:
            print(f"Failed to execute C: {res.stdout.strip()}")
            continue
            
        diff = abs(py_prob - c_prob)
        print(f"Sample {idx+1:02d} | Label: {label} | Python: {py_prob:.8f} | C: {c_prob:.8f} | Diff: {diff:.9f}")
        
        if diff < 1e-5:
            matches += 1
            
    print(f"\nFinal Consistency Match: {matches}/{len(test_cases)}")
    if matches == len(test_cases):
        print("Success! Hardware C exactly identical to Python pipeline.")
    else:
        print("Mismatch! The calculations drift.")

if __name__ == "__main__":
    main()
