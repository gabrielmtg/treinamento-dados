import os
import sys
import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_path, "..", "phase2")))

from stream_simulator import stream_fold_5
from windowing import augment_window_dynamic, get_contamination_ratio

from deployment_harness import generate_python_inferencer
from feature_engineering import apply_safeguard_transforms

def load_phase2_model(models_dir):
    NUM_INPUTS = 25
    BEST_H1 = 16
    parser, predictor = generate_python_inferencer(NUM_INPUTS, BEST_H1)
    weights = parser(os.path.join(models_dir, "pesos_treinados.h"))
    return predictor, weights

def retrieve_baseline_stats(benign_buffer, W):
    calib_feats = []
    for i in range(len(benign_buffer) - W + 1):
        window_slice = benign_buffer[i : i+W]
        calib_feats.append(augment_window_dynamic(window_slice))
    arr = np.array(calib_feats)
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)
    sigma[sigma == 0] = 1e-6
    return mu, sigma

def simulate_online_inference():
    print("=== Phase 3: Online Inference with Temporal Aggregation Sweep ===")
    phase2_dir = os.path.abspath(os.path.join(base_path, "..", "phase2"))
    models_dir = phase2_dir
    predictor, weights = load_phase2_model(models_dir)
    
    TARGET_THRESH = 0.5 
    THETA = 3.5
    
    # We sweep multiple window sizes using the single W=5 topology!
    W_SWEEP = [3, 5, 10, 20]
    
    os.makedirs(os.path.join(base_path, "..", "..", "results", "phase3"), exist_ok=True)
    
    for W in W_SWEEP:
        print(f"\n--- Running Sweep for Window Size: {W} ---")
        
        stream = stream_fold_5(os.path.join(base_path, "..", "..", "data", "validacao_20_porcento.csv"))
        logs = []
        buffer = []
        benign_init_buffer = []
        CALIBRATION_LIMIT = 500
        
        baseline_mu, baseline_sigma = None, None
        
        t = 0
        t_a = None
        t_e = None
        t_d = None
        t_s = None
        consecutive_dets = 0
        
        for raw_sample in stream:
            t += 1
            label = raw_sample[-1]
            
            if t <= CALIBRATION_LIMIT:
                benign_init_buffer.append(raw_sample)
                if t == CALIBRATION_LIMIT:
                    baseline_mu, baseline_sigma = retrieve_baseline_stats(benign_init_buffer, W)
                continue
            
            buffer.append(raw_sample)
            if len(buffer) > W:
                buffer.pop(0)
                
            if len(buffer) == W:
                current_window_feats = augment_window_dynamic(buffer)
                contam_ratio = get_contamination_ratio(buffer)
                current_label = label
                
                # We define t_a as the first time standard physical attack begins arriving in buffer at all
                if current_label >= 1.0 and t_a is None:
                    t_a = t
                    
                z_scores = (np.array(current_window_feats) - baseline_mu) / baseline_sigma
                D_t = np.max(np.abs(z_scores))
                
                if D_t > THETA and t_e is None and t_a is not None:
                    t_e = t
                    
                prob = predictor([current_window_feats], weights)[0]
                pred = 1.0 if prob > TARGET_THRESH else 0.0
                
                if pred == 1.0 and t_a is not None:
                    if t_d is None: t_d = t
                    consecutive_dets += 1
                    if consecutive_dets >= 3 and t_s is None:
                        t_s = t
                else:
                    consecutive_dets = 0
                    
                lat_ms = (t - t_a) if t_a is not None else 0
                
                logs.append([t, current_label, pred, lat_ms, contam_ratio, D_t, prob])
        
        print(f"Metrics (W={W}): t_a={t_a}, t_e={t_e}, t_d={t_d}, stable_t_s={t_s}")
        df_logs = pd.DataFrame(logs, columns=["Timestamp", "GT_t", "Prediction", "Latency_ms", "Contamination_Ratio", "Deviation_Score", "Model_Prob"])
        df_logs.to_csv(os.path.join(base_path, "..", "..", "results", "phase3", f"metrics_W{W}.csv"), index=False)

if __name__ == "__main__":
    simulate_online_inference()
