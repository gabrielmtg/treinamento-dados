import os
import random
import subprocess
import statistics
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from validate_model import parse_weights, predict
from separa_dados import obter_arquivos, WINDOW_SIZE
from feature_engineering import extract_raw_windows_from_file, compute_zscore_params, apply_zscore, export_zscore_params_c

def run_subprocess(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}")

def write_fann_train(amostras, filepath):
    # amostras eh (feats, label). feats eh tamanho 50.
    num_feats = len(amostras[0][0])
    with open(filepath, "w") as f:
        f.write(f"{len(amostras)} {num_feats} 1\n")
        for inp, out in amostras:
            f.write(" ".join(f"{v:.6f}" for v in inp) + "\n")
            f.write(f"{out:.1f}\n")

def get_metrics_at_threshold(y_true, y_pred_probs, threshold):
    y_pred = [1.0 if p >= threshold else 0.0 for p in y_pred_probs]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1.0 and yp == 1.0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0.0 and yp == 0.0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0.0 and yp == 1.0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1.0 and yp == 0.0)
    
    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    prec = tp / max((tp + fp), 1)
    rec = tp / max((tp + fn), 1)
    fpr = fp / max((fp + tn), 1)
    return acc, prec, rec, fpr, tn, fp, fn, tp

def plot_distributions(train_samples, val_samples, save_path):
    # flatten arrays to look at overall density shift
    # Or just select the first Feature column (Branch Misses Mean)
    tr_col_0 = [s[0][0] for s in train_samples]
    val_col_0 = [s[0][0] for s in val_samples]
    
    plt.figure(figsize=(8,5))
    sns.kdeplot(tr_col_0, label="Train Feature 0", fill=True, color='blue', alpha=0.3)
    sns.kdeplot(val_col_0, label="Valid Feature 0", fill=True, color='red', alpha=0.3)
    plt.title("Z-Score Distribution Shift (Feature 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_experiment(train_samples, val_samples, test_name, base_path):
    train_path = os.path.join(base_path, "dados_treino_80.train")
    weights_path = os.path.join(base_path, "pesos_treinados.h")
    
    # 1. Z-Score normalization based ON TRAINING DATA
    mu, sigma = compute_zscore_params(train_samples)
    export_zscore_params_c(mu, sigma, os.path.join(base_path, "zscore_params.h"))
    
    train_norm = apply_zscore(train_samples, mu, sigma)
    val_norm = apply_zscore(val_samples, mu, sigma) if val_samples else []
    
    # 2. Write to FANN and Train
    write_fann_train(train_norm, train_path)
    run_subprocess(["./treina_80_fast"], cwd=base_path)
    
    weights = parse_weights(weights_path)
    if not val_norm: return None, None
    
    # 3. Predict & Tuning
    X_val = [x[0] for x in val_norm]
    y_true = [x[1] for x in val_norm]
    y_pred_probs = predict(X_val, weights)
    
    # FPR-Aware Threshold Tuning on Training (find threshold for < 0.01 FPR on train)
    X_tr = [x[0] for x in train_norm]
    y_tr = [x[1] for x in train_norm]
    tr_probs = predict(X_tr, weights)
    fpr_tr, _, threshs = roc_curve(y_tr, tr_probs)
    
    # pick the lowest threshold that gives FPR < 0.01 on training
    target_thresh = 0.95
    for f, t in reversed(list(zip(fpr_tr, threshs))):
        if f <= 0.01:
            target_thresh = min(0.99, max(0.01, t))
            break
            
    metrics = get_metrics_at_threshold(y_true, y_pred_probs, target_thresh)
    return metrics, target_thresh

def format_metrics(m, t):
    return f"Acc={m[0]:.4f} | Rec={m[2]:.4f} | FPR={m[3]:.4f} (TP:{m[7]}, FP:{m[5]}, FN:{m[6]}, TN:{m[4]}) @ Thresh={t:.3f}"

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_path, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Compiling C trainer...")
    subprocess.run(["gcc", "train_80.c", "-o", "treina_80_fast", "-lfann", "-lm"], cwd=base_path)
        
    ataques = obter_arquivos("ataques")
    benchs = obter_arquivos("benchmarks")
    all_files = ataques + benchs
    
    file_data = {}
    for f in all_files:
        file_data[f] = extract_raw_windows_from_file(f)

    report_content = "# Generalization & Feature Optimization Report\n\n"
    report_content += "Using 50 Temporal Features (Raw + Mean + Std + Min + Max + Delta) alongside Static Z-score bounds mapping.\n\n"

    # --- Leave-One-Benchmark-Out ---
    print("\n[Leave-One-Benchmark-Out]")
    report_content += "### Generalization Test: Leave-One-Benchmark-Out\n"
    for b_file in benchs:
        if len(file_data[b_file]) == 0: continue
        
        train_samples = []
        for f in all_files:
            if f != b_file: train_samples.extend(file_data[f])
            
        m, t = run_experiment(train_samples, file_data[b_file], f"LOBO_{os.path.basename(b_file)}", base_path)
        report_content += f"- Left out **{os.path.basename(b_file)}**: {format_metrics(m, t)}\n"
        print(f"LOBO {os.path.basename(b_file)} -> FPR={m[3]:.4f}")
        
    # --- Leave-One-Attack-Out ---
    print("\n[Leave-One-Attack-Type-Out]")
    report_content += "\n### Generalization Test: Leave-One-Attack-Out\n"
    for a_file in ataques:
        if len(file_data[a_file]) == 0: continue
        
        train_samples = []
        for f in all_files:
            if f != a_file: train_samples.extend(file_data[f])
            
        m, t = run_experiment(train_samples, file_data[a_file], f"LOAO_{os.path.basename(a_file)}", base_path)
        report_content += f"- Left out **{os.path.basename(a_file)}**: {format_metrics(m, t)}\n"
        print(f"LOAO {os.path.basename(a_file)} -> Rec={m[2]:.4f}")
        
    # Generate density plot for random final configuration
    plot_distributions(train_samples, file_data[a_file], os.path.join(plots_dir, "zscore_distribution_shift.png"))
    
    with open(os.path.join(base_path, "generalization_report.md"), "w") as f:
        f.write(report_content)
        
    print(f"\nReport written to generalization_report.md!")

if __name__ == "__main__":
    main()
