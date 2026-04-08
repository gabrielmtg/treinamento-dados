import os
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import roc_curve, confusion_matrix

from feature_engineering import extract_raw_windows_from_file, compute_benign_bounds, apply_safeguard_transforms, export_deployment_params_c
from deployment_harness import rewrite_c_trainer, generate_python_inferencer, get_metrics_at

def run_subprocess(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}")

def write_fann_train(data, file_path):
    if not data: return
    sz = len(data[0][0])
    with open(file_path, "w") as f:
        f.write(f"{len(data)} {sz} 1\n")
        f.writelines([f"{' '.join(f'{v:.5f}' for v in i)} \n{o:.1f}\n" for i, o in data])

def get_extended_metrics(y_true, y_pred):
    fp, fpr, rec, tp, tn, fn = get_metrics_at(y_true, y_pred)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-9)
    return fpr, rec, prec, f1, fp, tn, fn, tp

def train_and_eval(base_path, tr_z, eval_vl, parser, predictor, label_shuffle=False):
    tr_copy = tr_z[:]
    if label_shuffle:
        np.random.shuffle(tr_copy)
        for i in range(len(tr_copy)):
            feat, _ = tr_copy[i]
            tr_copy[i] = (feat, float(random.choice([0.0, 1.0])))
    else:
        random.shuffle(tr_copy)

    train_path = os.path.join(base_path, "dados_treino_80.train")
    write_fann_train(tr_copy, train_path)
    run_subprocess(["./treina_80_fast"], cwd=base_path)

    weights = parser(os.path.join(base_path, "pesos_treinados.h"))
    
    tr_benign_X = [x[0] for x in tr_z if x[1] == 0.0]
    if not tr_benign_X:
        target_thresh = 0.5
    else:
        ben_probs = predictor(tr_benign_X, weights)
        target_thresh = np.percentile(ben_probs, 99.9)

    results = {}
    if eval_vl:
        X_val = [x[0] for x in eval_vl]
        y_val = [x[1] for x in eval_vl]
        val_probs = predictor(X_val, weights)
        y_pred = [1.0 if p >= target_thresh else 0.0 for p in val_probs]
        
        fpr, rec, prec, f1, fp, tn, fn, tp = get_extended_metrics(y_val, y_pred)
        results["FoldTest"] = {
            "FPR": fpr, "Recall": rec, "Prec": prec, "F1": f1,
            "FP": fp, "TN": tn, "FN": fn, "TP": tp,
            "probs": val_probs, "y_true": y_val, "thresh": target_thresh
        }
    return results, weights

def save_plots(plots_dir, results_dict):
    m = results_dict["FoldTest"]
    baseline_probs = m["probs"]
    y_baseline = m["y_true"]
    target_thresh = m["thresh"]

    plt.figure()
    fpr_roc, tpr_roc, _ = roc_curve(y_baseline, baseline_probs)
    plt.plot(fpr_roc, tpr_roc, label="ROC")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "ROC_curve.png"))
    plt.close()

    preds = [1.0 if p >= target_thresh else 0.0 for p in baseline_probs]
    cm = confusion_matrix(y_baseline, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Baseline)")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    plt.figure()
    v_b = [p for y, p in zip(y_baseline, baseline_probs) if y == 0.0]
    v_a = [p for y, p in zip(y_baseline, baseline_probs) if y == 1.0]
    if v_b: sns.kdeplot(v_b, label="Benign", fill=True, color="blue", clip=(0.0, 1.0))
    if v_a: sns.kdeplot(v_a, label="Attack", fill=True, color="red", clip=(0.0, 1.0))
    plt.axvline(x=target_thresh, color='k', linestyle='--', label="Threshold")
    plt.title("Probability Distribution (Benign vs Attack)")
    plt.xlabel("Assigned Attack Probability")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "Prediction distributions (benign vs attack).png"))
    plt.close()
    
    t_vals = np.linspace(max(0, target_thresh - 0.1), min(1.0, target_thresh + 0.1), 30)
    f_vals = []
    r_vals = []
    for tx in t_vals:
        yp_t = [1.0 if p >= tx else 0.0 for p in baseline_probs]
        _, fx, rx, _, _, _, _, _ = get_extended_metrics(y_baseline, yp_t)
        f_vals.append(fx)
        r_vals.append(rx)
        
    plt.figure()
    plt.plot(t_vals, f_vals, label="FPR")
    plt.plot(t_vals, r_vals, label="Recall")
    plt.axvline(x=target_thresh, color='r', linestyle='--', label=f"Selected T ({target_thresh:.4f})")
    plt.xlabel('Probability Threshold')
    plt.ylabel('Metric Rate')
    plt.legend()
    plt.title('Threshold vs FPR curve')
    plt.savefig(os.path.join(plots_dir, "Threshold vs FPR curve.png"))
    plt.close()

def align_labels(raw_windows):
    aligned = []
    for feats, label in raw_windows:
        # Prompt: 0 -> Benign (0.0). 2/3 -> Attack (1.0)
        if label == 0.0:
            aligned.append((feats, 0.0))
        elif label == 2.0 or label == 3.0:
            aligned.append((feats, 1.0))
    return aligned

def block_split(dataset, num_folds):
    blocks = []
    bz = len(dataset) // num_folds
    for i in range(num_folds):
        s = i * bz
        e = s + bz if i != num_folds - 1 else len(dataset)
        blocks.append(dataset[s:e])
    return blocks

def main():
    print("=== FULL VALIDATION PIPELINE STARTED ===")
    base_path = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_path, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    NUM_INPUTS = 25
    BEST_H1 = 16
    rewrite_c_trainer(base_path, NUM_INPUTS, BEST_H1)
    parser, predictor = generate_python_inferencer(NUM_INPUTS, BEST_H1)

    print("Extracting Datasets...")
    data_path = os.path.join(base_path, "..", "..", "data")
    raw_attacks = align_labels(extract_raw_windows_from_file(os.path.join(data_path, "data_final_clean.csv")))
    raw_benign1 = align_labels(extract_raw_windows_from_file(os.path.join(data_path, "data_final_clean1.csv")))
    raw_benign2 = align_labels(extract_raw_windows_from_file(os.path.join(data_path, "data_final_clean2.csv")))

    NUM_FOLDS = 5
    blocks_out = block_split(raw_attacks, NUM_FOLDS)
    blocks_b1 = block_split(raw_benign1, NUM_FOLDS)
    blocks_b2 = block_split(raw_benign2, NUM_FOLDS)
    
    csv_rows = []
    
    for fold in range(NUM_FOLDS):
        print(f"--- FOLD {fold+1} ---")
        
        vl_split = blocks_out[fold] + blocks_b1[fold] + blocks_b2[fold]
        
        tr_split = []
        for i in range(NUM_FOLDS):
            if i != fold:
                tr_split.extend(blocks_out[i])
                tr_split.extend(blocks_b1[i])
                tr_split.extend(blocks_b2[i])
        
        c_mins, c_maxs, means, stds = compute_benign_bounds(tr_split)
        
        tr_z = apply_safeguard_transforms(tr_split, c_mins, c_maxs, means, stds)
        vl_z = apply_safeguard_transforms(vl_split, c_mins, c_maxs, means, stds)
        
        tr_ben = [x for x in tr_z if x[1] == 0.0]
        tr_atk = [x for x in tr_z if x[1] == 1.0]

        # Strat 1: Baseline
        base_res, base_w = train_and_eval(base_path, tr_z, vl_z, parser, predictor)

        # Strat 2: Undersampling
        min_len = min(len(tr_ben), len(tr_atk))
        np.random.seed(42)
        idx_b = np.random.choice(len(tr_ben), min_len, replace=False)
        idx_a = np.random.choice(len(tr_atk), min_len, replace=False)
        tr_under = [tr_ben[i] for i in idx_b] + [tr_atk[i] for i in idx_a]
        under_res, _ = train_and_eval(base_path, tr_under, vl_z, parser, predictor)

        # Strat 3: Oversampling (Duplicate minority intentionally, which is typically benign)
        max_len = max(len(tr_ben), len(tr_atk))
        if len(tr_ben) < len(tr_atk):
            idx_b = np.random.choice(len(tr_ben), len(tr_atk), replace=True)
            tr_over = [tr_ben[i] for i in idx_b] + tr_atk
        else:
            idx_a = np.random.choice(len(tr_atk), len(tr_ben), replace=True)
            tr_over = tr_ben + [tr_atk[i] for i in idx_a]
            
        over_res, _ = train_and_eval(base_path, tr_over, vl_z, parser, predictor)

        # Log
        for strat, res in [("Baseline", base_res), ("Undersampling", under_res), ("Oversampling", over_res)]:
            m = res["FoldTest"]
            total = m['TP'] + m['TN'] + m['FP'] + m['FN']
            acc = (m['TP'] + m['TN']) / total if total > 0 else 0
            csv_rows.append([fold+1, strat, f"{acc:.6f}", f"{m['TP']}", f"{m['FP']}", f"{m['TN']}", f"{m['FN']}", f"{m['FPR']:.6f}", f"{m['Recall']:.6f}", f"{m['F1']:.6f}", f"{m['thresh']:.6f}"])

        # Label Shuffle (Overfit Test)
        shuff_res, _ = train_and_eval(base_path, tr_z, vl_z, parser, predictor, label_shuffle=True)
        m = shuff_res["FoldTest"]
        total = m['TP'] + m['TN'] + m['FP'] + m['FN']
        acc = (m['TP'] + m['TN']) / total if total > 0 else 0
        csv_rows.append([fold+1, "LabelShuffle", f"{acc:.6f}", f"{m['TP']}", f"{m['FP']}", f"{m['TN']}", f"{m['FN']}", f"{m['FPR']:.6f}", f"{m['Recall']:.6f}", f"{m['F1']:.6f}", f"{m['thresh']:.6f}"])
        
        # We only save plots mapping on the absolute first fold's oversampling strategy since it's the optimal.
        if fold == 0:
            save_plots(plots_dir, over_res)
            export_deployment_params_c(c_mins, c_maxs, means, stds, over_res["FoldTest"]["thresh"], os.path.join(base_path, "zscore_params.h"))

    with open(os.path.join(base_path, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Fold", "Strategy", "Accuracy", "TP", "FP", "TN", "FN", "FPR", "Recall", "F1_Score", "Threshold"])
        w.writerows(csv_rows)
        
    print("Validation Pipeline Complete! Output generated in:")
    print("- metrics.csv")
    print("- results/plots/")

if __name__ == "__main__":
    main()
