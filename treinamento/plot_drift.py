import os
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import time_to_ms, augment_window, apply_safeguard_transforms
from deployment_harness import generate_python_inferencer

def parse_zscore_params(filepath):
    with open(filepath, 'r') as f: content = f.read()
    matches = re.search(r'#define DECISION_THRESHOLD ([-+]?\d*\.\d+)f', content)
    thresh = float(matches.group(1)) if matches else 0.04000
    arrays = re.findall(r'\{(.*?)\}', content, re.DOTALL)
    c_mins = [float(x.strip('f \n,')) for x in arrays[0].split(',') if x.strip()]
    c_maxs = [float(x.strip('f \n,')) for x in arrays[1].split(',') if x.strip()]
    means = [float(x.strip('f \n,')) for x in arrays[2].split(',') if x.strip()]
    stds = [float(x.strip('f \n,')) for x in arrays[3].split(',') if x.strip()]
    return thresh, c_mins, c_maxs, means, stds

def extract_raw_windows_from_csv(arquivo):
    amostras = []
    dados = []
    with open(arquivo, 'r') as f:
        prev_time_ms = 0
        for linha in f:
            l = linha.replace(',', ' ').strip()
            if l and l[0].isdigit():
                partes = l.split()
                if len(partes) >= 6:
                    tempo_str = partes[0]
                    cycles, instr, cache, branch = map(float, partes[1:5])
                    label = 1.0 if float(partes[5]) > 0.0 else 0.0
                    curr_ms = time_to_ms(tempo_str)
                    delta_ms = curr_ms - prev_time_ms if prev_time_ms > 0 else 0.0
                    prev_time_ms = curr_ms
                    dados.append([branch, cache, instr, cycles, delta_ms, label])

    WINDOW_SIZE = 5
    print(f"Read {len(dados)} target lines")
    if len(dados) < WINDOW_SIZE: return amostras
    
    # Cap processing for visualization speed while covering a huge range
    for i in range(min(50000, len(dados) - WINDOW_SIZE + 1)):
        window_slice = dados[i:i+WINDOW_SIZE]
        raw_rows = [x[:5] for x in window_slice]
        features = augment_window(raw_rows)
        label_real = window_slice[-1][5]
        amostras.append((features, label_real))
    return amostras

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "..", "data", "data_final_clean.csv")
    
    print("Reading sampled bounds...")
    raw_samples = extract_raw_windows_from_csv(data_path)
    thresh, c_mins, c_maxs, means, stds = parse_zscore_params(os.path.join(base_path, "zscore_params.h"))

    parser, predictor = generate_python_inferencer(25, 20)
    w = parser(os.path.join(base_path, "pesos_treinados.h"))

    just_feats = [(s[0], s[1]) for s in raw_samples]
    norm_samples = apply_safeguard_transforms(just_feats, c_mins, c_maxs, means, stds)
    X = [x[0] for x in norm_samples]
    y_true = [x[1] for x in norm_samples]

    y_pred_prob = predictor(X, w)

    # 1. Probability Plot
    plt.figure(figsize=(10, 6))
    v_b = [p for y, p in zip(y_true, y_pred_prob) if y == 0.0]
    v_a = [p for y, p in zip(y_true, y_pred_prob) if y == 1.0]

    if len(v_b) > 0: sns.kdeplot(v_b, label="New Benign Data Classification", color="blue", fill=True, clip=(0.0, 1.0))
    if len(v_a) > 0: sns.kdeplot(v_a, label="New Attack Data Classification", color="red", fill=True, clip=(0.0, 1.0))
    plt.axvline(x=thresh, color='black', linestyle='--', label=f"Established Decision Threshold ({thresh:.3f})")
    
    plt.title("Failure Matrix: Why False Positive Rate = 1.0")
    plt.xlabel("Model Probability Assignment")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "prob_drift.png"))
    plt.close()

    # 2. Raw Drift Plot for Central Logic Justification
    # Mean CPU Cycles is Feature Index 15 (branch0, cache1, instr2, cycles3, deltams4) -> cycles3 is pos 15 in flat array of means, stds, mins, maxs, deltas
    mean_cycles_benign = [s[0][15] for s in raw_samples if s[1] == 0.0]
    old_min = c_mins[15]
    old_max = c_maxs[15]

    plt.figure(figsize=(10, 6))
    sns.histplot(mean_cycles_benign, color="orange", label="New Benign CPU Cycles Output", bins=50)
    plt.axvline(x=old_max, color='red', linestyle='--', linewidth=3, label=f"Authorized Max ({old_max/1e6:.1f}M Cycles)")
    plt.axvline(x=old_min, color='red', linestyle=':', linewidth=3, label=f"Authorized Min ({old_min/1e6:.1f}M Cycles)")
    
    plt.title("Data Shift Root Cause Analysis: Feature Distribution Collapse")
    plt.xlabel("Average CPU Cycles per Window")
    plt.ylabel("Frequency Count")
    plt.legend(loc="center")
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "feature_drift.png"))
    plt.close()
    
    print("Plots fully rendered!")

if __name__ == '__main__':
    main()
