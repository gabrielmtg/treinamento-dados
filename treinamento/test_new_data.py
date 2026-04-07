import os
import re
from feature_engineering import time_to_ms, augment_window, apply_safeguard_transforms
from deployment_harness import generate_python_inferencer, get_metrics_at

def extract_windows_from_csv(arquivo):
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
                        
                        binary_label = 1.0 if label > 0.0 else 0.0
                        
                        curr_ms = time_to_ms(tempo_str)
                        delta_ms = curr_ms - prev_time_ms if prev_time_ms > 0 else 0.0
                        prev_time_ms = curr_ms
                        
                        dados.append([branch, cache, instr, cycles, delta_ms, binary_label])

        WINDOW_SIZE = 5
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

def parse_zscore_params(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    matches = re.search(r'#define DECISION_THRESHOLD ([-+]?\d*\.\d+)f', content)
    thresh = float(matches.group(1)) if matches else 0.04000
    
    arrays = re.findall(r'\{(.*?)\}', content, re.DOTALL)
    c_mins = [float(x.strip('f \n,')) for x in arrays[0].split(',') if x.strip()]
    c_maxs = [float(x.strip('f \n,')) for x in arrays[1].split(',') if x.strip()]
    means = [float(x.strip('f \n,')) for x in arrays[2].split(',') if x.strip()]
    stds = [float(x.strip('f \n,')) for x in arrays[3].split(',') if x.strip()]
    
    return thresh, c_mins, c_maxs, means, stds

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "..", "data", "data_final_clean.csv")
    
    print("Extracting raw windows from clean data CSV...")
    raw_samples = extract_windows_from_csv(data_path)
    
    thresh, c_mins, c_maxs, means, stds = parse_zscore_params(os.path.join(base_path, "zscore_params.h"))
    print(f"Loaded bounds from deployment matrix. Decision Threshold: {thresh}")
    
    parser, predictor = generate_python_inferencer(25, 20)
    w = parser(os.path.join(base_path, "pesos_treinados.h"))
    
    norm_samples = apply_safeguard_transforms(raw_samples, c_mins, c_maxs, means, stds)
    X = [x[0] for x in norm_samples]
    y_true = [x[1] for x in norm_samples]
    
    print("Running 20-Neuron Base Model...")
    y_pred_prob = predictor(X, w)
    
    y_pred_class = [1.0 if p >= thresh else 0.0 for p in y_pred_prob]
    fp, fpr, rec, tp, tn, fn = get_metrics_at(y_true, y_pred_class)
    
    print(f"\n--- DATA_FINAL_CLEAN VALIDATION READOUT ---")
    print(f"Total Evaluated Windows: {len(y_true)}")
    print(f"Known Benign Traces: {tn+fp}")
    print(f"Known Attack Traces: {tp+fn}")
    print(f"\nThreats Found (TP): {tp} | Missed (FN): {fn} -> Recall: {rec:.6f}")
    print(f"Safe Allowed (TN): {tn}   | Falsely Blocked (FP): {fp} -> FPR: {fpr:.6f}")

if __name__ == '__main__':
    main()
