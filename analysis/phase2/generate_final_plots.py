import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

from separa_dados import obter_arquivos
from feature_engineering import extract_raw_windows_from_file, apply_safeguard_transforms, compute_benign_bounds
from deployment_harness import generate_python_inferencer, get_metrics_at

base_path = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_path, "results", "plots")
os.makedirs(plots_dir, exist_ok=True)

# 1. Load Data Structure
benchs = obter_arquivos("benchmarks")
ataques = obter_arquivos("ataques")

# Sub-select for training identical limits mapped to Benign structure
random.seed(99)
sh_att = ataques[:]; random.shuffle(sh_att)
sh_bnc = benchs[:]; random.shuffle(sh_bnc)
tr_files = sh_att[:int(len(sh_att)*0.8)] + sh_bnc[:int(len(sh_bnc)*0.8)]

raw_tr = []
for f in tr_files: raw_tr.extend(extract_raw_windows_from_file(f))

# Compute native limits natively exactly as deployed
c_mins, c_maxs, means, stds = compute_benign_bounds(raw_tr)

# Extract testing pool
raw_data = []
all_files = benchs + ataques
for f in all_files: raw_data.extend(extract_raw_windows_from_file(f))

norm_data = apply_safeguard_transforms(raw_data, c_mins, c_maxs, means, stds)
X = [x[0] for x in norm_data]
y_true = [x[1] for x in norm_data]

# 2. Load the locked 20-N deployment limit
parser, predictor = generate_python_inferencer(25, 20)
weights = parser(os.path.join(base_path, "pesos_treinados.h"))

print("Predicting across deployed FANN representations...")
y_probs = predictor(X, weights)

# Threshold strictly established via 99.9th Training Percentile Limit
T = 0.04005 
y_pred = [1.0 if p >= T else 0.0 for p in y_probs]

# 3. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.5f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (20 Neurons / 25 Features)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(plots_dir, "final_roc_curve.png"))
plt.close()

# 4. Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Attack'], 
            yticklabels=['Benign', 'Attack'])
plt.ylabel('True Class Bound')
plt.xlabel('Predicted Assigned Bound')
plt.title(f'Final Clamping Matrix (T={T:.4f})')
plt.savefig(os.path.join(plots_dir, "final_confusion_matrix.png"))
plt.close()

print("Plots successfully rendered via secure topology constraints!")
