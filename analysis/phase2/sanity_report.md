# PMU Overfitting & Leakage Sanity Report

### Imbalance Report
- Benign Windows: 53177
- Attack Windows: 93386
- Status: BALANCED

### Leave-One-File-Out Cross Validation (LOOCV)
- Testing on dados_spectre.txt:
  Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:21016, FP:0, FN:0, TN:0)
- Testing on dados_meltdown2.txt:
  Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:4901, FP:0, FN:0, TN:0)
- Testing on dados_zombieload.txt:
  Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:24791, FP:0, FN:0, TN:0)
- Testing on dados_meltdown.txt:
  Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:7531, FP:0, FN:0, TN:0)
- Testing on dados_meltdown3.txt:
  Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:12515, FP:0, FN:0, TN:0)
- Testing on dados_armageddon.txt:
  Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:22632, FP:0, FN:0, TN:0)
- Testing on dados_disparity.txt:
  Acc=1.0000 | Prec=0.0000 | Rec=0.0000 | F1=0.0000 | FPR=0.0000 (TP:0, FP:0, FN:0, TN:242)
- Testing on dados_disparity2.txt:
  Acc=0.9998 | Prec=0.0000 | Rec=0.0000 | F1=0.0000 | FPR=0.0002 (TP:0, FP:5, FN:0, TN:20744)
- Testing on dados_bandwidth.txt:
  Acc=0.0000 | Prec=0.0000 | Rec=0.0000 | F1=0.0000 | FPR=1.0000 (TP:0, FP:32186, FN:0, TN:0)
**Average LOOCV Accuracy: 0.8889 | Average FPR: 0.1111**

### Class Imbalance Mitigation (Undersampling)
### Leakage & Distribution Check
- Train features global mean: 0.176092
- Valid features global mean: 0.141008
- Variance between sets: 0.035084
- Conclusion: Distributions indicate clean separation.

Results on standard 80/20 grouped split:
- Without Imbalance Fix: Acc=1.0000 | Prec=1.0000 | Rec=1.0000 | F1=1.0000 | FPR=0.0000 (TP:43648, FP:0, FN:0, TN:0)
- With Undersampling:    Acc=0.5185 | Prec=1.0000 | Rec=0.5185 | F1=0.6829 | FPR=0.0000 (TP:22632, FP:0, FN:21016, TN:0)

### Label Shuffle Test (Checks fundamental architecture overfit)
- Metrics trained on shuffled labels: Acc=0.0000 | Prec=0.0000 | Rec=0.0000 | F1=0.0000 | FPR=0.0000 (TP:0, FP:0, FN:43648, TN:0)
- Conclusion: If Accuracy is near 0.50, the model is honestly learning features, not trivially memorizing data layout.
