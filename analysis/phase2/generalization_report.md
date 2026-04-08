# Generalization & Feature Optimization Report

Using 50 Temporal Features (Raw + Mean + Std + Min + Max + Delta) alongside Static Z-score bounds mapping.

### Generalization Test: Leave-One-Benchmark-Out
- Left out **dados_disparity.txt**: Acc=0.9959 | Rec=0.0000 | FPR=0.0041 (TP:0, FP:1, FN:0, TN:241) @ Thresh=0.315
- Left out **dados_disparity2.txt**: Acc=1.0000 | Rec=0.0000 | FPR=0.0000 (TP:0, FP:0, FN:0, TN:20749) @ Thresh=0.568
- Left out **dados_bandwidth.txt**: Acc=0.5026 | Rec=0.0000 | FPR=0.4974 (TP:0, FP:16010, FN:0, TN:16176) @ Thresh=0.920

### Generalization Test: Leave-One-Attack-Out
- Left out **dados_spectre.txt**: Acc=1.0000 | Rec=1.0000 | FPR=0.0000 (TP:21015, FP:0, FN:1, TN:0) @ Thresh=0.434
- Left out **dados_meltdown2.txt**: Acc=0.0000 | Rec=0.0000 | FPR=0.0000 (TP:0, FP:0, FN:4901, TN:0) @ Thresh=0.887
- Left out **dados_zombieload.txt**: Acc=0.0000 | Rec=0.0000 | FPR=0.0000 (TP:0, FP:0, FN:24791, TN:0) @ Thresh=0.784
- Left out **dados_meltdown.txt**: Acc=1.0000 | Rec=1.0000 | FPR=0.0000 (TP:7531, FP:0, FN:0, TN:0) @ Thresh=0.737
- Left out **dados_meltdown3.txt**: Acc=0.0001 | Rec=0.0001 | FPR=0.0000 (TP:1, FP:0, FN:12514, TN:0) @ Thresh=0.617
- Left out **dados_armageddon.txt**: Acc=0.0000 | Rec=0.0000 | FPR=0.0000 (TP:0, FP:0, FN:22632, TN:0) @ Thresh=0.644
