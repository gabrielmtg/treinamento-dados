# Deployment Ready Evaluation Report
Algorithm: Clamped Z-Score mapped on 25 explicitly Temporal features.

## Topology Size Review
| Nodes | Benign 99.9th Thresh | Validation FPR | Validation Recall |
|-------|----------------------|----------------|-------------------|
| 16 | 0.050721 | 0.000999 | 0.999822 |
| 20 | 0.896898 | 0.001212 | 0.999679 |

**Selected Architecture**: 16 Neurons. Cleanest threshold projection with simplest topology.

## Run Stability Profile
- Threshold ($T$): Mean 0.33747 (Variance: 0.33783)
- Val FPR: Mean 0.00092 (Variance: 0.00006)
- Val Recall: Mean 0.99984 (Variance: 0.00004)

## OOD Protection Limit Test
Tested `10000` extreme artificial combinations bounded to clamped vectors.
- Resulting FPR: **0.000300** (3 False Positives / 9997 True Benign).
- Interpretation: Clamping completely flattened the magnitude explosion, returning the OOD FPR from 1.0 to safe geometries.

