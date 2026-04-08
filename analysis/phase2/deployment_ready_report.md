# Deployment Ready Evaluation Report
Algorithm: Clamped Z-Score mapped on 25 explicitly Temporal features.

## Topology Size Review
| Nodes | Benign 99.9th Thresh | Validation FPR | Validation Recall |
|-------|----------------------|----------------|-------------------|
| 16 | 0.812323 | 0.000825 | 0.998288 |
| 20 | 0.566019 | 0.000928 | 0.999881 |

**Selected Architecture**: 16 Neurons. Cleanest threshold projection with simplest topology.

## Run Stability Profile
- Threshold ($T$): Mean 0.85165 (Variance: 0.13635)
- Val FPR: Mean 0.00093 (Variance: 0.00028)
- Val Recall: Mean 0.99888 (Variance: 0.00077)

## OOD Protection Limit Test
Tested `9695` extreme artificial combinations bounded to clamped vectors.
- Resulting FPR: **0.000000** (0 False Positives / 9695 True Benign).
- Interpretation: Clamping completely flattened the magnitude explosion, returning the OOD FPR from 1.0 to safe geometries.

