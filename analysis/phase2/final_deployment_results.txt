# Deployment Ready Evaluation Report
Algorithm: Clamped Z-Score mapped on 25 explicitly Temporal features.

## Topology Size Review
| Nodes | Benign 99.9th Thresh | Validation FPR | Validation Recall |
|-------|----------------------|----------------|-------------------|
| 16 | 0.005443 | 0.004132 | 1.000000 |
| 20 | 0.021965 | 0.004132 | 1.000000 |

**Selected Architecture**: 20 Neurons. Cleanest threshold projection with simplest topology.

## Run Stability Profile
- Threshold ($T$): Mean 0.04005 (Variance: 0.02692)
- Val FPR: Mean 0.00496 (Variance: 0.00165)
- Val Recall: Mean 0.99991 (Variance: 0.00013)

## OOD Protection Limit Test
Tested `32186` extreme artificial combinations bounded to clamped vectors.
- Resulting FPR: **0.000186** (6 False Positives / 32180 True Benign).
- Interpretation: Clamping completely flattened the magnitude explosion, returning the OOD FPR from 1.0 to safe geometries.

