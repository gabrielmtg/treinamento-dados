# PMU Microarchitectural Cross-Core Interference Detection Pipeline

This repository hosts the full research and deployment pipeline for a Microarchitectural Cross-Core Attack Detection system leveraging Performance Monitor Unit (PMU) telemetry. The end goal of this repository is to process raw PMU samples into an ultra-low latency, O(1) memory bare-metal Neural Network inference module optimized directly for the Bao Hypervisor.

The workflow is broken into three chronological execution phases.

## Directory Structure
- `data/`: Place raw CSVs here. The full pipeline typically expects `data_final_clean.csv`.
- `analysis/phase1/`: Data preparation and Feature Characterization.
- `analysis/phase2/`: Model Hardening, OOD Clamping, and Weights Export.
- `analysis/phase3/`: Deployment Readiness, Latency Profiling, and Mathematical Parity Validation.

---

## Phase 1: Feature Characterization & Distribution Analysis
**Location:** `analysis/phase1/`

The goal of Phase 1 is to profile the underlying dataset and statistically extract temporal distributions from the PMU channels (e.g. CPU Cycles, Cache Misses, Branch Misses, Instructions).
* **Script to run**: `python workload_characterization.py`
* **Output**: Generates initial feature distribution plots and baseline statistics.

---

## Phase 2: Secure Model Training & Hardening
**Location:** `analysis/phase2/`

In this phase, we build a topology-optimized Multi-Layer Perceptron (MLP) robust to distribution shifts.

1. **Clean Dataset Setup**: Ensure `data_final_clean.csv` exists in `data/`.
2. **Execute Deployment Harness**:
   Run `python deployment_harness.py`.
   
   **What this does:**
   - Evaluates node topologies (e.g. 16 vs 20 hidden nodes) for optimal bare-metal footprint.
   - Extracts Benign-Only 99.9th percentile Out-Of-Distribution (OOD) clamping vectors to prevent false positives under extreme hypervisor load.
   - Generates a bespoke C-trainer (`train_80.c`) which utilizes the FANN library natively.
   - Dumps `pesos_treinados.h` (Neural Network weight payload) and `zscore_params.h` (Normalization limits/thresholds) for the bare-metal port.
   - Compiles validation metrics into `results/plots`.

---

## Phase 3: Online Deployment Readiness & Validation
**Location:** `analysis/phase3/`

Phase 3 validates that the mathematically hardened model from Phase 2 can operate within the real-time, zero-OS standard library constraints of the Bao hypervisor. 

1. **Evaluate Latency Trades (`generate_phase3_reports.py`)**:
   - Run `python generate_phase3_reports.py` to simulate sliding window memory streams.
   - Measures detection latency ($\Delta d$) across sliding window choices (e.g., $W=3, 5, 10, 20$).
   - Outputs tradeoff graphics into `results/phase3/`.

2. **Benchmark Bare-Metal Runtime (`benchmark.c`)**:
   - Compile: `gcc benchmark.c -o benchmark_bin -lm`
   - Execute: `./benchmark_bin`
   - **What this does:** Operates a simulated continuous PMU stream through the specialized C-based $O(1)$ Incremental Ring Buffer. Benchmarks cycle counts and time overhead for (1) running variance accumulation and (2) Custom sigmoid inference. Expect overhead $< 0.5\mu s$.

3. **Validate Mathematical Parity**:
   - Before dropping code into Bao, we must guarantee the C compiler and Python's Numpy simulator evaluate exactly. 
   - Compile the pipeline checker: `gcc consistency_checker_full.c -o consistency_checker_full -lm`
   - Run evaluation: `python validate_consistency.py`
   - **What this does:** Pipes a 5000 row sequence through both the native Numpy predictor and the compiled Bare-Metal binary. It forcefully ensures numerical drift is $< 1e-5$, validating stable numerical limits on hardware.
