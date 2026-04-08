#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <x86intrin.h>

#include "../phase2/pesos_treinados.h"
#include "../phase2/zscore_params.h"

#define WINDOW_SIZE 20
#define NUM_CHANNELS 5
#define NUM_HIDDEN1 16
#define NUM_HIDDEN2 8
#define TOTAL_INPUTS 25
#define SAMPLING_BUDGET_US 1000.0 // Assuming 1kHz hypervisor tick budget

// --- AGGREGATOR STRUCTURES ---
float ring_buf[WINDOW_SIZE][NUM_CHANNELS];
uint32_t current_idx = 0;
uint32_t total_samples = 0;

float run_sum[NUM_CHANNELS] = {0};
float run_sq[NUM_CHANNELS] = {0};

typedef struct {
    uint32_t indices[WINDOW_SIZE];
    int head;
    int tail;
} MonotoneDeque;

MonotoneDeque min_q[NUM_CHANNELS];
MonotoneDeque max_q[NUM_CHANNELS];

void init_deques() {
    for (int c = 0; c < NUM_CHANNELS; c++) {
        min_q[c].head = 0; min_q[c].tail = 0;
        max_q[c].head = 0; max_q[c].tail = 0;
    }
}

static inline void push_min_q(int c, uint32_t idx, float val) {
    while (min_q[c].tail > min_q[c].head && ring_buf[min_q[c].indices[(min_q[c].tail - 1) % WINDOW_SIZE] % WINDOW_SIZE][c] >= val) {
        min_q[c].tail--;
    }
    min_q[c].indices[min_q[c].tail % WINDOW_SIZE] = idx;
    min_q[c].tail++;
}

static inline void pop_min_q(int c, uint32_t oldest_idx) {
    if (min_q[c].tail > min_q[c].head && min_q[c].indices[min_q[c].head % WINDOW_SIZE] == oldest_idx) {
        min_q[c].head++;
    }
}

static inline void push_max_q(int c, uint32_t idx, float val) {
    while (max_q[c].tail > max_q[c].head && ring_buf[max_q[c].indices[(max_q[c].tail - 1) % WINDOW_SIZE] % WINDOW_SIZE][c] <= val) {
        max_q[c].tail--;
    }
    max_q[c].indices[max_q[c].tail % WINDOW_SIZE] = idx;
    max_q[c].tail++;
}

static inline void pop_max_q(int c, uint32_t oldest_idx) {
    if (max_q[c].tail > max_q[c].head && max_q[c].indices[max_q[c].head % WINDOW_SIZE] == oldest_idx) {
        max_q[c].head++;
    }
}

// --- FANN MATH MATH ---
static inline float baremetal_expf(float x) {
    if (x <= -15.0f) return 0.0f;
    if (x >= 15.0f) return 3269017.0f;
    float d = 1.0f + (x / 1024.0f);
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d;                 
    return d;
}

static inline float fann_sigmoid_symmetric(float x) {
    return -1.0f + (2.0f / (1.0f + baremetal_expf(-x)));
}
static inline float fann_sigmoid(float x) {
    return 1.0f / (1.0f + baremetal_expf(-x));
}

int main() {
    init_deques();
    
    // Create random mock stream matching hypervisor environment reality (e.g. 50k samples)
    uint32_t test_size = 50000;
    
    uint64_t agg_cycles_total = 0, agg_cycles_max = 0;
    uint64_t infer_cycles_total = 0, infer_cycles_max = 0;
    
    // Variance calculation for cycles (Welford's)
    double infer_cycle_M2 = 0.0, infer_cycle_mean = 0.0;
    double agg_cycle_M2 = 0.0, agg_cycle_mean = 0.0;
    
    struct timespec start_ts, end_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    
    float feature_vec[TOTAL_INPUTS];

    for (uint32_t t = 0; t < test_size; t++) {
        // Mock new PMU readings
        float new_vals[NUM_CHANNELS];
        for(int c=0; c<NUM_CHANNELS; c++) new_vals[c] = (float)(rand() % 100);
        
        // --------------------------------------------------------------------------------
        // 1. O(1) AGGREGATION BLOCK
        // --------------------------------------------------------------------------------
        uint64_t start_agg = __rdtsc();
        
        uint32_t evict_idx = (current_idx + 1) % WINDOW_SIZE;
        float is_full = (total_samples >= WINDOW_SIZE) ? 1.0f : 0.0f;
        uint32_t oldest_absolute = (total_samples >= WINDOW_SIZE) ? (total_samples - WINDOW_SIZE) : 0;
        
        for (int c = 0; c < NUM_CHANNELS; c++) {
            float old_val = ring_buf[evict_idx][c] * is_full;
            float new_v = new_vals[c];
            
            run_sum[c] += new_v - old_val;
            run_sq[c] += (new_v * new_v) - (old_val * old_val);
            
            ring_buf[current_idx][c] = new_v;
            
            pop_min_q(c, oldest_absolute);
            push_min_q(c, total_samples, new_v);
            
            pop_max_q(c, oldest_absolute);
            push_max_q(c, total_samples, new_v);
            
            // Build the exact 25-feature vector mapping
            uint32_t W = (total_samples < WINDOW_SIZE) ? (total_samples + 1) : WINDOW_SIZE;
            
            float mean = run_sum[c] / W;
            float var = (run_sq[c] / W) - (mean * mean);
            if (var < 0.0f) var = 0.0f; // float safety
            
            feature_vec[c * 5 + 0] = mean;
            feature_vec[c * 5 + 1] = sqrtf(var);
            feature_vec[c * 5 + 2] = ring_buf[min_q[c].indices[min_q[c].head % WINDOW_SIZE] % WINDOW_SIZE][c];
            feature_vec[c * 5 + 3] = ring_buf[max_q[c].indices[max_q[c].head % WINDOW_SIZE] % WINDOW_SIZE][c];
            // Delta: Current - Oldest
            feature_vec[c * 5 + 4] = new_v - ((W == WINDOW_SIZE) ? ring_buf[evict_idx][c] : ring_buf[0][c]);
        }
        
        uint64_t end_agg = __rdtsc();
        uint64_t agg_cost = end_agg - start_agg;
        agg_cycles_total += agg_cost;
        if (agg_cost > agg_cycles_max) agg_cycles_max = agg_cost;
        
        double agg_delta = agg_cost - agg_cycle_mean;
        agg_cycle_mean += agg_delta / (t + 1);
        agg_cycle_M2 += agg_delta * (agg_cost - agg_cycle_mean);
        
        // --------------------------------------------------------------------------------
        // 2. FANN INFERENCE BLOCK (Same variables exactly as C consistency script)
        // --------------------------------------------------------------------------------
        uint64_t start_infer = __rdtsc();
        
        float norm_feats[TOTAL_INPUTS];
        for (int i = 0; i < TOTAL_INPUTS; i++) {
            float val = feature_vec[i];
            if (val < clamp_mins[i]) val = clamp_mins[i];
            if (val > clamp_maxs[i]) val = clamp_maxs[i];
            norm_feats[i] = (val - feature_means[i]) / feature_stds[i];
        }

        float hidden1[NUM_HIDDEN1];
        float hidden2[NUM_HIDDEN2];
        float out;
        int p = 0;

        for (int i = 0; i < NUM_HIDDEN1; i++) {
            hidden1[i] = pesos_iniciais[p++]; 
            for (int j = 0; j < TOTAL_INPUTS; j++) hidden1[i] += pesos_iniciais[p++] * norm_feats[j];
            hidden1[i] = fann_sigmoid_symmetric(hidden1[i]);
        }

        for (int i = 0; i < NUM_HIDDEN2; i++) {
            hidden2[i] = pesos_iniciais[p++]; 
            for (int j = 0; j < NUM_HIDDEN1; j++) hidden2[i] += pesos_iniciais[p++] * hidden1[j];
            hidden2[i] = fann_sigmoid_symmetric(hidden2[i]);
        }

        out = pesos_iniciais[p++]; 
        for (int j = 0; j < NUM_HIDDEN2; j++) out += pesos_iniciais[p++] * hidden2[j];
        float prediction = fann_sigmoid(out);
        
        volatile float force_keep = prediction; // Prevent aggressive -O3 destruction
        
        uint64_t end_infer = __rdtsc();
        uint64_t infer_cost = end_infer - start_infer;
        infer_cycles_total += infer_cost;
        if (infer_cost > infer_cycles_max) infer_cycles_max = infer_cost;
        
        double infer_delta = infer_cost - infer_cycle_mean;
        infer_cycle_mean += infer_delta / (t + 1);
        infer_cycle_M2 += infer_delta * (infer_cost - infer_cycle_mean);

        // Update tracking indices
        current_idx = evict_idx;
        total_samples++;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    double elapsed_us = (end_ts.tv_sec - start_ts.tv_sec) * 1e6 + (end_ts.tv_nsec - start_ts.tv_nsec) / 1e3;
    
    double agg_cycle_variance = agg_cycle_M2 / test_size;
    double infer_cycle_variance = infer_cycle_M2 / test_size;

    double agg_overhead_us = (double)(agg_cycles_total / test_size) / 1500.0; // Approximation if 1.5 GHz CPU
    double infer_overhead_us = (double)(infer_cycles_total / test_size) / 1500.0;

    printf("STAT,AVG_CYCLES,MAX_CYCLES,VARIANCE,OVERHEAD_US,BUDGET_PERCENT\n");
    printf("AGGREGATION,%llu,%llu,%f,%f,%f%%\n", 
           (unsigned long long)(agg_cycles_total / test_size), 
           (unsigned long long)agg_cycles_max, 
           agg_cycle_variance, 
           agg_overhead_us,
           (agg_overhead_us / SAMPLING_BUDGET_US) * 100.0);
           
    printf("INFERENCE,%llu,%llu,%f,%f,%f%%\n", 
           (unsigned long long)(infer_cycles_total / test_size), 
           (unsigned long long)infer_cycles_max, 
           infer_cycle_variance, 
           infer_overhead_us,
           (infer_overhead_us / SAMPLING_BUDGET_US) * 100.0);
           
    printf("TOTAL,%llu,%llu,%f,%f,%f%%\n", 
           (unsigned long long)((agg_cycles_total + infer_cycles_total) / test_size), 
           (unsigned long long)(agg_cycles_max + infer_cycles_max), 
           agg_cycle_variance + infer_cycle_variance, 
           elapsed_us / test_size,
           ((elapsed_us / test_size) / SAMPLING_BUDGET_US) * 100.0);

    return 0;
}
