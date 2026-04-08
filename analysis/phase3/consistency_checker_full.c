#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "../phase2/pesos_treinados.h"
#include "../phase2/zscore_params.h"

#define WINDOW_SIZE 20
#define NUM_CHANNELS 5
#define NUM_HIDDEN1 16
#define NUM_HIDDEN2 8
#define TOTAL_INPUTS 25

double ring_buf[WINDOW_SIZE][NUM_CHANNELS];
uint32_t current_idx = 0;
uint32_t total_samples = 0;

double run_sum[NUM_CHANNELS] = {0};
double run_sq[NUM_CHANNELS] = {0}; // Use double for sq to minimize drift natively in C for $x^2$

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

static inline void push_min_q(int c, uint32_t idx, double val) {
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

static inline void push_max_q(int c, uint32_t idx, double val) {
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

static inline double baremetal_expf(double x) {
    if (x <= -15.0) return 0.0;
    if (x >= 15.0) return 3269017.0;
    double d = 1.0 + (x / 1024.0);
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d;                 
    return d;
}

static inline double fann_sigmoid_symmetric(double x) {
    return -1.0 + (2.0 / (1.0 + baremetal_expf(-x)));
}
static inline double fann_sigmoid(double x) {
    return 1.0 / (1.0 + baremetal_expf(-x));
}

int main() {
    init_deques();
    double feature_vec[TOTAL_INPUTS];
    double new_vals[NUM_CHANNELS];

    while (scanf("%lf %lf %lf %lf %lf", &new_vals[0], &new_vals[1], &new_vals[2], &new_vals[3], &new_vals[4]) == 5) {
        uint32_t evict_idx = (current_idx + 1) % WINDOW_SIZE;
        double is_full = (total_samples >= WINDOW_SIZE) ? 1.0 : 0.0;
        uint32_t oldest_absolute = (total_samples >= WINDOW_SIZE) ? (total_samples - WINDOW_SIZE) : 0;
        
        for (int c = 0; c < NUM_CHANNELS; c++) {
            double new_v = new_vals[c];
            ring_buf[current_idx][c] = new_v;
            
            pop_min_q(c, oldest_absolute);
            push_min_q(c, total_samples, new_v);
            
            pop_max_q(c, oldest_absolute);
            push_max_q(c, total_samples, new_v);
            
            uint32_t W = (total_samples < WINDOW_SIZE) ? (total_samples + 1) : WINDOW_SIZE;
            
            // Stable O(W) two-pass variance calculation to prevent catastrophic
            // cancellation of huge cumulative PMU cycle floats
            double m_sum = 0.0;
            for(int i=0; i<W; i++) {
                // Iterate oldest to newest to match Python list layout memory associativity
                uint32_t r_idx = (total_samples >= WINDOW_SIZE) ? ((current_idx + 1 + i) % WINDOW_SIZE) : i;
                m_sum += ring_buf[r_idx][c];
            }
            double mean = m_sum / W;
            
            double var_sum = 0.0;
            for(int i=0; i<W; i++) {
                uint32_t r_idx = (total_samples >= WINDOW_SIZE) ? ((current_idx + 1 + i) % WINDOW_SIZE) : i;
                double diff = ring_buf[r_idx][c] - mean;
                var_sum += diff * diff;
            }
            double var = var_sum / W;
            if (var < 0.0) var = 0.0;
            
            feature_vec[c * 5 + 0] = mean;
            feature_vec[c * 5 + 1] = sqrt(var);
            feature_vec[c * 5 + 2] = ring_buf[min_q[c].indices[min_q[c].head % WINDOW_SIZE] % WINDOW_SIZE][c];
            feature_vec[c * 5 + 3] = ring_buf[max_q[c].indices[max_q[c].head % WINDOW_SIZE] % WINDOW_SIZE][c];
            feature_vec[c * 5 + 4] = new_v - ((W == WINDOW_SIZE) ? ring_buf[evict_idx][c] : ring_buf[0][c]);
        }
        
        double norm_feats[TOTAL_INPUTS];
        for (int i = 0; i < TOTAL_INPUTS; i++) {
            double val = feature_vec[i];
            if (val < clamp_mins[i]) val = clamp_mins[i];
            if (val > clamp_maxs[i]) val = clamp_maxs[i];
            norm_feats[i] = (val - feature_means[i]) / feature_stds[i];
        }

        if (total_samples == 0) {
            printf("--- First Sample Debug (C) ---\n");
            printf("Feats: [");
            for(int i=0; i<TOTAL_INPUTS; i++) printf("%.8f ", feature_vec[i]);
            printf("]\n");
            printf("C Normed: [");
            for(int i=0; i<TOTAL_INPUTS; i++) printf("%.8f ", norm_feats[i]);
            printf("]\n");
        }

        double hidden1[NUM_HIDDEN1];
        double hidden2[NUM_HIDDEN2];
        double out;
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
        double prediction = fann_sigmoid(out);
        
        printf("%.12e\n", prediction);

        current_idx = evict_idx;
        total_samples++;
    }

    return 0;
}
