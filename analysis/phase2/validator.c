#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "pesos_treinados.h" 
#include "zscore_params.h" 

#define PMU_MAX_CPUS 4
typedef int cpuid_t;

typedef struct {
    uint64_t cpu_cycles;
    uint64_t instructions;
    uint64_t cache_misses;
    uint64_t branch_misses;
    uint64_t timestamp;
} pmu_data_t;

#define WINDOW_SIZE 5
#define METRICS 5
#define TOTAL_INPUTS 25
#define NUM_HIDDEN1 20
#define NUM_HIDDEN2 8
#define NUM_OUTPUT 1

static float sliding_window_buffer[PMU_MAX_CPUS][WINDOW_SIZE][METRICS];
static int window_index[PMU_MAX_CPUS] = {0};
static int window_filled[PMU_MAX_CPUS] = {0};
static uint64_t prev_timestamp[PMU_MAX_CPUS] = {0};

/* NATIVE OS-LESS MATH REPLACEMENTS */
static float baremetal_expf(float x) {
    if (x <= -15.0f) return 0.0f;
    if (x >= 15.0f) return 3269017.0f;
    
    float d = 1.0f + (x / 1024.0f);
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d;                 
    return d;
}

static float baremetal_sqrtf(float x) {
    if (x <= 0.0000001f) return 0.0f;
    float result = x;
    for (int i = 0; i < 10; i++) {
        result = 0.5f * (result + (x / result));
    }
    return result;
}

static float fann_sigmoid_symmetric(float x) {
    return -1.0f + (2.0f / (1.0f + baremetal_expf(-x)));
}

static float fann_sigmoid(float x) {
    return 1.0f / (1.0f + baremetal_expf(-x));
}

static float run_mlp_inference(float *inputs) {
    float hidden1[NUM_HIDDEN1];
    float hidden2[NUM_HIDDEN2];
    float out;
    int p = 0;

    for (int i = 0; i < NUM_HIDDEN1; i++) {
        hidden1[i] = pesos_iniciais[p++]; 
        for (int j = 0; j < TOTAL_INPUTS; j++) {
            hidden1[i] += pesos_iniciais[p++] * inputs[j];
        }
        hidden1[i] = fann_sigmoid_symmetric(hidden1[i]);
    }

    for (int i = 0; i < NUM_HIDDEN2; i++) {
        hidden2[i] = pesos_iniciais[p++]; 
        for (int j = 0; j < NUM_HIDDEN1; j++) {
            hidden2[i] += pesos_iniciais[p++] * hidden1[j];
        }
        hidden2[i] = fann_sigmoid_symmetric(hidden2[i]);
    }

    out = pesos_iniciais[p++]; 
    for (int j = 0; j < NUM_HIDDEN2; j++) {
        out += pesos_iniciais[p++] * hidden2[j];
    }
    return fann_sigmoid(out);
}

void pc_run_interference_detection(cpuid_t cpu_id, pmu_data_t *pmu, float label_real, FILE *log_file) {
    
    // Simulate Time Delta natively
    float delta_ms = 0.0f;
    if (prev_timestamp[cpu_id] > 0) {
        delta_ms = (float)(pmu->timestamp - prev_timestamp[cpu_id]);
    }
    prev_timestamp[cpu_id] = pmu->timestamp;

    int idx = window_index[cpu_id];
    
    // Exact mapping order expected by Python Feature Engineering: 
    // Branch, Cache, Instr, Cycles, Delta
    sliding_window_buffer[cpu_id][idx][0] = (float)pmu->branch_misses;
    sliding_window_buffer[cpu_id][idx][1] = (float)pmu->cache_misses;
    sliding_window_buffer[cpu_id][idx][2] = (float)pmu->instructions; 
    sliding_window_buffer[cpu_id][idx][3] = (float)pmu->cpu_cycles;
    sliding_window_buffer[cpu_id][idx][4] = delta_ms;

    window_index[cpu_id] = (idx + 1) % WINDOW_SIZE;
    if (window_index[cpu_id] == 0) {
        window_filled[cpu_id] = 1;
    }

    if (window_filled[cpu_id]) {
        float features[TOTAL_INPUTS]; // 25
        int f_idx = 0;

        // Iterate identically across channels
        for (int m = 0; m < METRICS; m++) {
            float min_val = sliding_window_buffer[cpu_id][0][m];
            float max_val = min_val;
            float sum = 0.0f;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                float v = sliding_window_buffer[cpu_id][i][m];
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
                sum += v;
            }
            float mean_val = sum / (float)WINDOW_SIZE;
            
            float var_sum = 0.0f;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                float diff = sliding_window_buffer[cpu_id][i][m] - mean_val;
                var_sum += diff * diff;
            }
            float std_val = baremetal_sqrtf(var_sum / (float)WINDOW_SIZE);
            if (std_val == 0.0f) std_val = 1e-9f; // Native python floor replication

            int first_idx = window_index[cpu_id];
            int last_idx = (window_index[cpu_id] + WINDOW_SIZE - 1) % WINDOW_SIZE;
            float delta_val = sliding_window_buffer[cpu_id][last_idx][m] - sliding_window_buffer[cpu_id][first_idx][m];

            features[f_idx++] = mean_val;
            features[f_idx++] = std_val;
            features[f_idx++] = min_val;
            features[f_idx++] = max_val;
            features[f_idx++] = delta_val;
        }

        // Apply Native Clamping & Normalization securely
        for (int i = 0; i < TOTAL_INPUTS; i++) {
            float val = features[i];
            if (val < clamp_mins[i]) val = clamp_mins[i];
            if (val > clamp_maxs[i]) val = clamp_maxs[i];
            features[i] = (val - feature_means[i]) / feature_stds[i];
        }

        float is_attack = run_mlp_inference(features);

        char buffer_resultado[100];
        if (is_attack >= DECISION_THRESHOLD && label_real == 1.0f) {
            sprintf(buffer_resultado, "[TRUE-POSITIVE] Securing Hypervisor! Ataque bloqueado.\n");
        } else if (is_attack < DECISION_THRESHOLD && label_real == 0.0f) {
            sprintf(buffer_resultado, "[TRUE-NEGATIVE] Carga segura validada.\n");
        } else if (is_attack >= DECISION_THRESHOLD && label_real == 0.0f) {
            sprintf(buffer_resultado, "[FALSE-POSITIVE] Alarme falso!\n");
        } else {
            sprintf(buffer_resultado, "[FALSE-NEGATIVE] Quebra de barreira! Ataque passou.\n");
        }

        if (log_file != NULL) {
            fprintf(log_file, "Pred(%.4f) T(%.4f) | %s", is_attack, DECISION_THRESHOLD, buffer_resultado);
        }
    }
}

// ----------------------------------------------------
// Mock Desktop Execution Wrapper for easy logical testing
// ----------------------------------------------------
uint64_t parse_time(const char* time_str) {
    int h, m, s, ms;
    if (sscanf(time_str, "%d:%d:%d.%d", &h, &m, &s, &ms) == 4) {
        return (h * 3600 + m * 60 + s) * 1000 + ms;
    }
    return 0;
}

int main() {
    FILE *file = fopen("../ataques/dados_spectre.txt", "r");
    if (!file) {
        printf("Arquivo de simulacao ausente!\n");
        return 1;
    }

    FILE *log_file = fopen("log_validacao_baremetal.txt", "w");
    if (log_file) {
        fprintf(log_file, "--- BAREMETAL INFERENCE MODULE STANDBY ---\n\n");
    }

    char line[256];
    pmu_data_t pmu_mock;
    int core_id = 0;
    float label;

    printf("Booting Native Hypervisor PMU Interceptor...\n\n");

    int test_samples = 0;
    while (fgets(line, sizeof(line), file) && test_samples < 500) {
        char tempo_str[50];
        if (sscanf(line, "%s %lu %lu %lu %lu %f", 
                   tempo_str, 
                   &pmu_mock.cpu_cycles, 
                   &pmu_mock.instructions, 
                   &pmu_mock.cache_misses, 
                   &pmu_mock.branch_misses, 
                   &label) == 6) {
            
            pmu_mock.timestamp = parse_time(tempo_str);
            pc_run_interference_detection(core_id, &pmu_mock, label, log_file);
            test_samples++;
        }
    }

    if (log_file) {
        fprintf(log_file, "\n--- SHUTDOWN ---\n");
        fclose(log_file);
        printf("Simulacao executada! Verifique 'log_validacao_baremetal.txt'.\n");
    }

    fclose(file);
    return 0;
}
