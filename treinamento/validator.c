#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "pesos_treinados.h" 

#define PMU_MAX_CPUS 4
typedef int cpuid_t;

typedef struct {
    uint64_t cpu_cycles;
    uint64_t instructions;
    uint64_t cache_misses;
    uint64_t branch_misses;
    uint64_t timestamp;
} pmu_data_t;

#define NORM_BRANCH      100000000.0f
#define NORM_CACHE       50000000.0f 
#define NORM_INSTR       200000000.0f
#define NORM_CYCLES      400000000.0f
#define NORM_TIMESTAMP   1000000.0f

#define WINDOW_SIZE 5
#define METRICS 5
#define TOTAL_INPUTS (WINDOW_SIZE * METRICS) // 25
#define NUM_HIDDEN1 16
#define NUM_HIDDEN2 8
#define NUM_OUTPUT 1

static float sliding_window_buffer[PMU_MAX_CPUS][WINDOW_SIZE][METRICS];
static int window_index[PMU_MAX_CPUS] = {0};
static int window_filled[PMU_MAX_CPUS] = {0};

static float baremetal_expf(float x) {
    if (x <= -15.0f) return 0.0f;
    if (x >= 15.0f) return 3269017.0f;
    
    float d = 1.0f + (x / 1024.0f);
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d; d *= d; d *= d; 
    d *= d; d *= d;                 
    return d;
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
    
    float delta_ms = 200.0f;
    int idx = window_index[cpu_id];
    
    sliding_window_buffer[cpu_id][idx][0] = (float)pmu->branch_misses / NORM_BRANCH;
    sliding_window_buffer[cpu_id][idx][1] = (float)pmu->cache_misses / NORM_CACHE;
    sliding_window_buffer[cpu_id][idx][2] = (float)pmu->instructions / NORM_INSTR; 
    sliding_window_buffer[cpu_id][idx][3] = (float)pmu->cpu_cycles / NORM_CYCLES;
    sliding_window_buffer[cpu_id][idx][4] = delta_ms / NORM_TIMESTAMP;

    window_index[cpu_id] = (idx + 1) % WINDOW_SIZE;
    if (window_index[cpu_id] == 0) {
        window_filled[cpu_id] = 1;
    }

    if (window_filled[cpu_id]) {
        float flattened_input[TOTAL_INPUTS];
        int flat_idx = 0;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            int circular_idx = (window_index[cpu_id] + i) % WINDOW_SIZE;
            for (int m = 0; m < METRICS; m++) {
                flattened_input[flat_idx++] = sliding_window_buffer[cpu_id][circular_idx][m];
            }
        }

        float is_attack = run_mlp_inference(flattened_input);

        // Strings para imprimir e salvar
        char buffer_inicio[100];
        char buffer_resultado[100];

        sprintf(buffer_inicio, "CPU %d | Predicao NN: %.4f | Real (Label): %.1f | ", cpu_id, is_attack, label_real);

        if (is_attack > 0.80f && label_real == 1.0f) {
            sprintf(buffer_resultado, "[ACERTO] Ataque detectado corretamente!\n");
        } else if (is_attack <= 0.80f && label_real == 0.0f) {
            sprintf(buffer_resultado, "[ACERTO] Silencio mantido. Sistema seguro.\n");
        } else if (is_attack > 0.80f && label_real == 0.0f) {
            sprintf(buffer_resultado, "[ERRO] Falso Positivo!\n");
        } else {
            sprintf(buffer_resultado, "[ERRO] Falso Negativo! Ataque passou despercebido.\n");
        }

        // Imprime na tela
        printf("%s%s", buffer_inicio, buffer_resultado);
        
        // Salva no arquivo de log
        if (log_file != NULL) {
            fprintf(log_file, "%s%s", buffer_inicio, buffer_resultado);
        }
    }
}

int main() {
    FILE *file = fopen("validacao_20_porcento.csv", "r");
    if (!file) {
        printf("Erro ao abrir o arquivo validacao_20_porcento.csv\n");
        return 1;
    }

    // Abre o arquivo de log para escrita
    FILE *log_file = fopen("log_validacao.txt", "w");
    if (!log_file) {
        printf("Erro ao criar o arquivo log_validacao.txt. Os logs serao exibidos apenas na tela.\n");
    } else {
        fprintf(log_file, "--- INICIO DA VALIDACAO DA REDE NEURAL ---\n\n");
    }

    char line[256];
    // Ignora a primeira linha (cabeçalho)
    if (fgets(line, sizeof(line), file) == NULL) {
        printf("Arquivo vazio ou formato incorreto.\n");
        fclose(file);
        if(log_file) fclose(log_file);
        return 1;
    }

    pmu_data_t pmu_mock;
    int core_id;
    float label;

    printf("Iniciando simulacao do EL2 no PC...\n\n");

    // Lê os dados linha por linha: CORE_ID, TIMESTAMP, CYCLES, INSTR, CACHE, BRANCH, LABEL
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%d,%lu,%lu,%lu,%lu,%lu,%f", 
                   &core_id, 
                   &pmu_mock.timestamp, 
                   &pmu_mock.cpu_cycles, 
                   &pmu_mock.instructions, 
                   &pmu_mock.cache_misses, 
                   &pmu_mock.branch_misses, 
                   &label) == 7) {
            
            // Simula a interrupção do timer passando também o ponteiro do arquivo de log
            pc_run_interference_detection(core_id, &pmu_mock, label, log_file);
            
        } else {
            break;
        }
    }

    if (log_file) {
        fprintf(log_file, "\n--- FIM DA VALIDACAO ---\n");
        fclose(log_file);
        printf("\nValidacao concluida. Os resultados foram salvos em 'log_validacao.txt'.\n");
    } else {
        printf("\nValidacao concluida.\n");
    }

    fclose(file);
    return 0;
}
