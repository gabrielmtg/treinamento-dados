#include <stdio.h>
#include <stdlib.h>
#include "pesos_treinados.h"
#include "zscore_params.h"

#define TOTAL_INPUTS 25
#define NUM_HIDDEN1 16
#define NUM_HIDDEN2 8

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

int main(int argc, char *argv[]) {
    if (argc != TOTAL_INPUTS + 1) {
        printf("Usage: %s <25 floats>\n", argv[0]);
        return 1;
    }

    float raw_feats[TOTAL_INPUTS];
    for (int i = 0; i < TOTAL_INPUTS; i++) {
        raw_feats[i] = atof(argv[i+1]);
    }

    float norm_feats[TOTAL_INPUTS];
    for (int i = 0; i < TOTAL_INPUTS; i++) {
        float val = raw_feats[i];
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
        for (int j = 0; j < TOTAL_INPUTS; j++) {
            hidden1[i] += pesos_iniciais[p++] * norm_feats[j];
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
    
    printf("%.8f\n", fann_sigmoid(out));
    return 0;
}
