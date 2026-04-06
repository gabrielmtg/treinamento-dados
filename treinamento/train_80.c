#include <stdio.h>
#include <stdlib.h>
#include "floatfann.h"

float get_weight(struct fann_connection *conexoes, unsigned int total, unsigned int from, unsigned int to) {
    for(unsigned int i = 0; i < total; i++) {
        if(conexoes[i].from_neuron == from && conexoes[i].to_neuron == to) return conexoes[i].weight;
    }
    return 0.0f; 
}

int main() {
    const unsigned int num_input = 25; 
    const unsigned int num_output = 1;
    const unsigned int num_layers = 4;
    const unsigned int num_neurons_hidden1 = 20;
    const unsigned int num_neurons_hidden2 = 8;
    const float desired_error = 0.001f;
    const unsigned int max_epochs = 1000;
    const unsigned int epochs_between_reports = 1000;

    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden1, num_neurons_hidden2, num_output);
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);
    
    fann_train_on_file(ann, "dados_treino_80.train", max_epochs, epochs_between_reports, desired_error);
    fann_save(ann, "modelo_pmu_timestamp.net");

    unsigned int tc = fann_get_total_connections(ann);
    struct fann_connection *connections = malloc(sizeof(struct fann_connection) * tc);
    fann_get_connection_array(ann, connections);

    FILE *f = fopen("pesos_treinados.h", "w");
    fprintf(f, "#ifndef PESOS_TREINADOS_H\n#define PESOS_TREINADOS_H\n\n");
    fprintf(f, "#define NUM_PESOS %u\n\n", tc);
    fprintf(f, "const float pesos_iniciais[NUM_PESOS] = {\n");

    int c = 0;
    for (int i = 0; i < 20; i++) {
        int to_node = 26 + i;
        fprintf(f, "    %ff, // Bias -> H1[%d]\n", get_weight(connections, tc, 25, to_node), i);
        c++;
        for (int j = 0; j < 25; j++) {
            fprintf(f, "    %ff, // Input[%d] -> H1[%d]\n", get_weight(connections, tc, j, to_node), j, i);
            c++;
        }
    }

    for (int i = 0; i < 8; i++) {
        int to_node = 47 + i;
        fprintf(f, "    %ff, // Bias -> H2[%d]\n", get_weight(connections, tc, 46, to_node), i);
        c++;
        for (int j = 0; j < 20; j++) {
            fprintf(f, "    %ff, // H1[%d] -> H2[%d]\n", get_weight(connections, tc, 26 + j, to_node), j, i);
            c++;
        }
    }

    for (int i = 0; i < 1; i++) {
        int to_node = 56 + i;
        fprintf(f, "    %ff, // Bias -> Out[%d]\n", get_weight(connections, tc, 55, to_node), i);
        c++;
        for (int j = 0; j < 8; j++) {
            if (c == tc) fprintf(f, "    %ff  // H2[%d] -> Out[%d]\n", get_weight(connections, tc, 47 + j, to_node), j, i);
            else fprintf(f, "    %ff, // H2[%d] -> Out[%d]\n", get_weight(connections, tc, 47 + j, to_node), j, i);
            c++;
        }
    }

    fprintf(f, "};\n\n#endif\n");
    fclose(f);
    free(connections);
    fann_destroy(ann);
    return 0;
}
