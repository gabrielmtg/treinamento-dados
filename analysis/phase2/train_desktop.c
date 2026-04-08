#include <stdio.h>
#include <stdlib.h>
#include "floatfann.h"

// Função auxiliar para buscar o peso de uma conexão específica (origem -> destino)
float get_weight(struct fann_connection *conexoes, unsigned int total, unsigned int from, unsigned int to) {
    for(unsigned int i = 0; i < total; i++) {
        if(conexoes[i].from_neuron == from && conexoes[i].to_neuron == to) {
            return conexoes[i].weight;
        }
    }
    return 0.0f; 
}

int main() {
    const unsigned int num_input = 25; // 5 métricas * WINDOW_SIZE 5
    const unsigned int num_output = 1;
    const unsigned int num_layers = 4;
    const unsigned int num_neurons_hidden1 = 16;
    const unsigned int num_neurons_hidden2 = 8;
    const float desired_error = 0.001f;
    const unsigned int max_epochs = 500000;
    const unsigned int epochs_between_reports = 1000;

    // 1. Cria a rede com a topologia desejada
    struct fann *ann = fann_create_standard(num_layers, num_input, 
                                            num_neurons_hidden1, 
                                            num_neurons_hidden2, 
                                            num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);

    // 2. Treina a rede usando um arquivo .train
    printf("Iniciando treinamento...\n");

    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);
    
    fann_train_on_file(ann, "dados_projeto.train", max_epochs, 
                       epochs_between_reports, desired_error);

    fann_save(ann, "modelo_pmu_timestamp.net");

    unsigned int total_connections = fann_get_total_connections(ann);
    struct fann_connection *connections = malloc(sizeof(struct fann_connection) * total_connections);
    fann_get_connection_array(ann, connections);

    FILE *f = fopen("pesos_treinados.h", "w");
    fprintf(f, "#ifndef PESOS_TREINADOS_H\n#define PESOS_TREINADOS_H\n\n");
    fprintf(f, "#define NUM_PESOS %u\n\n", total_connections);
    fprintf(f, "const float pesos_iniciais[NUM_PESOS] = {\n");

    int count = 0;

    for (int i = 0; i < 16; i++) {
        int to_node = 26 + i;
        // Bias primeiro
        fprintf(f, "    %ff, // Bias -> H1[%d]\n", get_weight(connections, total_connections, 25, to_node), i);
        count++;
        // Depois os pesos das entradas
        for (int j = 0; j < 25; j++) {
            int from_node = 0 + j;
            fprintf(f, "    %ff, // Input[%d] -> H1[%d]\n", get_weight(connections, total_connections, from_node, to_node), j, i);
            count++;
        }
    }

    for (int i = 0; i < 8; i++) {
        int to_node = 43 + i;
        // Bias primeiro
        fprintf(f, "    %ff, // Bias -> H2[%d]\n", get_weight(connections, total_connections, 42, to_node), i);
        count++;
        // Depois os pesos da H1
        for (int j = 0; j < 16; j++) {
            int from_node = 26 + j;
            fprintf(f, "    %ff, // H1[%d] -> H2[%d]\n", get_weight(connections, total_connections, from_node, to_node), j, i);
            count++;
        }
    }

    for (int i = 0; i < 1; i++) {
        int to_node = 52 + i;
        // Bias primeiro
        fprintf(f, "    %ff, // Bias -> Out[%d]\n", get_weight(connections, total_connections, 51, to_node), i);
        count++;
        // Depois os pesos da H2
        for (int j = 0; j < 8; j++) {
            int from_node = 43 + j;
            
            // Controle para não colocar vírgula no último item
            if (count == total_connections) {
                fprintf(f, "    %ff  // H2[%d] -> Out[%d]\n", get_weight(connections, total_connections, from_node, to_node), j, i);
            } else {
                fprintf(f, "    %ff, // H2[%d] -> Out[%d]\n", get_weight(connections, total_connections, from_node, to_node), j, i);
            }
            count++;
        }
    }

    fprintf(f, "};\n\n#endif\n");
    fclose(f);

    printf("Sucesso! Arquivo 'pesos_treinados.h' gerado com %d pesos ordenados corretamente para o Bao.\n", count);

    free(connections);
    fann_destroy(ann);
    return 0;
}