import pandas as pd
import os
import glob
from io import StringIO
import random

# Configurações de Normalização
NORM_BRANCH = 100000000.0
NORM_CACHE = 50000000.0
NORM_INSTR = 200000000.0
NORM_CYCLES = 400000000.0
NORM_DELTA_TIME = 1000.0 

WINDOW_SIZE = 5

def processar_e_dividir(nome_pasta, amostras_treino, linhas_validacao):
    base_path = os.path.dirname(os.path.abspath(__file__))
    caminho_completo = os.path.normpath(os.path.join(base_path, "..", nome_pasta))
    
    arquivos = glob.glob(os.path.join(caminho_completo, "*.txt"))
    
    for arquivo in arquivos:
        print(f"Processando: {os.path.basename(arquivo)}")
        try:
            linhas_limpas = []
            with open(arquivo, 'r') as f:
                for linha in f:
                    l = linha.replace(',', ' ').strip()
                    if l and l[0].isdigit():
                        partes = l.split()
                        if len(partes) > 0:
                            tempo = partes[0]
                            if tempo.count(':') == 3:
                                last_colon = tempo.rfind(':')
                                tempo = tempo[:last_colon] + '.' + tempo[last_colon+1:]
                                partes[0] = tempo
                            linhas_limpas.append(" ".join(partes))

            if not linhas_limpas:
                continue
            
            # DIVISÃO 80/20 (Série Temporal Sequencial)
            ponto_corte = int(len(linhas_limpas) * 0.8)
            treino_raw = linhas_limpas[:ponto_corte]
            validacao_raw = linhas_limpas[ponto_corte:]

            # --- 1. PROCESSA TREINO (Cria janelas deslizantes) ---
            if len(treino_raw) >= WINDOW_SIZE:
                df = pd.read_csv(StringIO("\n".join(treino_raw)), sep=r'\s+', header=None, engine='python')
                df.columns = ['time', 'cycles', 'instr', 'cache', 'branch', 'label_orig']
                df['time_dt'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
                df['delta_ms'] = df['time_dt'].diff().dt.total_seconds() * 1000
                df['delta_ms'] = df['delta_ms'].fillna(0)

                # --- OTIMIZAÇÃO: Converte colunas para arrays rápidos ---
                branch_vals = df['branch'].values
                cache_vals = df['cache'].values
                instr_vals = df['instr'].values
                cycles_vals = df['cycles'].values
                delta_vals = df['delta_ms'].values
                label_vals = df['label_orig'].values

                # Loop sem usar df.iterrows() (100x mais rápido)
                for i in range(len(df) - WINDOW_SIZE + 1):
                    input_vector = []
                    
                    # Constrói a janela de 5 amostras usando índices de array
                    for j in range(WINDOW_SIZE):
                        idx = i + j
                        input_vector.append(float(branch_vals[idx]) / NORM_BRANCH)
                        input_vector.append(float(cache_vals[idx]) / NORM_CACHE)
                        input_vector.append(float(instr_vals[idx]) / NORM_INSTR)
                        input_vector.append(float(cycles_vals[idx]) / NORM_CYCLES)
                        input_vector.append(float(delta_vals[idx]) / NORM_DELTA_TIME)
                    
                    # Pega o label da última linha da janela
                    label_real = float(label_vals[i + WINDOW_SIZE - 1])
                    amostras_treino.append((input_vector, label_real))

            # --- 2. SALVA VALIDAÇÃO (Formato RAW para o Validador C) ---
            # O validador C espera: CORE_ID, TIMESTAMP, CYCLES, INSTR, CACHE, BRANCH, LABEL
            for linha in validacao_raw:
                partes = linha.split()
                cycles = partes[1]
                instr = partes[2]
                cache = partes[3]
                branch = partes[4]
                label = partes[5]
                # Passamos o Core ID como 0 e o Timestamp como 0 (pois o C ignora-o e usa 200ms fixos)
                linhas_validacao.append(f"0,0,{cycles},{instr},{cache},{branch},{label}")

        except Exception as e:
            print(f"Erro em {os.path.basename(arquivo)}: {e}")

def gerar_datasets():
    print("--- A Iniciar Divisao 80/20 ---")
    amostras_treino = []
    linhas_validacao = ["CORE_ID,TIMESTAMP,CPU_CYCLES,INSTRUCTIONS,CACHE_MISSES,BRANCH_MISSES,LABEL"]

    processar_e_dividir("ataques", amostras_treino, linhas_validacao)
    processar_e_dividir("benchmarks", amostras_treino, linhas_validacao)

    # Guarda o ficheiro de Treino (80%) baralhado
    random.shuffle(amostras_treino)
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados_treino_80.train")
    with open(train_path, "w") as f:
        f.write(f"{len(amostras_treino)} 25 1\n")
        for inp, out in amostras_treino:
            f.write(" ".join(f"{v:.6f}" for v in inp) + "\n")
            f.write(f"{out:.1f}\n")

    # Guarda o ficheiro de Validação (20%) em bruto
    val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validacao_20_porcento.csv")
    with open(val_path, "w") as f:
        f.write("\n".join(linhas_validacao) + "\n")

    print(f"\n--- SUCESSO ---")
    print(f"Treino (80% gerou): {train_path} ({len(amostras_treino)} janelas preparadas)")
    print(f"Validacao (20% guardou): {val_path} ({len(linhas_validacao)-1} linhas de log bruto)")

if __name__ == "__main__":
    gerar_datasets()