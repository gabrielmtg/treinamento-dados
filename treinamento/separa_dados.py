import os
import glob
import random

# Configurações de Normalização
NORM_BRANCH = 100000000.0
NORM_CACHE = 50000000.0
NORM_INSTR = 200000000.0
NORM_CYCLES = 400000000.0
NORM_DELTA_TIME = 1000.0 

WINDOW_SIZE = 5

def time_to_ms(time_str):
    """Converte um timestamp string HH:MM:SS.ms ou HH:MM:SS:ms para millisegundos."""
    parts = time_str.split(':')
    if len(parts) >= 3:
        h = int(parts[0])
        m = int(parts[1])
        s_parts = parts[2].split('.')
        if len(s_parts) == 2:
            s, ms = int(s_parts[0]), int(s_parts[1])
        else: # Trata casos em que é tudo ':'
            s = int(parts[2])
            ms = int(parts[3]) if len(parts) > 3 else 0
        return (h * 3600 + m * 60 + s) * 1000 + ms
    return 0

def extract_windows_from_file(arquivo):
    """
    Lê um arquivo .txt e gera janelas deslizantes sem dependência do pandas.
    """
    amostras = []
    try:
        dados = []
        with open(arquivo, 'r') as f:
            prev_time_ms = 0
            for linha in f:
                l = linha.replace(',', ' ').strip()
                if l and l[0].isdigit():
                    partes = l.split()
                    if len(partes) >= 6:
                        tempo_str = partes[0]
                        cycles = float(partes[1])
                        instr = float(partes[2])
                        cache = float(partes[3])
                        branch = float(partes[4])
                        label = float(partes[5])
                        
                        curr_ms = time_to_ms(tempo_str)
                        delta_ms = curr_ms - prev_time_ms if prev_time_ms > 0 else 0.0
                        prev_time_ms = curr_ms
                        
                        dados.append([branch, cache, instr, cycles, delta_ms, label])

        if len(dados) < WINDOW_SIZE:
            return amostras

        for i in range(len(dados) - WINDOW_SIZE + 1):
            input_vector = []
            for j in range(WINDOW_SIZE):
                idx = i + j
                input_vector.append(dados[idx][0] / NORM_BRANCH)
                input_vector.append(dados[idx][1] / NORM_CACHE)
                input_vector.append(dados[idx][2] / NORM_INSTR)
                input_vector.append(dados[idx][3] / NORM_CYCLES)
                input_vector.append(dados[idx][4] / NORM_DELTA_TIME)
            
            label_real = dados[i + WINDOW_SIZE - 1][5]
            amostras.append((input_vector, label_real))

    except Exception as e:
        print(f"Erro em {os.path.basename(arquivo)}: {e}")
        
    return amostras

def obter_arquivos(nome_pasta):
    base_path = os.path.dirname(os.path.abspath(__file__))
    caminho_completo = os.path.normpath(os.path.join(base_path, "..", nome_pasta))
    return glob.glob(os.path.join(caminho_completo, "*.txt"))

def gerar_datasets():
    print("--- Iniciando Divisao 80/20 baseada em Arquivos (Pure Python) ---")
    
    arquivos_ataques = obter_arquivos("ataques")
    arquivos_benchmarks = obter_arquivos("benchmarks")
    
    random.seed(42)
    random.shuffle(arquivos_ataques)
    random.shuffle(arquivos_benchmarks)
    
    corte_ataques = int(len(arquivos_ataques) * 0.8)
    corte_benchmarks = int(len(arquivos_benchmarks) * 0.8)
    
    train_files = arquivos_ataques[:corte_ataques] + arquivos_benchmarks[:corte_benchmarks]
    val_files = arquivos_ataques[corte_ataques:] + arquivos_benchmarks[corte_benchmarks:]
    
    print(f"Treino: {len(train_files)} files. Validação: {len(val_files)} files.")

    # --- PROCESSA TREINO ---
    amostras_treino = []
    for arquivo in train_files:
        amostras_treino.extend(extract_windows_from_file(arquivo))
        
    random.shuffle(amostras_treino)
    
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados_treino_80.train")
    with open(train_path, "w") as f:
        f.write(f"{len(amostras_treino)} {WINDOW_SIZE * 5} 1\n")
        for inp, out in amostras_treino:
            f.write(" ".join(f"{v:.6f}" for v in inp) + "\n")
            f.write(f"{out:.1f}\n")

    # --- PROCESSA VALIDAÇÃO ---
    val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validacao_20_porcento_features.csv")
    with open(val_path, "w") as f:
        header = ["file_name", "window_idx"] + [f"f_{i}" for i in range(WINDOW_SIZE * 5)] + ["label"]
        f.write(",".join(header) + "\n")
        
        total_val = 0
        for arquivo in val_files:
            file_name = os.path.basename(arquivo)
            amostras = extract_windows_from_file(arquivo)
            for w_idx, (inp, out) in enumerate(amostras):
                linha = [file_name, str(w_idx)] + [f"{v:.6f}" for v in inp] + [f"{out:.1f}"]
                f.write(",".join(linha) + "\n")
                total_val += 1

    print(f"Salvo: {train_path} e {val_path}")

if __name__ == "__main__":
    gerar_datasets()