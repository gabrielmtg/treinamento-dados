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

# Removemos o 'label_padrao' da assinatura da função
def processar_pasta(nome_pasta):
    amostras_locais = []
    base_path = os.path.dirname(os.path.abspath(__file__))
    caminho_completo = os.path.normpath(os.path.join(base_path, "..", nome_pasta))
    
    arquivos = glob.glob(os.path.join(caminho_completo, "*.txt"))
    
    if not arquivos:
        print(f"[AVISO] Nenhum arquivo .txt em: {caminho_completo}")
        return []

    for arquivo in arquivos:
        print(f"Lendo: {os.path.basename(arquivo)}")
        try:
            linhas_limpas = []
            with open(arquivo, 'r') as f:
                for linha in f:
                    l = linha.replace(',', ' ').strip()
                    # Garante que a linha começa com um número (Timestamp)
                    if l and l[0].isdigit():
                        partes = l.split()
                        if len(partes) > 0:
                            tempo = partes[0]
                            # Conserta o timestamp de 13:09:49:770 para 13:09:49.770
                            if tempo.count(':') == 3:
                                last_colon = tempo.rfind(':')
                                tempo = tempo[:last_colon] + '.' + tempo[last_colon+1:]
                                partes[0] = tempo
                            linhas_limpas.append(" ".join(partes))

            if not linhas_limpas:
                continue

            df = pd.read_csv(StringIO("\n".join(linhas_limpas)), sep=r'\s+', header=None, engine='python')
            df.columns = ['time', 'cycles', 'instr', 'cache', 'branch', 'label_orig']

            df['time_dt'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
            df['delta_ms'] = df['time_dt'].diff().dt.total_seconds() * 1000
            df['delta_ms'] = df['delta_ms'].fillna(0)

            # Cria a Sliding Window (Janela Deslizante)
            for i in range(len(df) - WINDOW_SIZE + 1):
                window = df.iloc[i:i+WINDOW_SIZE]
                input_vector = []
                for _, row in window.iterrows():
                    input_vector.append(float(row['branch']) / NORM_BRANCH)
                    input_vector.append(float(row['cache']) / NORM_CACHE)
                    input_vector.append(float(row['instr']) / NORM_INSTR)
                    input_vector.append(float(row['cycles']) / NORM_CYCLES)
                    input_vector.append(float(row['delta_ms']) / NORM_DELTA_TIME)
                
                # LÊ O LABEL REAL DO LOG DA PLACA! (Pega da última linha da janela)
                label_real = float(window.iloc[-1]['label_orig'])
                amostras_locais.append((input_vector, label_real))
        
        except Exception as e:
            print(f"      -> Erro técnico no arquivo {os.path.basename(arquivo)}: {e}")
            
    return amostras_locais

def gerar_dataset_fann():
    print("--- Iniciando Processamento de Datasets ---")
    
    # Agora não importa em qual pasta o log está, o script confia no label do arquivo
    dados_ataques = processar_pasta("ataques")
    dados_benchmarks = processar_pasta("benchmarks")
    
    todas_amostras = dados_ataques + dados_benchmarks
    
    if not todas_amostras:
        print("\n[ERRO] Nenhuma amostra gerada.")
        return

    # O SEGREDO DO TREINAMENTO ESTÁVEL: Embaralhar as amostras!
    random.shuffle(todas_amostras)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados_projeto.train")
    with open(output_path, "w") as f:
        # Cabeçalho exigido pela FANN
        f.write(f"{len(todas_amostras)} 25 1\n")
        for inp, out in todas_amostras:
            f.write(" ".join(f"{v:.6f}" for v in inp) + "\n")
            f.write(f"{out:.1f}\n")

    print(f"\n--- SUCESSO ---")
    print(f"Arquivo: {output_path}")
    print(f"Total de Janelas Processadas e Embaralhadas: {len(todas_amostras)}")

if __name__ == "__main__":
    gerar_dataset_fann()