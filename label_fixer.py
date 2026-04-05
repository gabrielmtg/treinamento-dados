import os
import glob

# Pasta onde estão os arquivos de ataque
PASTA_ATAQUES = "ataques"

def rotular_como_ataque(pasta):
    # Procura por todos os arquivos .txt ou .log (ajuste a extensão se necessário)
    arquivos = glob.glob(os.path.join(pasta, "*.txt"))
    
    if not arquivos:
        print(f"Nenhum arquivo encontrado na pasta '{pasta}'")
        return

    for caminho_arquivo in arquivos:
        linhas_modificadas = []
        
        with open(caminho_arquivo, 'r') as f:
            for linha in f:
                partes = linha.strip().split()
                
                # Verifica se a linha tem o formato esperado (6 colunas: time + 4 métricas + label)
                if len(partes) == 6:
                    # Mantém as 5 primeiras colunas e força a última para 1.0
                    nova_linha = f"{' '.join(partes[:5])} 1.0\n"
                    linhas_modificadas.append(nova_linha)
                else:
                    # Se for cabeçalho ou linha vazia, mantém como está
                    linhas_modificadas.append(linha)

        # Sobrescreve o arquivo com os novos labels
        with open(caminho_arquivo, 'w') as f:
            f.writelines(linhas_modificadas)
        
        print(f"Processado: {os.path.basename(caminho_arquivo)} -> Todos labels em 1.0")

if __name__ == "__main__":
    rotular_como_ataque(PASTA_ATAQUES)