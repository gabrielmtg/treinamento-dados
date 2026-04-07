import sys
import os

def generate_giant_csv(input_file, output_file):
    print(f"Lendo dados textuais de {input_file} e gerando CSV puro em {output_file}...")
    
    header = "CORE_ID,TIMESTAMP,CPU_CYCLES,INSTRUCTIONS,CACHE_MISSES,BRANCH_MISSES,LABEL"
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        # Escreve o cabeçalho exatamente 1 vez
        fout.write(header + '\n')
        
        in_csv = False
        lines_written = 0
        
        for line in fin:
            stripped = line.strip()
            
            if stripped == "START_OF_CSV_DATA":
                in_csv = True
                continue
            elif stripped == "END_OF_CSV_DATA":
                in_csv = False
                continue
                
            if in_csv:
                # Ignora cabeçalhos extras que aparecem repetidos em blocos diferentes
                if stripped.startswith("CORE_ID") or stripped == "":
                    continue
                    
                parts = stripped.split(',')
                
                # Certifica que é uma linha de dados com 7 colunas e começando pelas CPUs
                if len(parts) == 7 and parts[0] in ['1', '2', '3']:
                    fout.write(','.join(parts) + '\n')
                    lines_written += 1
                    
    print(f"Sucesso! Novo arquivo CSV gigante criado com {lines_written} análises catalogadas.")

if __name__ == '__main__':
    # Usaremos o txt que já foi consertado pelos labels no passo anterior
    input_path = 'data/data_fixed.txt'
    output_path = 'data/data_final.csv'
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo '{input_path}' não encontrado.")
        sys.exit(1)
        
    generate_giant_csv(input_path, output_path)
