import csv
import sys
import os

def clean_csv(input_file, output_file):
    print(f"Lendo de {input_file} e salvando dados processados em {output_file}...")
    
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        
        # Colunas que desejamos manter
        expected_columns = ["TIMESTAMP", "CPU_CYCLES", "INSTRUCTIONS", "CACHE_MISSES", "BRANCH_MISSES", "LABEL"]
        writer.writerow(expected_columns)
        
        # Encontrar os índices das colunas no arquivo original
        try:
            core_id_idx = header.index("CORE_ID")
            col_indices = {col: header.index(col) for col in expected_columns}
        except ValueError as e:
            print(f"Erro: Coluna não encontrada no cabeçalho: {e}")
            return
            
        count = 0
        for row in reader:
            if not row:
                continue
                
            core_id = row[core_id_idx].strip()
            
            # Altera o label para 1 se for vm3 (core 3), senão 0
            if core_id == '3':
                row[col_indices["LABEL"]] = '1'
            else:
                row[col_indices["LABEL"]] = '0'
                
            # Cria a nova linha apenas com as colunas desejadas
            new_row = [row[col_indices[col]] for col in expected_columns]
            writer.writerow(new_row)
            count += 1
            
    print(f"Concluído! {count} linhas processadas.")

if __name__ == '__main__':
    # O diretório do script para resolver os caminhos relativos de forma correta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(script_dir, 'data_final.csv')
    output_path = os.path.join(script_dir, 'data_final_clean.csv')
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo '{input_path}' não encontrado.")
        sys.exit(1)
        
    clean_csv(input_path, output_path)
