import sys
import os

def fix_labels(input_file, output_file):
    print(f"Lendo de {input_file} e salvando dados corrigidos em {output_file}...")
    
    in_csv = False
    last_raw_label = None
    current_task_index = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            stripped = line.strip()
            
            if stripped == "START_OF_CSV_DATA":
                in_csv = True
                fout.write(line)
                continue
            elif stripped == "END_OF_CSV_DATA":
                in_csv = False
                fout.write(line)
                continue
                
            if in_csv:
                parts = stripped.split(',')
                # Verifica se a linha é uma linha de dados CSV válida
                if len(parts) == 7 and parts[0] in ['1', '2', '3']:
                    core_id = parts[0]
                    # Corrige os labels APENAS da VM 3 (onde rodam os ataques)
                    if core_id == '3':
                        raw_label = parts[6]
                        
                        if last_raw_label is None:
                            last_raw_label = raw_label
                            
                        # Detecta transição de tarefa com base na mudança de label bruto
                        if raw_label != last_raw_label:
                            # A transição do label 0 (atraso inicial) para 1 ainda pertence à 1ª tarefa
                            if last_raw_label == '0' and raw_label == '1':
                                pass
                            else:
                                current_task_index += 1
                                
                            last_raw_label = raw_label
                            
                        # Mapeia as tarefas sequencialmente: 1, 2, 3, 1, 2, 3...
                        # O current_task_index começa em 0
                        new_label = (current_task_index % 3) + 1
                        parts[6] = str(new_label)
                        fout.write(','.join(parts) + '\n')
                        continue
            
            # Se não modificado (como os dados de outros cores), escreve igual
            # Mantendo a quebra de linha original (strip só remove espaço, usaremos newLine unix)
            fout.write(line)

    print("Sucesso! Os labels foram corrigidos com base no deslocamento.")

if __name__ == '__main__':
    input_path = 'data/data.txt'
    output_path = 'data/data_fixed.txt'
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo '{input_path}' não encontrado.")
        sys.exit(1)
        
    fix_labels(input_path, output_path)
