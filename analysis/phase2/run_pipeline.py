import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.normpath(os.path.join(base_path, ".."))
    
    print("=== PIPELINE DE TREINAMENTO E VALIDAÇÃO DE PMU ===")
    
    # 1. Fixed labels (Garante que os arquivos de ataque tenham label 1.0)
    print("\n1. Corrigindo labels de ataques...")
    run_command([sys.executable, "label_fixer.py"], cwd=root_path)
    
    # 2. Extract Data and Split (Group-based 80/20)
    print("\n2. Preparando os dados de Treino e Validação...")
    run_command([sys.executable, "separa_dados.py"], cwd=base_path)
    
    # 3. Compile C Trainer
    print("\n3. Compilando o FANN Trainer...")
    run_command(["gcc", "train_80.c", "-o", "treina_80", "-lfann", "-lm"], cwd=base_path)
    
    # 4. Run C Trainer
    print("\n4. Treinando o modelo...")
    run_command(["./treina_80"], cwd=base_path)
    
    # 5. Run Python Validation
    print("\n5. Executando Validação e Métricas (Python)...")
    run_command([sys.executable, "validate_model.py"], cwd=base_path)
    
    print("\n=== PIPELINE CONCLUÍDO COM SUCESSO ===")

if __name__ == "__main__":
    main()
