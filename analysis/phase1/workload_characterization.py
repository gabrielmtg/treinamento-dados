import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "..", "..", "data", "data_final_clean.csv")
    plots_dir = os.path.join(base_path, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading expansive data for advanced characterization...")
    df_atk = pd.read_csv(os.path.join(base_path, "..", "..", "data", "data_final_clean.csv"))
    df_b1 = pd.read_csv(os.path.join(base_path, "..", "..", "data", "data_final_clean1.csv"))
    df_b2 = pd.read_csv(os.path.join(base_path, "..", "..", "data", "data_final_clean2.csv"))
    df = pd.concat([df_atk, df_b1, df_b2], ignore_index=True)
    df = df[(df['CPU_CYCLES'] > 0) & (df['INSTRUCTIONS'] > 0)].copy()

    # Derived HPC Signatures
    df['IPC'] = df['INSTRUCTIONS'] / df['CPU_CYCLES']
    df['CACHE_MISS_RATE'] = df['CACHE_MISSES'] / df['INSTRUCTIONS']
    df['BRANCH_MISS_RATE'] = df['BRANCH_MISSES'] / df['INSTRUCTIONS']
    df['BUS_PRESSURE'] = df['CACHE_MISSES'] / df['CPU_CYCLES']
    
    # Advanced Combinatory Metric: Wasted Pipeline Ratio (Branch + Cache Misses vs Cycles)
    df['PIPELINE_WASTE_RATIO'] = (df['CACHE_MISSES'] + (df['BRANCH_MISSES'] * 5)) / df['CPU_CYCLES']

    # Map Labels
    df['Domain'] = df['LABEL'].apply(lambda x: "Attack" if x > 0 else "Benign")
    
    subset = df.sample(n=min(50000, len(df)), random_state=42)

    print("Formulating graphical matrices...")

    # 1. Pipeline Waste Disparity (Density Map)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Domain']=="Benign"], x="PIPELINE_WASTE_RATIO", fill=True, label="Secure Domain (Benign)", color="blue", clip=(0, 0.05))
    sns.kdeplot(data=df[df['Domain']=="Attack"], x="PIPELINE_WASTE_RATIO", fill=True, label="Untrusted Domain (Attack)", color="red", clip=(0, 0.05))
    plt.title("Execution Waste Isolation (Stall vs Compute Phase)", fontsize=14, fontweight='bold')
    plt.xlabel("Pipeline Waste Ratio (Penalties / Usable Cycles)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pipeline_waste_density.png"))
    plt.close()

    # 2. Advanced 2D Subspace Filtering (Branch Penalty vs IPC)
    plt.figure(figsize=(10, 6))
    sns.jointplot(
        data=subset, x='IPC', y='BRANCH_MISS_RATE', hue='Domain',
        palette={"Benign": "blue", "Attack": "red"}, alpha=0.3, s=20, height=8
    )
    plt.suptitle("Architectural Segmentation: IPC vs Branch Targeting", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ipc_branch_segmentation.png"))
    plt.close()

    # 3. Correlation Heatmaps showing internal Coupling Distortions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cols = ['IPC', 'BUS_PRESSURE', 'CACHE_MISS_RATE', 'BRANCH_MISS_RATE', 'PIPELINE_WASTE_RATIO']
    
    benign_corr = df[df['Domain'] == 'Benign'][cols].corr()
    attack_corr = df[df['Domain'] == 'Attack'][cols].corr()

    sns.heatmap(benign_corr, annot=True, cmap="Blues", ax=axes[0], fmt=".2f", cbar=False)
    axes[0].set_title("Coupling in Native Benign Workloads", fontweight='bold')
    
    sns.heatmap(attack_corr, annot=True, cmap="Reds", ax=axes[1], fmt=".2f", cbar=True)
    axes[1].set_title("Decoupled Signatures in Attack Execution", fontweight='bold')
    
    plt.suptitle("Ground Truth Matrix: How Malicious Code Distorts Hardware Coupling", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "domain_correlation.png"))
    plt.close()

    print("Writing deep statistical profile...")
    print(df.groupby('Domain')[cols].mean().to_string())

if __name__ == '__main__':
    main()
