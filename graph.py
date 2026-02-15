import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(r"C:\Users\sunshine\Desktop\github\speakerbox\exps\V")
ACC_GRAPH = BASE_DIR / "accuracy_progress.png"
LOSS_GRAPH = BASE_DIR / "loss_progress.png"

def extract_metrics(file_path):
    """Parses Accuracy and Loss from the results.md template."""
    content = file_path.read_text()
    acc_match = re.search(r"\*\*Accuracy:\*\*\s*([\d\.]+)", content)
    loss_match = re.search(r"\*\*Validation Loss:\*\*\s*([\d\.]+)", content)
    
    acc = float(acc_match.group(1)) if acc_match else None
    loss = float(loss_match.group(1)) if loss_match else None
    return acc, loss

def save_individual_plots():
    epochs, accuracies, losses = [], [], []

    # 1. Gather Data
    epoch_folders = sorted(
        list(BASE_DIR.glob("final_speakerbox_epoch_*")),
        key=lambda x: int(x.name.split("_")[-1])
    )

    for folder in epoch_folders:
        epoch_num = int(folder.name.split("_")[-1])
        results_file = folder / "results.md"
        
        if results_file.exists():
            acc, loss = extract_metrics(results_file)
            if acc is not None and loss is not None:
                epochs.append(epoch_num)
                accuracies.append(acc)
                losses.append(loss)

    if not epochs:
        print("❌ Data not found.")
        return

    # 2. Generate Accuracy Graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker='o', color='#2c7bb6', linewidth=2.5, label='Validation Accuracy')
    plt.title('Accuracy Learning Curve')
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy (0.0 - 1.0)')
    plt.ylim(min(accuracies) - 0.005, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(ACC_GRAPH)
    print(f"✅ Accuracy graph saved: {ACC_GRAPH}")
    plt.close()

    # 3. Generate Loss Graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='s', color='#d7191c', linewidth=2.5, label='Validation Loss')
    plt.title('Loss Learning Curve')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(LOSS_GRAPH)
    print(f"✅ Loss graph saved: {LOSS_GRAPH}")
    plt.close()

if __name__ == "__main__":
    save_individual_plots()