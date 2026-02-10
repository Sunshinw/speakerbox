import logging
import re
from pathlib import Path
from speakerbox import eval_model
from RUN_THIS_FILE import prepare_prod_dataset
from transformers import Wav2Vec2FeatureExtractor

# Setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_DIR = Path("exps/V")
VCTK_ROOT = r"C:\Users\sunshine\Desktop\DS_10283_3443\VCTK-Corpus-0.92\wav48_silence_trimmed"

def get_metrics_from_md(md_path: Path) -> dict:
    """Extracts accuracy and loss from an existing results.md file."""
    content = md_path.read_text()
    # Using regex to find the float values in the markdown list
    acc = re.search(r"Accuracy:\*\* ([\d.]+)", content)
    loss = re.search(r"Loss:\*\* ([\d.]+)", content)
    return {
        "acc": float(acc.group(1)) if acc else 0.0,
        "loss": float(loss.group(1)) if loss else 0.0
    }

def run_archived_evals():
    fe = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
    
    # Identify epoch folders
    epoch_folders = sorted(list(MODEL_DIR.glob("final_speakerbox_epoch_*")), 
                           key=lambda x: int(x.name.split("_")[-1]))
    
    # We only load the dataset if we actually have new work to do
    dataset = None 

    for folder in epoch_folders:
        results_path = folder / "results.md"
        
        # 1. CHECK IF ALREADY DONE
        if results_path.exists():
            metrics = get_metrics_from_md(results_path)
            print(f"⏩ Skipping {folder.name} (Found existing results.md)")
            print(f"   📊 Cached Results -> Acc: {metrics['acc']:.4f}, Loss: {metrics['loss']:.4f}")
            continue

        # 2. IF NOT DONE, INITIALIZE DATA AND RUN
        if dataset is None:
            log.info("New evaluation needed. Preparing dataset...")
            dataset = prepare_prod_dataset(VCTK_ROOT, fe)
        
        if not (folder / "preprocessor_config.json").exists():
            fe.save_pretrained(folder)

        print(f"\n🚀 Evaluating: {folder.name}")
        acc, prec, rec, loss = eval_model(dataset["valid"], model_name=str(folder))
        print(f"✅ {folder.name} Results -> Acc: {acc:.4f}, Loss: {loss:.4f}")
        
if __name__ == "__main__":
    run_archived_evals()