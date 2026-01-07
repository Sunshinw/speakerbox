import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from speakerbox import train, eval_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# VCTK_ROOT = r"C:\Users\sunshine\Desktop\DS_10283_3443\VCTK-Corpus-0.92\wav48_silence_trimmed"
VCTK_ROOT = ""
OUTPUT_MODEL_NAME = "vctk_prod_model_full"
MIN_FILES = 5

def prepare_prod_dataset(root_path: str) -> DatasetDict:
    root = Path(root_path)
    data = []
    
    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} files. Processing...")

    for filepath in audio_files:
        parts = filepath.name.split("_")
        if len(parts) >= 2:
            data.append({
                "audio": str(filepath.resolve()), 
                "label": parts[0],             
                "conversation_id": parts[1],
                "duration": 1.0 
            })
            
    df = pd.DataFrame(data)
    
    counts = df.groupby("label")["conversation_id"].nunique()
    valid_speakers = counts[counts >= MIN_FILES].index
    df = df[df.label.isin(valid_speakers)]
    log.info(f"Training on {len(valid_speakers)} speakers")

    unique_ids = df.conversation_id.unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    train_df = df[df.conversation_id.isin(train_ids)]
    test_df = df[df.conversation_id.isin(test_ids)]

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })

    dataset = dataset.class_encode_column("label")

    return dataset

def main():
    dataset = prepare_prod_dataset(VCTK_ROOT)
    
    print("STARTING PRODUCTION TRAINING")
    
    model_path = train(
        dataset=dataset,
        model_name=OUTPUT_MODEL_NAME,
        max_duration=2.0,
        trainer_arguments_kws={
            "gradient_accumulation_steps": 2,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 3,
            "save_total_limit": 2,
            "fp16": True,
        }
    )
    
    metrics = eval_model(dataset["test"], str(model_path))
    print(f"Results: {metrics}")

if __name__ == "__main__":
    main()