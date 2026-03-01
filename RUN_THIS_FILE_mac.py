
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Audio, Dataset, DatasetDict
from speakerbox import train, eval_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# VCTK_ROOT = "/Users/pakap/Documents/Senior/Code/ECAPA_Libri/VCTK_WAV"
VCTK_ROOT = "/Users/pakap/Documents/Senior/Code/Dataset/libri500_WAV"
# VCTK_ROOT = ""
OUTPUT_MODEL_NAME = "exps/libri500_model"
MIN_FILES = 1

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
    # --- 80-10-10 SPLIT LOGIC ---
    unique_files = df["audio"].unique()

    # 1. Split off 80% for training
    train_files, temp_files = train_test_split(
        unique_files, test_size=0.20, random_state=42
    )

    # 2. Split the remaining 20% into two halves (10% Valid, 10% Test)
    valid_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=42
    )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df[df["audio"].isin(train_files)], preserve_index=False),
        "valid": Dataset.from_pandas(df[df["audio"].isin(valid_files)], preserve_index=False),
        "test": Dataset.from_pandas(df[df["audio"].isin(test_files)], preserve_index=False)
    })

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.class_encode_column("label")

    return dataset

def main():
    dataset = prepare_prod_dataset(VCTK_ROOT)
    
    print("STARTING PRODUCTION TRAINING")
    
    model_path = train(
        dataset=dataset,
        model_name=OUTPUT_MODEL_NAME,
        max_duration=3.0,
        trainer_arguments_kws={
            "save_strategy": "steps",
            "gradient_accumulation_steps": 2,
            "per_device_train_batch_size": 8,
            "num_train_epochs": 10,
            "save_total_limit": 2,
            "fp16": False,
        }
    )

    dataset["test"] = dataset["test"].cast_column("audio", Audio(decode=False))
    metrics = eval_model(dataset["test"], OUTPUT_MODEL_NAME)
    print(f"Results: {metrics}")

if __name__ == "__main__":
    main()