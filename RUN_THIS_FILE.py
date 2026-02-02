
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Audio, Dataset, DatasetDict
from speakerbox import train, eval_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# VCTK_ROOT = "/Users/pakap/Documents/Senior/Code/ECAPA_Libri/VCTK_WAV"
# VCTK_ROOT = "/Users/pakap/Documents/Senior/Code/ECAPA_TDNN/Dataset/libri500_WAV"
VCTK_ROOT = r"C:\Users\sunshine\Desktop\DS_10283_3443\VCTK-Corpus-0.92\wav48_silence_trimmed"
OUTPUT_MODEL_NAME = "exps/VTCK"
MIN_FILES = 1

def prepare_prod_dataset(root_path: str, feature_extractor) -> DatasetDict:
    root = Path(root_path)
    data = []
    
    # 1. Gather all VCTK audio files
    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} files. Processing...")

    for filepath in audio_files:
        parts = filepath.name.split("_") 
        if len(parts) >= 2:
            data.append({
                "audio": str(filepath.resolve()), 
                "label": parts[0], 
            })
            
    df = pd.DataFrame(data)
    
    # 2. Split files to ensure unseen speakers are isolated for the test set
    unique_files = df["audio"].unique()
    train_files, temp_files = train_test_split(unique_files, test_size=0.20, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42)

    ds = DatasetDict({
        "train": Dataset.from_pandas(df[df["audio"].isin(train_files)], preserve_index=False),
        "valid": Dataset.from_pandas(df[df["audio"].isin(valid_files)], preserve_index=False),
        "test": Dataset.from_pandas(df[df["audio"].isin(test_files)], preserve_index=False)
    })

    # 3. CAST TO AUDIO: This enables internal decoding for the map function
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    # 4. PRE-CACHING LOGIC: Move extraction out of the training loop
    log.info("Starting GPU-optimized Feature Pre-Caching...")
    
    def preprocess_for_cache(batch):
        # Decode audio arrays from the batch
        audio_arrays = [x["array"] for x in batch["audio"]]
        
        # Extract features (Wav2Vec2 style)
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=16000, 
            max_length=int(16000 * 3.0), # Fixed 3.0s window
            truncation=True, 
            padding="max_length",
            return_tensors="np"
        )
        return {"input_values": inputs["input_values"]}

    # Perform the mapping
    ds = ds.map(
        preprocess_for_cache, 
        batched=True, 
        batch_size=32, 
        remove_columns=["audio"] # Remove raw audio to keep the tensor cache light
    )
    
    # 5. FINAL PREP: Encode labels and set format for PyTorch
    ds = ds.class_encode_column("label")
    ds.set_format("torch", columns=["input_values", "label"])
    
    return ds

def main():
    from transformers import Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(DEFAULT_BASE_MODEL)
    
    # 2. Pass the extractor into the preparation function
    dataset = prepare_prod_dataset(VCTK_ROOT, feature_extractor)
    
    print("STARTING PRODUCTION TRAINING")
    
    model_path = train(
        dataset=dataset,
        model_name=OUTPUT_MODEL_NAME,
        max_duration=3.0,
        trainer_arguments_kws={
            "save_strategy": "steps",
            "save_steps": 100,
            "fp16": True,      # Enable for speed
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "dataloader_num_workers": 0, # Windows stability
            "remove_unused_columns": False,  
        }
    )
    
    metrics = eval_model(dataset["test"], str(model_path))
    print(f"Results: {metrics}")

if __name__ == "__main__":
    main()