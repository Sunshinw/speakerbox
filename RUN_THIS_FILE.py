import logging
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Audio, Dataset, DatasetDict
from speakerbox import train, eval_model

# Standard logging setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Configuration Constants ---
# DEFAULT_BASE_MODEL must be defined for the feature extractor and trainer
DEFAULT_BASE_MODEL = "superb/wav2vec2-base-superb-sid"
VCTK_ROOT = r"C:\Users\sunshine\Desktop\DS_10283_3443\VCTK-Corpus-0.92\wav48_silence_trimmed"
# VCTK_ROOT = r"C:\Users\sunshine\Desktop\github\speakerbox\example-speakerbox-dataset"
OUTPUT_MODEL_NAME = "exps/V"
MIN_FILES = 1

def prepare_prod_dataset(root_path: str, feature_extractor) -> DatasetDict:
    # cache_path = Path("vctk_processed_cache")
    
    # # 1. IMMEDIATE CHECK: If cache exists, load and return instantly
    # if cache_path.exists():
    #     log.info(f"🚀 Loading processed dataset from disk: {cache_path}")
    #     from datasets import load_from_disk
    #     return load_from_disk(str(cache_path))

    root = Path(root_path)
    data = []
    
    # 2. Gather VCTK audio files
    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} files. Processing...")

    # For VCTK, the speaker ID is the first part of the filename before the underscore
    # for filepath in audio_files:
    #     parts = filepath.name.split("_") 
    #     if len(parts) >= 2:
    #         data.append({
    #             "audio": str(filepath.resolve()), 
    #             "label": parts[0], 
    #         })
    
    # # For example Speakerbox dataset filenames like '5e881a137b6d-monologue_9-pedersen-chunk_0.wav'
    for filepath in audio_files:
        # 1. Get the filename without extension (e.g., '5e881a137b6d-monologue_9-pedersen-chunk_0')
        name_stem = filepath.stem 
        
        # 2. Split by underscore first to isolate the main segments
        parts = name_stem.split("_") 
        
        if len(parts) >= 2:
            # In Speakerbox examples, the label is often the name following a dash
            # Adjust this logic based on whether you want 'pedersen' or the UUID
            label = parts[0].split("-")[0] 
            
            data.append({
                "audio": str(filepath.resolve()), 
                "label": label, 
            })
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True) # 🔥 SHUFFLE HERE
    
    unique_files = df["audio"].unique()    
    # df = pd.DataFrame(data)
    # unique_files = df["audio"].unique()
    train_files, temp_files = train_test_split(unique_files, test_size=0.20, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42)

    ds = DatasetDict({
        "train": Dataset.from_pandas(df[df["audio"].isin(train_files)], preserve_index=False),
        "valid": Dataset.from_pandas(df[df["audio"].isin(valid_files)], preserve_index=False),
        "test": Dataset.from_pandas(df[df["audio"].isin(test_files)], preserve_index=False)
    })

    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.class_encode_column("label")
    
    # # 3. CAST TO AUDIO: Enables internal decoding
    # ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    # # 4. PRE-CACHING LOGIC: Move extraction out of training loop
    # log.info("Starting GPU-optimized Feature Pre-Caching (Memory Safe)...")
    
    # def preprocess_for_cache(batch):
    #     audio_arrays = [x["array"] for x in batch["audio"]]
    #     inputs = feature_extractor(
    #         audio_arrays, 
    #         sampling_rate=16000, 
    #         max_length=int(16000 * 3.0),
    #         truncation=True, 
    #         padding="max_length",
    #         return_tensors="np"
    #     )
    #     return {"input_values": inputs["input_values"]}

    # # 5. MEMORY-SAFE MAPPING: Fixes realloc error
    # # Perform the mapping with extreme memory safety
    # ds = ds.map(
    #     preprocess_for_cache, 
    #     batched=True, 
    #     batch_size=8,              # Lowered from 16 to 8 to keep RAM footprint tiny
    #     writer_batch_size=50,      # Lowered from 100 to 50 to avoid the 17GB realloc
    #     num_proc=1,                # Keep at 1 for Windows to avoid multiprocess memory spikes
    #     remove_columns=["audio"],
    #     keep_in_memory=False,      # Force writing to disk immediately
    #     load_from_cache_file=True  # 🚀 CRITICAL: This will let you skip the 63% you already did
    # )
    
    # 6. FINAL PREP
    # ds = ds.class_encode_column("label")
    # ds.set_format("torch", columns=["input_values", "label"])
    
    # log.info(f"💾 Saving processed dataset to: {cache_path}")
    # ds.save_to_disk(str(cache_path))
    
    return ds

def main():
    from transformers import Wav2Vec2FeatureExtractor
    
    # Initialize the extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(DEFAULT_BASE_MODEL)
    
    # Prepare the pre-cached dataset
    dataset = prepare_prod_dataset(VCTK_ROOT, feature_extractor)
    
    log.info("STARTING PRODUCTION TRAINING ON GPU")
    
    # model_path = train(
    #     dataset=dataset,
    #     model_name=OUTPUT_MODEL_NAME,
    #     max_duration=3.0,
    #     trainer_arguments_kws={
    #         "save_strategy": "steps",
    #         "save_steps": 100,
    #         "fp16": True,                    # Uses NVIDIA Tensor Cores for speedup
    #         "per_device_train_batch_size": 8,
    #         "num_train_epochs": 10,  # 🔥 Explicitly set this here
    #         "gradient_accumulation_steps": 2, # Total batch size = 16
    #         "dataloader_num_workers": 0,      # Required for Windows stability
    #         "remove_unused_columns": False,   # Ensures input_values are not dropped
    #         "dataloader_pin_memory": True,    # Faster CPU-to-GPU transfer
    #     }
    # )
    
    # Evaluate using the isolated test set
    metrics = eval_model(dataset["test"], "exps\V")
    log.info(f"Final Evaluation Results: {metrics}")

if __name__ == "__main__":
    main()