import torch
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model

# 1. SETUP: Enable hidden state output
MODEL_PATH = r"exps/V/final_speakerbox_epoch_1"
# output_hidden_states=True is key to "seeing" inside the model layers
model_base = Wav2Vec2Model.from_pretrained(MODEL_PATH, output_hidden_states=True)
print(model_base)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH, output_hidden_states=True)
print(model)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)

# 2. RAW INPUT: Load 1 second of audio
speech, sr = librosa.load(librosa.ex("choice"), duration=1.0, sr=16000)
print(f"📥 1. Raw Audio Shape: {speech.shape} (16k samples/sec)")

# 3. PRE-PROCESSING
inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
print(f"⚙️  2. Pre-processed Shape: {inputs.input_values.shape} (Normalized Tensor)")

# 4. THE DEEP DIVE: Extracting Important Layers
with torch.no_grad():
    outputs = model(inputs.input_values)
    
    # --- LAYER A: The CNN Latent Features ---
    # These are the local acoustic features (the "texture" of the sound)
    cnn_features = outputs.hidden_states[0] 
    
    # --- LAYER B: The Transformer Output ---
    # The final contextual representation before classification
    last_hidden_state = outputs.hidden_states[-1]
    
    # --- LAYER C: The Logits (The Identification Head) ---
    logits = outputs.logits

# 5. SIZE & DIMENSION SUMMARY
print(f"\n--- Layer Analysis ---")
print(f"🧩 CNN Latents:    {cnn_features.shape} -> (Batch, Time Frames, 768 features)")
print(f"🧠 Context State:  {last_hidden_state.shape} -> (Batch, Time Frames, 768 features)")
print(f"🧬 Speaker Embed: {logits.shape} -> (Batch, 256-dim Embedding / 110 Classes)")

# 6. DECISION LOGIC
probs = F.softmax(logits, dim=-1)
pred_id = torch.argmax(probs, dim=-1).item()
confidence = probs[0][pred_id].item()
speaker_name = model.config.id2label[pred_id]

print(f"\n--- Final Decision ---")
print(f"📤 Speaker: '{speaker_name}' | Confidence: {confidence*100:.2f}%")

import torch
from transformers import Wav2Vec2ForSequenceClassification

MODEL_PATH = r"exps/V"
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

def print_param_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📁 Total Parameters: {total_params:,}")
    print(f"🔥 Trainable Parameters: {trainable_params:,}")
    print("-" * 30)
    
    # Detailed Layer Breakdown
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        print(f"{name:18}: {count:,} params")

print_param_summary(model)