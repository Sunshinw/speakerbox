import os
import torch
import numpy as np
import tqdm
import soundfile
import logging
import torch.nn.functional as F
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf

# --- Configuration ---
MODEL_PATH = r"exps/V" 
EVAL_PAIR  = r"lists/vctk/spk_test_pairs.txt"
EVAL_PATH  = r"C:\Users\sunshine\Desktop\DS_10283_3443\VCTK-Corpus-0.92\wav48_silence_trimmed"

# Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SpeakerboxVerifier:
    def __init__(self, model_path):
        log.info(f"🚀 Initializing Speakerbox Encoder from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
            
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model.eval()

    def eval_network(self, eval_list, eval_path):
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        
        # 1. Collect unique files
        files = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                files.extend([parts[1], parts[2]])
        setfiles = sorted(list(set(files)))

        # 2. Pre-compute Embeddings (Hybrid Strategy)
        log.info(f"Extracting embeddings for {len(setfiles)} unique files...")
        for rel in tqdm.tqdm(setfiles):
            # Normalize slashes for Windows
            clean_rel = rel.replace("\\", "/").strip("/")
            wav_path = Path(eval_path) / clean_rel
            
            if not wav_path.exists():
                log.warning(f"⚠️ File not found: {wav_path}. Skipping.")
                continue

            audio, sr = soundfile.read(str(wav_path))
            
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Segment length (2 seconds)
            max_audio = 2 * 16000 
            if audio.shape[0] < max_audio:
                audio = np.pad(audio, (0, max_audio - audio.shape[0]), 'wrap')
            
            feats = []
            startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            
            input_full = self.fe(audio, sampling_rate=16000, return_tensors="pt").to(device)
            input_seg = self.fe(feats, sampling_rate=16000, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                emb_full = self.model(input_full.input_values).logits
                emb_full = F.normalize(emb_full, p=2, dim=1)
                
                emb_seg = self.model(input_seg.input_values).logits
                emb_seg = F.normalize(emb_seg, p=2, dim=1)

            embeddings[rel] = [emb_full, emb_seg]

        # 3. Scoring Trials
        scores, labels = [], []
        log.info("Computing trial scores...")
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue
            
            lab, p1, p2 = parts[0], parts[1], parts[2]
            if p1 not in embeddings or p2 not in embeddings: continue
                
            e1_f, e1_s = embeddings[p1]
            e2_f, e2_s = embeddings[p2]
            
            score_f = torch.mean(torch.matmul(e1_f, e2_f.T))
            score_s = torch.mean(torch.matmul(e1_s, e2_s.T))
            
            final_score = ((score_f + score_s) / 2).cpu().item()
            scores.append(final_score)
            labels.append(int(lab))

        # 4. Final Metrics using tool.py logic
        # target_fa = [0.01] finds the threshold for a 1% False Acceptance rate
        ret, EER, fpr, fnr, threshold = tuneThresholdfromScore(scores, labels, [0.01,0.1])
        
        # Calculate minDCF with standard NIST parameters
        fnrs_err, fprs_err, thresholds_err = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs_err, fprs_err, thresholds_err, p_target=0.05, c_miss=1, c_fa=1)
        
        return EER, minDCF, threshold

if __name__ == "__main__":
    verifier = SpeakerboxVerifier(MODEL_PATH)
    EER, minDCF, threshold = verifier.eval_network(EVAL_PAIR, EVAL_PATH)

    print("\n" + "="*35)
    print("🔥 VOXCELEB-STYLE EVALUATION")
    print("="*35)
    print(f"Equal Error Rate : {EER:.2f}")
    print(f"Minimum DCF      : {minDCF:.4f}")
    print(f"Decision Thresh  : {threshold:.4f}")
    print("="*35)