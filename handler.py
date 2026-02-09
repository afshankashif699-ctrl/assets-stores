import runpod
import torch
import os
import boto3
import requests
import numpy as np
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from huggingface_hub import snapshot_download

# --- AUTHENTICATION ---
HF_TOKEN = "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"
MODEL_PATH = "checkpoints/s1-mini"

# Worker start hote hi download karega (RunPod ki disk space use hogi, Codespaces ki nahi)
if not os.path.exists(MODEL_PATH):
    print("--- Downloading Model Weights on RunPod ---")
    snapshot_download(repo_id='fishaudio/openaudio-s1-mini', local_dir=MODEL_PATH, token=HF_TOKEN)

from fish_speech.utils.inference import load_checkpoint, generate_tokens, decode_audio

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_manager = load_checkpoint(MODEL_PATH, device)

def handler(job):
    ins = job['input']
    text = ins.get("text", "")
    ref_audio_url = ins.get("ref_audio_url")

    if not text or not ref_audio_url:
        return {"error": "Missing input: 'text' or 'ref_audio_url'"}

    try:
        # 1. Reference Audio Download
        ref_path = "temp_ref.wav"
        with open(ref_path, "wb") as f:
            f.write(requests.get(ref_audio_url).content)
        
        ref_audio = AudioSegment.from_file(ref_path).set_frame_rate(44100).set_channels(1)
        ref_audio.export("refined_ref.wav", format="wav")

        # 2. 100k Chars Chunking
        sentences = sent_tokenize(text)
        segments = []
        for i, s in enumerate(sentences):
            tokens = generate_tokens(model=model_manager.llama, text=s, device=device)
            wav = decode_audio(model_manager.dac, tokens)
            segments.append(wav)
            if i % 10 == 0: torch.cuda.empty_cache()

        # 3. Stitch & Upload
        final_wav = np.concatenate(segments)
        int_audio = (final_wav * 32767).astype(np.int16)
        combined_audio = AudioSegment(int_audio.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        
        out_file = f"{job['id']}.mp3"
        combined_audio.export(out_file, format="mp3", bitrate="192k")
        
        s3 = boto3.client('s3', 
            aws_access_key_id=os.getenv('S3_KEY'), 
            aws_secret_access_key=os.getenv('S3_SECRET')
        )
        s3.upload_file(out_file, os.getenv('S3_BUCKET'), out_file)
        
        return {"s3_url": f"https://{os.getenv('S3_BUCKET')}.s3.amazonaws.com/{out_file}"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})