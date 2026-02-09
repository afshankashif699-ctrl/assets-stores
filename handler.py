import runpod
import torch
import os
import boto3
import numpy as np
import requests
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from huggingface_hub import snapshot_download

# --- AUTHENTICATION & PATHS ---
HF_TOKEN = "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" # Aapki Read Token
MODEL_PATH = "checkpoints/s1-mini"

# Worker start hote hi model download karega (Authenticated)
if not os.path.exists(MODEL_PATH):
    print("--- Downloading Gated Model Weights ---")
    snapshot_download(
        repo_id='fishaudio/openaudio-s1-mini', 
        local_dir=MODEL_PATH, 
        token=HF_TOKEN
    )

# --- INFERENCE ENGINE ---
from fish_speech.utils.inference import load_checkpoint, generate_tokens, decode_audio

# Model ko VRAM mein load karein
device = "cuda" if torch.cuda.is_available() else "cpu"
model_manager = load_checkpoint(MODEL_PATH, device)

def handler(job):
    ins = job['input']
    text = ins.get("text", "")
    ref_audio_url = ins.get("ref_audio_url")
    
    if not text or not ref_audio_url:
        return {"error": "Missing input: text or ref_audio_url"}

    try:
        # 1. Reference Audio download aur set karein
        ref_data = requests.get(ref_audio_url).content
        with open("temp_ref", "wb") as f: f.write(ref_data)
        ref_audio = AudioSegment.from_file("temp_ref").set_frame_rate(44100).set_channels(1)
        ref_audio.export("refined_ref.wav", format="wav")

        # 2. Smart Chunking (100k characters fix)
        sentences = sent_tokenize(text)
        segments = []
        
        print(f"Generating voice for {len(sentences)} sentences...")
        for i, s in enumerate(sentences):
            tokens = generate_tokens(model=model_manager.llama, text=s, device=device)
            wav = decode_audio(model_manager.dac, tokens)
            segments.append(wav)
            
            # Har 10 sentences ke baad GPU cache saaf karein
            if i % 10 == 0: torch.cuda.empty_cache()

        # 3. Stitching & Export
        final_wav = np.concatenate(segments)
        int_audio = (final_wav * 32767).astype(np.int16)
        final_audio = AudioSegment(int_audio.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        
        out_file = f"{job['id']}.mp3"
        final_audio.export(out_file, format="mp3", bitrate="192k")

        # 4. S3 Upload
        s3 = boto3.client('s3', 
            aws_access_key_id=os.getenv('S3_KEY'), 
            aws_secret_access_key=os.getenv('S3_SECRET')
        )
        s3.upload_file(out_file, os.getenv('S3_BUCKET'), out_file)
        
        return {"status": "success", "s3_url": f"https://{os.getenv('S3_BUCKET')}.s3.amazonaws.com/{out_file}"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})