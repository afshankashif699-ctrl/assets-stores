import runpod
import torch
import os
import boto3
import requests
import numpy as np
from TTS.api import TTS
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("--- Loading XTTS-v2 Model ---")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def handler(job):
    ins = job['input']
    text = ins.get("text", "")
    speaker_url = ins.get("speaker_wav_url")

    if not text or not speaker_url:
        return {"error": "Missing input: 'text' or 'speaker_wav_url'"}

    try:
        # 1. Reference Audio Download
        ref_path = "speaker_ref.wav"
        with open(ref_path, "wb") as f:
            f.write(requests.get(speaker_url).content)

        # 2. 100k Characters Chunking
        sentences = sent_tokenize(text)
        combined_audio = AudioSegment.empty()
        
        for i, s in enumerate(sentences):
            temp_file = f"temp_{i}.wav"
            tts.tts_to_file(
                text=s, speaker_wav=ref_path, language="en", file_path=temp_file,
                temperature=0.65, repetition_penalty=5.0, top_p=0.8
            )
            combined_audio += AudioSegment.from_wav(temp_file)
            os.remove(temp_file)
            if i % 20 == 0: torch.cuda.empty_cache()

        # 3. Export & S3 Upload
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