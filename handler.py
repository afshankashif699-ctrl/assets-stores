import runpod
import torch
import os
import boto3
import requests
import numpy as np
from TTS.api import TTS
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize

# --- Initial Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model ko ek hi dafa load karein taake speed fast ho
print("--- Loading XTTS-v2 Model ---")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def handler(job):
    ins = job['input']
    text = ins.get("text", "")
    speaker_url = ins.get("speaker_wav_url") # 10-20 seconds ki reference audio ka link

    if not text or not speaker_url:
        return {"error": "Input missing: 'text' and 'speaker_wav_url' are required."}

    try:
        # 1. Reference audio download karein
        ref_path = "speaker_ref.wav"
        ref_data = requests.get(speaker_url).content
        with open(ref_path, "wb") as f:
            f.write(ref_data)

        # 2. Smart Chunking (100k characters handle karne ke liye)
        sentences = sent_tokenize(text)
        combined_audio = AudioSegment.empty()
        print(f"Starting processing of {len(sentences)} sentences...")

        for i, sentence in enumerate(sentences):
            temp_file = f"temp_{i}.wav"
            
            # Ultra-realistic settings ke saath voice generate karein
            tts.tts_to_file(
                text=sentence,
                speaker_wav=ref_path,
                language="en", 
                file_path=temp_file,
                temperature=0.65,      # Emotion aur stability ka balance
                repetition_penalty=5.0, # Robotic repetition ko rokta hai
                top_p=0.8
            )
            
            # Audio segments ko join karein
            segment = AudioSegment.from_wav(temp_file)
            combined_audio += segment
            
            # Memory aur disk clean karein
            os.remove(temp_file)
            if i % 20 == 0:
                torch.cuda.empty_cache()

        # 3. High-Quality MP3 mein export karein
        output_filename = f"{job['id']}.mp3"
        combined_audio.export(output_filename, format="mp3", bitrate="192k")

        # 4. S3 par upload karein (Lambi files ke liye zaruri hai)
        s3 = boto3.client('s3', 
            aws_access_key_id=os.getenv('S3_KEY'), 
            aws_secret_access_key=os.getenv('S3_SECRET')
        )
        s3.upload_file(output_filename, os.getenv('S3_BUCKET'), output_filename)
        
        # Cleanup
        os.remove(output_filename)
        os.remove(ref_path)

        return {
            "status": "success", 
            "s3_url": f"https://{os.getenv('S3_BUCKET')}.s3.amazonaws.com/{output_filename}"
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})