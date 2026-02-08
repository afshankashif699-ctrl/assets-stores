# Official Coqui base image jo GPU support ke liye best hai
FROM ghcr.io/coqui-ai/tts:latest

WORKDIR /app

# Audio processing ke liye ffmpeg install karna lazmi hai
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsox-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Requirements install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK download karein taake 100k characters ko sentences mein tora ja sakay
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Model weights ko pehle hi download karlein taake RunPod par error na aaye
RUN python3 -c 'from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2")'

# Handler file copy karein
COPY handler.py .

# RunPod ke liye unbuffered output mode
CMD ["python3", "-u", "handler.py"]