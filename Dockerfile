# Official Coqui base image
FROM ghcr.io/coqui-ai/tts:latest

WORKDIR /app

# CPML License ko automated tareeqe se agree karne ke liye
ENV COQUI_TOS_AGREED=1
ENV DEBIAN_FRONTEND=noninteractive

# Audio processing dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsox-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Requirements install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK download karein
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Model weights download (Ab license error nahi aayega)
RUN python3 -c 'from TTS.api import TTS; TTS("tts_models/multilingual/multi-dataset/xtts_v2")'

# Handler copy karein
COPY handler.py .

CMD ["python3", "-u", "handler.py"]