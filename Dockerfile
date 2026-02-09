# Official PyTorch base image for 2026 stability 
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Audio processing aur build dependencies install karein 
RUN apt-get update && apt-get install -y \
    ffmpeg git python3-dev build-essential \
    portaudio19-dev libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Fish-Speech repository clone karein
RUN git clone https://github.com/fishaudio/fish-speech.git .

# Requirements install karein [cite: 2]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# WEIGHT BAKING: Build ke waqt hi weights download karlein 
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fishaudio/openaudio-s1-mini', \
    local_dir='checkpoints/s1-mini', \
    token='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')"

# NLTK download karein
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Handler file copy karein
COPY handler.py .

CMD ["python3", "-u", "handler.py"]