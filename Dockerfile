FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git python3-dev build-essential \
    portaudio19-dev libasound2-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Fish-Speech code clone karein
RUN git clone https://github.com/fishaudio/fish-speech.git .

# 3. Python packages install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# 4. WEIGHT BAKING: Build ke waqt hi weights download karein
# Note: Is step ke baad image 10GB+ ho jayegi
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='fishaudio/openaudio-s1-mini', \
    local_dir='checkpoints/s1-mini', \
    token='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')"

# 5. NLTK data download
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 6. Handler copy karein
COPY handler.py .

ENV TORCH_COMPILE=1
CMD ["python", "-u", "handler.py"]