FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 1. System dependencies aur compilers install karein
RUN apt-get update && apt-get install -y \
    ffmpeg git python3-dev build-essential \
    portaudio19-dev libasound2-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Fish-Speech repository clone karein
RUN git clone https://github.com/fishaudio/fish-speech.git .

# 3. Python packages install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Repository ko as a library install karein (Zaruri step)
RUN pip install -e .

# 5. Chunking ke liye NLTK data pehle se download karein
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 6. Handler copy karein
COPY handler.py .

ENV TORCH_COMPILE=1
CMD ["python", "-u", "handler.py"]