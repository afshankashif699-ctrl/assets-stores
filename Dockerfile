FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Basic tools install karein
RUN apt-get update && apt-get install -y \
    ffmpeg git python3-dev build-essential \
    portaudio19-dev libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Code clone aur install karein
RUN git clone https://github.com/fishaudio/fish-speech.git .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# NLTK download
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY handler.py .
CMD ["python3", "-u", "handler.py"]