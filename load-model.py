from transformers import pipeline
import torch

# Модель будет автоматически скачана и кэширована
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0 if torch.cuda.is_available() else -1
)