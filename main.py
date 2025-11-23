# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import librosa
import soundfile as sf
import numpy as np
import io
import tempfile
import os
from pydantic import BaseModel
import jiwer
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import ffmpeg
import logging
import warnings
from tts_utils import tts_engine
import time

# Подавляем предупреждения о deprecated функциях
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели Whisper-Tiny с правильной конфигурацией
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Используем явную загрузку модели вместо pipeline для большего контроля
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Настройка generation config для Whisper
model.generation_config.language = "ru"  # или "en" если нужен английский
model.generation_config.task = "transcribe"  # "transcribe" или "translate"
model.generation_config.forced_decoder_ids = None  # Явно отключаем устаревший параметр

logger.info(f"Model loaded successfully: {model_name}")
logger.info(f"Generation config: language={model.generation_config.language}, task={model.generation_config.task}")

app = FastAPI(title="Speech-to-Text Processor", version="1.0.1")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    reference_text: str
    hypothesis_text: str

def convert_audio_to_wav(audio_data: bytes) -> bytes:
    """Конвертирует аудио в WAV формат с частотой 16kHz"""
    try:
        # Создаем временные файлы
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as temp_in:
            temp_in.write(audio_data)
            temp_in_path = temp_in.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
            temp_out_path = temp_out.name
        
        # Конвертируем в WAV с помощью ffmpeg
        ffmpeg.input(temp_in_path).output(
            temp_out_path,
            acodec='pcm_s16le',
            ac=1,
            ar='16000'
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        
        # Читаем результат
        with open(temp_out_path, 'rb') as f:
            wav_data = f.read()
        
        # Удаляем временные файлы
        os.unlink(temp_in_path)
        os.unlink(temp_out_path)
        
        return wav_data
    
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

def process_audio_file(file_content: bytes) -> np.ndarray:
    """Обрабатывает аудио файл и возвращает numpy массив"""
    try:
        # Конвертируем в WAV если нужно
        if not file_content.startswith(b'RIFF'):
            file_content = convert_audio_to_wav(file_content)
        
        # Загружаем аудио с помощью librosa
        audio_io = io.BytesIO(file_content)
        audio, sr = librosa.load(audio_io, sr=16000)
        
        logger.info(f"Audio loaded successfully. Shape: {audio.shape}, SR: {sr}")
        return audio
    
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

def transcribe_audio(audio_array: np.ndarray) -> str:
    """Транскрибация аудио с использованием Whisper"""
    try:
        # Подготовка входных данных
        input_features = processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Генерация с правильными параметрами
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                # language=model.generation_config.language,
                # task=model.generation_config.task,
                max_new_tokens=255,  # Ограничение длины генерации
                no_repeat_ngram_size=2,  # Предотвращение повторений
                early_stopping=True,
            )
        
        # Декодирование результата
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True,
            normalize=True  # Нормализация текста
        )[0]
        
        return transcription.strip()
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-file/")
async def transcribe_file(file: UploadFile = File(...)):
    """Транскрибация загруженного аудио файла"""
    try:
        logger.info(f"Processing file: {file.filename}")
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        audio = process_audio_file(file_content)
        
        # Транскрибация
        transcription = transcribe_audio(audio)
        
        if not transcription:
            transcription = "No speech detected in the audio."
        
        logger.info(f"Transcription successful: {transcription[:50]}...")
        return {"transcription": transcription}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/transcribe-microphone/")
async def transcribe_microphone(file: UploadFile = File(...)):
    """Транскрибация аудио с микрофона"""
    return await transcribe_file(file)

@app.post("/evaluate/")
async def evaluate_transcription(request: EvaluationRequest):
    """Оценка качества транскрибации"""
    try:
        reference = request.reference_text.strip().lower()
        hypothesis = request.hypothesis_text.strip().lower()
        
        if not reference or not hypothesis:
            raise HTTPException(status_code=400, detail="Both reference and hypothesis text are required")
        
        # Вычисление WER (Word Error Rate)
        wer = jiwer.wer(reference, hypothesis)
        
        # Вычисление CER (Character Error Rate)
        cer = jiwer.cer(reference, hypothesis)
        
        # Вычисление BLEU
        reference_tokens = [reference.split()]
        hypothesis_tokens = hypothesis.split()
        
        if len(hypothesis_tokens) == 0:
            bleu = 0.0
        else:
            bleu = sentence_bleu(reference_tokens, hypothesis_tokens)
        
        # Анализ результатов
        analysis = {
            "wer_interpretation": "Отлично" if wer < 0.1 else "Хорошо" if wer < 0.3 else "Удовлетворительно" if wer < 0.5 else "Плохо",
            "cer_interpretation": "Отлично" if cer < 0.05 else "Хорошо" if cer < 0.15 else "Удовлетворительно" if cer < 0.3 else "Плохо",
            "bleu_interpretation": "Отлично" if bleu > 0.7 else "Хорошо" if bleu > 0.5 else "Удовлетворительно" if bleu > 0.3 else "Плохо"
        }
        
        logger.info(f"Evaluation completed: WER={wer:.4f}, CER={cer:.4f}, BLEU={bleu:.4f}")
        
        return {
            "wer": wer,
            "cer": cer,
            "bleu": bleu,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/health/")
async def health_check():
    """Проверка состояния сервера"""
    return {
        "status": "healthy",
        "model": "whisper-tiny",
        "device": device,
        "language": model.generation_config.language,
        "task": model.generation_config.task,
        "model_version": model.config._commit_hash if hasattr(model.config, '_commit_hash') else "unknown"
    }



class TTSRequest(BaseModel):
    text: str
    speaker: str = "aidar"  # aidar, baya, kseniya, xenia, eugene, random

@app.post("/synthesize-speech/")
async def synthesize_speech(request: TTSRequest):
    """Синтез речи из текста"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Пустой текст для озвучки")
        
        # Ограничение длины текста
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Текст слишком длинный (максимум 1000 символов)")
        
        # Генерация аудио
        audio_data = tts_engine.synthesize(request.text, request.speaker)
        
        # Возврат аудио файла
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{int(time.time())}.wav"
            }
        )
    
    except Exception as e:
        logger.error(f"TTS synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")