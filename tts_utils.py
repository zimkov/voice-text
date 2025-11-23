# backend/tts_utils.py
import torch
import soundfile as sf
import numpy as np
import os
import tempfile
from typing import Optional

class TTSEngine:
    def __init__(self):
        self.device = torch.device('cpu')  # Silero работает на CPU
        self.model = None
        self.sample_rate = 48000  # 24kHz
        
        # Загрузка модели при инициализации
        self.load_model()
    
    def load_model(self):
        """Загружает предобученную модель Silero TTS"""
        try:
            language = 'ru'
            model_id = 'v4_ru'
            processor = 'cpu'
            
            # Загрузка модели с Hugging Face
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=model_id
            )
            
            self.model.to(self.device)
            print("✓ Silero TTS модель успешно загружена")
            
        except Exception as e:
            print(f"Ошибка загрузки TTS модели: {e}")
            raise
    
    def synthesize(self, text: str, speaker: str = 'aidar') -> bytes:
        """
        Синтезирует речь из текста
        
        Args:
            text (str): Текст для озвучки
            speaker (str): Имя диктора ('aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random')
        
        Returns:
            bytes: Аудио данные в формате WAV
        """
        if not self.model:
            self.load_model()
        
        try:
            # Генерация аудио
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate
            )
            
            # Преобразование в numpy массив
            audio_np = audio.cpu().numpy()
            
            # Создание временного файла
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_np, self.sample_rate, format='WAV')
                tmp_file_path = tmp_file.name
            
            # Чтение аудио файла
            with open(tmp_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Удаление временного файла
            os.unlink(tmp_file_path)
            
            return audio_data
            
        except Exception as e:
            print(f"Ошибка синтеза речи: {e}")
            raise

# Глобальный экземпляр TTS движка
tts_engine = TTSEngine()