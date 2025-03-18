import torch
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Callable
from collections import deque
import threading
import time

from .model import GigaAMASR
from .preprocess import FeatureExtractor
from .utils import format_time


class AudioStream:
    """
    Класс для обработки потокового аудио в реальном времени.
    
    Parameters
    ----------
    model : GigaAMASR
        Модель ASR для распознавания речи.
    sample_rate : int
        Частота дискретизации аудиопотока (по умолчанию 16000 Гц).
    chunk_size : int
        Размер обрабатываемого аудиочанка в сэмплах.
    buffer_size : int
        Максимальный размер буфера в секундах.
    threshold : float
        Порог энергии для определения наличия речи.
    min_silence_duration : float
        Минимальная продолжительность тишины в секундах для завершения сегмента.
    stabilization_frames : int
        Количество фреймов для стабилизации результата.
    callback : Optional[Callable]
        Функция обратного вызова для получения результатов распознавания.
    """
    
    def __init__(
        self,
        model: GigaAMASR,
        sample_rate: int = 16000,
        chunk_size: int = 4000,
        buffer_size: int = 30,
        threshold: float = 0.01,
        min_silence_duration: float = 0.8,
        stabilization_frames: int = 5,
        callback: Optional[Callable] = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_buffer_samples = buffer_size * sample_rate
        self.threshold = threshold
        self.min_silence_samples = int(min_silence_duration * sample_rate)
        self.stabilization_frames = stabilization_frames
        
        # Буфер для хранения аудиоданных
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Состояние обработки
        self.is_speaking = False
        self.silence_counter = 0
        self.speaking_buffer = []
        self.last_end_time = 0
        
        # Результаты распознавания
        self.transcription_buffer = []
        self.interim_results = []
        self.final_results = []
        
        # Экстрактор признаков
        self.feature_extractor = FeatureExtractor(sample_rate, self.model.cfg.encoder.feat_in)
        
        # Обратный вызов для получения результатов
        self.callback = callback
        
        # Запуск фонового процесса обработки
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def add_audio_chunk(self, audio_chunk: Union[np.ndarray, torch.Tensor, List[float]]) -> None:
        """
        Добавляет новый аудиочанк в буфер.
        
        Parameters
        ----------
        audio_chunk : Union[np.ndarray, torch.Tensor, List[float]]
            Аудиоданные для добавления в буфер.
        """
        if isinstance(audio_chunk, list):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk).float()
        
        # Нормализация амплитуды, если необходимо
        if audio_chunk.abs().max() > 1.0:
            audio_chunk = audio_chunk / audio_chunk.abs().max()
        
        with self.buffer_lock:
            self.audio_buffer.append(audio_chunk)
            
            # Обрезаем буфер, если он превышает максимальный размер
            total_samples = sum(chunk.shape[0] for chunk in self.audio_buffer)
            while total_samples > self.max_buffer_samples:
                removed = self.audio_buffer.pop(0)
                total_samples -= removed.shape[0]
    
    def _detect_speech(self, audio: torch.Tensor) -> bool:
        """
        Определяет наличие речи в аудиосегменте.
        
        Parameters
        ----------
        audio : torch.Tensor
            Аудиоданные для анализа.
            
        Returns
        -------
        bool
            True, если обнаружена речь, иначе False.
        """
        energy = (audio ** 2).mean().item()
        return energy > self.threshold
    
    def _process_stream(self) -> None:
        """
        Фоновый процесс для обработки аудиопотока.
        """
        while self.running:
            # Копируем текущий буфер для обработки
            with self.buffer_lock:
                if not self.audio_buffer:
                    time.sleep(0.01)  # Небольшая задержка, если буфер пуст
                    continue
                
                buffer_copy = self.audio_buffer.copy()
                audio_segment = torch.cat(buffer_copy)
            
            # Обнаружение речи
            is_speech = self._detect_speech(audio_segment)
            
            # Логика обработки речи
            if is_speech and not self.is_speaking:
                # Начало речи
                self.is_speaking = True
                self.speaking_buffer = [audio_segment]
                self.silence_counter = 0
                
            elif is_speech and self.is_speaking:
                # Продолжение речи
                self.speaking_buffer.append(audio_segment)
                self.silence_counter = 0
                
            elif not is_speech and self.is_speaking:
                # Возможное завершение речи
                self.silence_counter += audio_segment.shape[0]
                
                if self.silence_counter >= self.min_silence_samples:
                    # Завершение сегмента речи
                    self._finalize_speech_segment()
                else:
                    # Добавляем тишину в буфер речи, чтобы сохранить непрерывность
                    self.speaking_buffer.append(audio_segment)
            
            # Если речь активна, выполняем распознавание на накопленных данных
            if self.is_speaking and len(self.speaking_buffer) >= self.stabilization_frames:
                self._perform_interim_recognition()
            
            time.sleep(0.01)  # Предотвращение высокой загрузки CPU
    
    def _finalize_speech_segment(self) -> None:
        """
        Завершает обработку сегмента речи и выполняет финальное распознавание.
        """
        if not self.speaking_buffer:
            self.is_speaking = False
            self.silence_counter = 0
            return
        
        # Объединяем буфер речи
        speech_segment = torch.cat(self.speaking_buffer)
        
        # Получаем окончательный результат распознавания
        transcript = self._recognize_audio(speech_segment)
        
        # Расчет временных меток
        current_time = time.time()
        duration = speech_segment.shape[0] / self.sample_rate
        start_time = current_time - duration
        
        # Создаем структуру результата
        result = {
            "text": transcript,
            "start_time": format_time(start_time),
            "end_time": format_time(current_time),
            "duration": format_time(duration),
            "is_final": True
        }
        
        # Добавляем результат в буфер финальных результатов
        self.final_results.append(result)
        
        # Вызываем callback, если он задан
        if self.callback:
            self.callback(result)
        
        # Сбрасываем состояние
        self.is_speaking = False
        self.silence_counter = 0
        self.speaking_buffer = []
        self.last_end_time = current_time
    
    def _perform_interim_recognition(self) -> None:
        """
        Выполняет промежуточное распознавание на текущем буфере речи.
        """
        # Объединяем текущий буфер речи
        speech_segment = torch.cat(self.speaking_buffer)
        
        # Получаем промежуточный результат распознавания
        transcript = self._recognize_audio(speech_segment)
        
        # Расчет временных меток
        current_time = time.time()
        duration = speech_segment.shape[0] / self.sample_rate
        start_time = current_time - duration
        
        # Создаем структуру результата
        result = {
            "text": transcript,
            "start_time": format_time(start_time),
            "end_time": format_time(current_time),
            "duration": format_time(duration),
            "is_final": False
        }
        
        # Добавляем результат в буфер промежуточных результатов
        self.interim_results.append(result)
        
        # Вызываем callback, если он задан
        if self.callback:
            self.callback(result)
    
    def _recognize_audio(self, audio: torch.Tensor) -> str:
        """
        Распознает аудио с помощью модели ASR.
        
        Parameters
        ----------
        audio : torch.Tensor
            Аудиоданные для распознавания.
            
        Returns
        -------
        str
            Распознанный текст.
        """
        device = next(self.model.parameters()).device
        audio = audio.to(device)
        
        # Предобработка аудио
        features = self.feature_extractor(
            audio.unsqueeze(0), 
            torch.tensor([audio.shape[0]]).to(device)
        )
        
        # Кодирование
        encoded, encoded_len = self.model.encoder(features, torch.tensor([features.shape[2]]).to(device))
        
        # Декодирование в зависимости от типа модели
        if hasattr(self.model, "rnnt_decoder"):
            transcription = self.model.rnnt_decoder.decode(self.model.head, encoded, encoded_len)[0]
        else:
            transcription = self.model.ctc_decoder.decode(self.model.head, encoded, encoded_len)[0]
        
        return transcription
    
    def get_final_results(self) -> List[Dict]:
        """
        Возвращает список финальных результатов распознавания.
        
        Returns
        -------
        List[Dict]
            Список распознанных сегментов.
        """
        return self.final_results
    
    def get_interim_results(self) -> List[Dict]:
        """
        Возвращает список промежуточных результатов распознавания.
        
        Returns
        -------
        List[Dict]
            Список промежуточных результатов.
        """
        return self.interim_results
    
    def reset(self) -> None:
        """
        Сбрасывает состояние стрима.
        """
        with self.buffer_lock:
            self.audio_buffer = []
        
        self.is_speaking = False
        self.silence_counter = 0
        self.speaking_buffer = []
        self.interim_results = []
    
    def stop(self) -> None:
        """
        Останавливает обработку потока.
        """
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0) 