import torch
import numpy as np
from typing import Optional, List, Union, Dict, Callable
import threading
import time
import os
import logging

from .model import GigaAMASR
from .preprocess import FeatureExtractor
from .utils import format_time
try:
    from .vad_utils import get_pipeline
    VAD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    VAD_AVAILABLE = False


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
        Порог энергии для определения наличия речи (используется при отключенном VAD).
    min_silence_duration : float
        Минимальная продолжительность тишины в секундах для завершения сегмента.
    stabilization_frames : int
        Количество фреймов для стабилизации результата.
    use_vad : bool
        Использовать VAD для обнаружения речи.
    vad_threshold : float
        Порог для определения наличия речи с использованием VAD.
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
        use_vad: bool = True,
        vad_threshold: float = 0.5,
        callback: Optional[Callable] = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_buffer_samples = buffer_size * sample_rate
        self.threshold = threshold
        self.min_silence_samples = int(min_silence_duration * sample_rate)
        self.stabilization_frames = stabilization_frames
        self.vad_threshold = vad_threshold
        
        # Проверка и настройка VAD
        self.use_vad = use_vad and VAD_AVAILABLE
        if self.use_vad:
            if "HF_TOKEN" not in os.environ:
                import warnings
                warnings.warn("HF_TOKEN не найден в переменных окружения. VAD будет отключен. Используйте os.environ['HF_TOKEN'] = '...'")
                self.use_vad = False
            else:
                try:
                    self.device = next(self.model.parameters()).device
                    self.vad = get_pipeline(self.device)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Не удалось загрузить VAD модель: {e}. VAD будет отключен.")
                    self.use_vad = False
        
        # Буфер для хранения аудиоданных
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Состояние обработки
        self.is_speaking = False
        self.silence_counter = 0
        self.speaking_buffer = []
        self.last_end_time = 0
        
        # Буфер для VAD
        self.vad_buffer = []
        self.vad_buffer_size = 3 * sample_rate  # 3 секунды для VAD буфера
        
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
    
    def _detect_speech_energy(self, audio: torch.Tensor) -> bool:
        """
        Определяет наличие речи в аудиосегменте на основе энергии.
        
        Parameters
        ----------
        audio : torch.Tensor
            Аудиоданные для анализа.
            
        Returns
        -------
        bool
            True, если обнаружена речь, иначе False.
        """
        # Простой детектор речи на основе энергии
        energy = (audio ** 2).mean().item()
        return energy > self.threshold
    
    def _detect_speech_vad(self, audio: torch.Tensor) -> bool:
        """
        Определяет наличие речи с использованием VAD.
        
        Parameters
        ----------
        audio : torch.Tensor
            Аудиоданные для анализа.
            
        Returns
        -------
        bool
            True, если обнаружена речь, иначе False.
        """
        if not self.use_vad:
            return self._detect_speech_energy(audio)
        
        import logging
        logging.info(f"VAD: Добавление {audio.shape[0]} сэмплов в буфер")
        
        # Добавляем данные в VAD буфер
        self.vad_buffer.append(audio)
        
        # Если накоплено достаточно данных, анализируем
        vad_audio_length = sum(chunk.shape[0] for chunk in self.vad_buffer)
        logging.info(f"VAD: Размер буфера {vad_audio_length}/{self.vad_buffer_size} сэмплов")
        
        if vad_audio_length < self.vad_buffer_size:
            logging.info("VAD: Недостаточно данных для анализа")
            return False  # Недостаточно данных для анализа
        
        # Объединяем данные для анализа
        vad_audio = torch.cat(self.vad_buffer)
        
        # Если буфер стал слишком большим, обрезаем его
        if vad_audio.shape[0] > self.vad_buffer_size:
            vad_audio = vad_audio[-self.vad_buffer_size:]
            self.vad_buffer = [vad_audio]
        
        # Анализируем с помощью VAD
        try:
            import io
            from pydub import AudioSegment
            
            # Преобразуем тензор в аудиофайл в памяти
            audio_bytes = (vad_audio * 32767).to(torch.int16).cpu().numpy().tobytes()
            audio_segment = AudioSegment(
                audio_bytes,
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            
            # Сохраняем во временный поток
            audio_io = io.BytesIO()
            audio_segment.export(audio_io, format="wav")
            audio_io.seek(0)
            
            # Анализируем с помощью VAD
            vad_result = self.vad({"uri": "stream", "audio": audio_io})
            
            # Проверяем наличие речи в последнем сегменте
            has_speech = False
            for segment in vad_result.get_timeline().support():
                # Проверяем, что сегмент относится к последней части аудио
                if segment.end > (vad_audio.shape[0] / self.sample_rate - 0.5):
                    speech_proba = getattr(segment, 'score', 1.0)
                    has_speech = speech_proba > self.vad_threshold
                    break
            
            return has_speech
        except Exception as e:
            logging.warning(f"Ошибка при использовании VAD: {e}. Переключение на определение по энергии.")
            return self._detect_speech_energy(audio)
    
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
        if self.use_vad:
            return self._detect_speech_vad(audio)
        else:
            return self._detect_speech_energy(audio)
    
    def _process_stream(self) -> None:
        """
        Обрабатывает аудиопоток в отдельном потоке.
        """
        while self.running:
            with self.buffer_lock:
                if not self.audio_buffer:
                    time.sleep(0.01)
                    continue
                
                # Логируем состояние буфера
                buffer_size = sum(chunk.shape[0] for chunk in self.audio_buffer)
                logging.info(f"Обработка аудиобуфера: {len(self.audio_buffer)} чанков, {buffer_size} сэмплов")
                
                # Обработка накопленных аудиоданных
                audio_chunk = torch.cat(self.audio_buffer)
                self.audio_buffer = []
            
            # Обрабатываем аудиочанк порциями
            for i in range(0, audio_chunk.shape[0], self.chunk_size):
                if not self.running:
                    break
                
                # Получаем текущий подчанк
                end_idx = min(i + self.chunk_size, audio_chunk.shape[0])
                subchunk = audio_chunk[i:end_idx]
                
                # Логируем данные о подчанке
                energy = (subchunk ** 2).mean().item()
                logging.info(f"Обработка подчанка: {subchunk.shape[0]} сэмплов, энергия: {energy:.6f}, порог: {self.threshold:.6f}")
                
                # Обнаружение речи
                is_speech = self._detect_speech(subchunk)
                logging.info(f"Обнаружение речи: {is_speech}, использование VAD: {self.use_vad}")
                
                if is_speech:
                    # Если это начало речи, начинаем новый сегмент
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speaking_buffer = [subchunk]
                    else:
                        self.speaking_buffer.append(subchunk)
                    
                    # Сбрасываем счетчик тишины
                    self.silence_counter = 0
                    
                    # Выполняем промежуточное распознавание каждые N фреймов
                    if len(self.speaking_buffer) % self.stabilization_frames == 0:
                        self._perform_interim_recognition()
                else:
                    # Если была речь, увеличиваем счетчик тишины
                    if self.is_speaking:
                        self.speaking_buffer.append(subchunk)
                        self.silence_counter += subchunk.shape[0]
                        
                        # Если тишина достаточно длинная, завершаем сегмент
                        if self.silence_counter >= self.min_silence_samples:
                            self._finalize_speech_segment()
            
            # Пауза перед следующей итерацией
            time.sleep(0.01)
    
    def _finalize_speech_segment(self) -> None:
        """
        Завершает обработку сегмента речи.
        """
        # Объединяем буфер речи
        speech_segment = torch.cat(self.speaking_buffer)
        
        # Распознаем речь
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
        # Очищаем VAD буфер
        self.vad_buffer = []
    
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
        features, feature_lengths = self.feature_extractor(
            audio.unsqueeze(0), 
            torch.tensor([audio.shape[0]]).to(device)
        )
        
        # Кодирование
        encoded, encoded_len = self.model.encoder(features, feature_lengths)
        
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
        self.vad_buffer = []
    
    def stop(self) -> None:
        """
        Останавливает обработку потока.
        """
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0) 