import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa
import os
import logging
import joblib
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from threading import Thread, Lock
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

# --------------- 라즈베리파이 GUI 설정 ---------------
mpl_use('TkAgg')  # RaspberryPi에서 그래프 표시 필수
matplotlib.rcParams['toolbar'] = 'None'  # 툴바 숨기기

# ------------------ 설정 클래스 ------------------
@dataclass
class AudioConfig:
    sample_rate: int
    duration: int
    channels: int
    yamnet_sample_rate: int
    buffer_size: int

@dataclass
class ModelConfig:
    contamination: str
    n_estimators: int
    threshold_percentile: int
    grid_search: bool
    update_interval: float

@dataclass
class Config:
    audio: AudioConfig
    model: ModelConfig
    dataset_path: str
    model_path: str

# ------------------ 오디오 버퍼 ------------------
class RingBuffer:
    def __init__(self, max_size: int, frame_size: int):
        self.max_size = max_size
        self.frame_size = frame_size
        self.buffer = np.zeros((max_size, frame_size), dtype=np.float32)
        self.index = 0
        self.lock = Lock()

    def put(self, data: np.ndarray):
        flattened = data.squeeze()
        with self.lock:
            if len(flattened) == self.frame_size:
                self.buffer[self.index % self.max_size] = flattened
                self.index += 1

    def get(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.index == 0:
                return None
            idx = (self.index - 1) % self.max_size
            return self.buffer[idx]

# ------------------ 코어 로직 ------------------
class AccidentSoundAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self._init_gpu()
        self._load_yamnet()
        self._init_model()
        self.audio_buffer = RingBuffer(
            max_size=config.audio.buffer_size,
            frame_size=int(config.audio.sample_rate * config.audio.duration)
        )
        self.logger = self._configure_logger()

    def _init_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                self.logger.warning(f"GPU 설정 실패: {e}")

    def _load_yamnet(self):
        self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

    def _init_model(self):
        self.model = IsolationForest(
            n_estimators=self.config.model.n_estimators,
            contamination=self.config.model.contamination,
            random_state=42
        )
        self.threshold = None

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.squeeze()
        target_len = self.config.audio.yamnet_sample_rate * self.config.audio.duration
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        if sr != self.config.audio.yamnet_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.audio.yamnet_sample_rate)
        _, embeddings, _ = self.yamnet(audio)
        return tf.reduce_mean(embeddings, axis=0).numpy()

    def train(self):
        X = []
        valid_files = 0
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"데이터셋 경로 없음: {self.config.dataset_path}")
        for fname in os.listdir(self.config.dataset_path):
            if not fname.lower().endswith(('.wav', '.mp3')):
                continue
            path = os.path.join(self.config.dataset_path, fname)
            try:
                self._validate_audio(path)
                audio, sr = librosa.load(path, sr=self.config.audio.yamnet_sample_rate)
                embedding = self.preprocess_audio(audio, sr)
                X.append(embedding)
                valid_files += 1
                self.logger.info(f"학습 데이터 처리: {fname}")
            except Exception as e:
                self.logger.error(f"파일 실패: {fname} - {e}")
        if valid_files == 0:
            raise ValueError("학습 데이터 없음")
        X = np.array(X)
        self.model.fit(X)
        self._set_threshold(X)
        self._save_model()

    def _validate_audio(self, path: str):
        sr = librosa.get_samplerate(path)
        if sr != self.config.audio.yamnet_sample_rate:
            raise ValueError(f"잘못된 샘플링 레이트: {sr}Hz")
        duration = librosa.get_duration(filename=path)
        if abs(duration - self.config.audio.duration) > 0.5:
            raise ValueError(f"잘못된 길이: {duration:.2f}초")

    def _set_threshold(self, X: np.ndarray):
        scores = self.model.score_samples(X)
        self.threshold = np.percentile(scores, self.config.model.threshold_percentile)
        self.logger.info(f"임계값 설정: {self.threshold:.2f}")

    def _save_model(self):
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'threshold': self.threshold}, self.config.model_path)
        self.logger.info(f"모델 저장 완료: {self.config.model_path}")

    def load_model(self):
        data = joblib.load(self.config.model_path)
        self.model = data['model']
        self.threshold = data['threshold']
        self.logger.info("모델 로드 완료")

    def compute_log_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        return librosa.power_to_db(S, ref=np.max)

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        embedding = self.preprocess_audio(audio, sr)
        score = self.model.score_samples([embedding])[0]
        mel_spec = self.compute_log_mel_spectrogram(audio, sr)
        return {
            'is_accident': score < self.threshold,
            'score': score,
            'confidence': max(0, (self.threshold - score) / self.threshold),
            'waveform': audio,
            'mel_spectrogram': mel_spec
        }

# ------------------ 모니터링 ------------------
class AudioMonitor:
    def __init__(self, detector: AccidentSoundAnomalyDetector, config: Config):
        self.detector = detector
        self.config = config.audio
        self.model_config = config.model
        self.is_running = False
        self.stream = None
        self.last_status = None
        self.last_update_time = 0
        self.active_figures = []  # 활성 그래프 창 관리
        self.max_figures = 5      # 최대 동시 표시 창 수

    def _callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"오디오 스트림 오류: {status}")
        self.detector.audio_buffer.put(indata.copy().squeeze())

    def start(self):
        import time
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            callback=self._callback,
            blocksize=int(self.config.sample_rate * self.config.duration)
        )
        self.stream.start()
        logging.info("모니터링 시작... (Ctrl+C 종료)")

        try:
            while self.is_running:
                current_time = time.time()
                if current_time - self.last_update_time >= self.model_config.update_interval:
                    audio = self.detector.audio_buffer.get()
                    if audio is not None:
                        result = self.detector.predict(audio, self.config.sample_rate)
                        current_status = '🚨 사고' if result['is_accident'] else '✅ 정상'
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        log = f"\r[{timestamp}] {current_status} | 신뢰도: {result['confidence']:.1%}"
                        print(log, end='', flush=True)
                        if result['is_accident']:
                            self._async_visualize(
                                result['waveform'], 
                                result['mel_spectrogram']
                            )
                    self.last_update_time = current_time
                time.sleep(0.001)
        except KeyboardInterrupt:
            self.stop()

    def _async_visualize(self, waveform: np.ndarray, mel_spec: np.ndarray):
        def plot():
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(np.linspace(0, len(waveform)/self.config.sample_rate, len(waveform)), waveform)
            plt.title("Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            
            plt.subplot(2, 1, 2)
            librosa.display.specshow(mel_spec, 
                                   sr=self.config.sample_rate, 
                                   x_axis='time', 
                                   y_axis='mel', 
                                   cmap='inferno')
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel Spectrogram")
            plt.tight_layout()
            
            # 최대 창 수 관리
            if len(self.active_figures) >= self.max_figures:
                oldest_fig = self.active_figures.pop(0)
                plt.close(oldest_fig)
            self.active_figures.append(plt.gcf())
            
            plt.show(block=False)
            plt.pause(0.1)
        
        Thread(target=plot, daemon=True).start()

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("\n")
        logging.info("모니터링 중지")
        for fig in self.active_figures:
            plt.close(fig)