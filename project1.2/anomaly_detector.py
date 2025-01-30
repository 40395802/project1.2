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
from typing import Optional, Tuple, Dict, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

# ------------------ 환경 설정 ------------------
mpl_use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'

# ------------------ 데이터 클래스 ------------------
@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    duration: int
    channels: int
    yamnet_sample_rate: int
    buffer_size: int

@dataclass(frozen=True)
class ModelConfig:
    contamination: str
    n_estimators: int
    threshold_percentile: int
    grid_search: bool
    update_interval: float

@dataclass(frozen=True)
class Config:
    audio: AudioConfig
    model: ModelConfig
    dataset_path: str
    model_path: str

# ------------------ 버퍼 클래스 ------------------
class RingBuffer:
    def __init__(self, max_size: int, frame_size: int):
        self.max_size = max_size
        self.frame_size = frame_size
        self.buffer = np.zeros((max_size, frame_size), dtype=np.float32)
        self.index = 0
        self.lock = Lock()

    def put(self, data: np.ndarray) -> None:
        """버퍼에 데이터 추가"""
        with self.lock:
            if data.size == self.frame_size:
                self.buffer[self.index % self.max_size] = data.ravel()
                self.index += 1

    def get(self) -> Optional[np.ndarray]:
        """최신 데이터 조회"""
        with self.lock:
            return self.buffer[(self.index - 1) % self.max_size] if self.index > 0 else None

# ------------------ 코어 로직 ------------------
class AccidentSoundAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_gpu()
        self.yamnet = self._load_yamnet()
        self.model: Optional[IsolationForest] = None
        self.threshold: Optional[float] = None
        self.audio_buffer = RingBuffer(
            max_size=config.audio.buffer_size,
            frame_size=int(config.audio.sample_rate * config.audio.duration)
        )
        self._init_model()

    def _init_gpu(self) -> None:
        """GPU 설정 초기화"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            self.logger.warning(f"GPU 초기화 실패: {e}")

    def _load_yamnet(self) -> tf.Module:
        """YAMNet 모델 로드"""
        return hub.load('https://tfhub.dev/google/yamnet/1')

    def _init_model(self) -> None:
        """모델 초기화"""
        self.model = IsolationForest(
            n_estimators=self.config.model.n_estimators,
            contamination=self.config.model.contamination,
            random_state=42
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """오디오 전처리 파이프라인"""
        audio = librosa.resample(
            audio.squeeze(), 
            orig_sr=sr, 
            target_sr=self.config.audio.yamnet_sample_rate
        )
        target_len = self.config.audio.yamnet_sample_rate * self.config.audio.duration
        audio = librosa.util.fix_length(audio, target_len)
        _, embeddings, _ = self.yamnet(audio)
        return tf.reduce_mean(embeddings, axis=0).numpy()

    def _load_audio_file(self, path: str) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드 및 검증"""
        sr = librosa.get_samplerate(path)
        if sr != self.config.audio.yamnet_sample_rate:
            raise ValueError(f"Invalid sample rate: {sr}Hz (expected {self.config.audio.yamnet_sample_rate}Hz)")
        
        duration = librosa.get_duration(filename=path)
        if abs(duration - self.config.audio.duration) > 0.5:
            raise ValueError(f"Invalid duration: {duration:.2f}s (expected {self.config.audio.duration}s)")
        
        return librosa.load(path, sr=self.config.audio.yamnet_sample_rate)

    def train(self) -> None:
        """모델 학습 실행"""
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.config.dataset_path}")

        X = []
        for fname in os.listdir(self.config.dataset_path):
            if not fname.lower().endswith(('.wav', '.mp3')):
                continue
            
            path = os.path.join(self.config.dataset_path, fname)
            try:
                audio, sr = self._load_audio_file(path)
                embedding = self.preprocess_audio(audio, sr)
                X.append(embedding)
                self.logger.info(f"Processed: {fname}")
            except Exception as e:
                self.logger.error(f"File error: {fname} - {str(e)}")
        
        if not X:
            raise ValueError("No valid training data")
        
        X = np.array(X)
        self.model.fit(X)
        self._set_threshold(X)
        self._save_model()

    def _set_threshold(self, X: np.ndarray) -> None:
        """이상치 임계값 설정"""
        scores = self.model.score_samples(X)
        self.threshold = np.percentile(scores, self.config.model.threshold_percentile)
        self.logger.info(f"Threshold set: {self.threshold:.2f}")

    def _save_model(self) -> None:
        """모델 저장"""
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'threshold': self.threshold}, self.config.model_path)
        self.logger.info(f"Model saved: {self.config.model_path}")

    def load_model(self) -> None:
        """모델 로드"""
        data = joblib.load(self.config.model_path)
        self.model = data['model']
        self.threshold = data['threshold']
        self.logger.info("Model loaded")

    def predict(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """실시간 예측 수행"""
        embedding = self.preprocess_audio(audio, sr)
        score = self.model.score_samples([embedding])[0]
        mel_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=128, 
                fmax=8000
            ), 
            ref=np.max
        )
        return {
            'is_accident': score < self.threshold,
            'score': score,
            'confidence': max(0, (self.threshold - score) / self.threshold),
            'waveform': audio,
            'mel_spectrogram': mel_spec
        }

# ------------------ 모니터링 클래스 ------------------
class AudioMonitor:
    def __init__(self, detector: AccidentSoundAnomalyDetector, config: Config):
        self.detector = detector
        self.audio_cfg = config.audio
        self.model_cfg = config.model
        self.is_running = False
        self.stream: Optional[sd.InputStream] = None
        self.active_figures = []
        self.max_figures = 5

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        """오디오 입력 콜백"""
        if status:
            logging.warning(f"Audio stream error: {status}")
        self.detector.audio_buffer.put(indata.copy())

    def start(self) -> None:
        """모니터링 시작"""
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.audio_cfg.sample_rate,
            channels=self.audio_cfg.channels,
            callback=self._callback,
            blocksize=int(self.audio_cfg.sample_rate * self.audio_cfg.duration)
        )
        self.stream.start()
        logging.info("Monitoring started (Ctrl+C to stop)")

        try:
            while self.is_running:
                self._process_audio()
        except KeyboardInterrupt:
            self.stop()

    def _process_audio(self) -> None:
        """오디오 데이터 처리 루프"""
        current_time = datetime.now().timestamp()
        if current_time - self._last_update >= self.model_cfg.update_interval:
            audio = self.detector.audio_buffer.get()
            if audio is not None:
                result = self.detector.predict(audio, self.audio_cfg.sample_rate)
                self._update_display(result)
            self._last_update = current_time

    def _update_display(self, result: Dict[str, Any]) -> None:
        """결과 표시 업데이트"""
        status = '🚨 사고' if result['is_accident'] else '✅ 정상'
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {status} | 신뢰도: {result['confidence']:.1%}"
        print(f"\r{log_msg}", end='', flush=True)
        
        if result['is_accident']:
            self._async_visualize(result['waveform'], result['mel_spectrogram'])

    def _async_visualize(self, waveform: np.ndarray, mel_spec: np.ndarray) -> None:
        """비동기 시각화"""
        def plot() -> None:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(np.linspace(0, len(waveform)/self.audio_cfg.sample_rate, len(waveform)), waveform)
            plt.title("Waveform")
            
            plt.subplot(2, 1, 2)
            librosa.display.specshow(mel_spec, sr=self.audio_cfg.sample_rate, cmap='inferno')
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel Spectrogram")
            
            plt.tight_layout()
            self._manage_figures(plt.gcf())
            plt.show(block=False)
            plt.pause(0.1)
        
        Thread(target=plot, daemon=True).start()

    def _manage_figures(self, new_fig: plt.Figure) -> None:
        """그래프 창 관리"""
        if len(self.active_figures) >= self.max_figures:
            old_fig = self.active_figures.pop(0)
            plt.close(old_fig)
        self.active_figures.append(new_fig)

    def stop(self) -> None:
        """모니터링 중지"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("\n")
        logging.info("Monitoring stopped")
        for fig in self.active_figures:
            plt.close(fig)