import logging
import sys
import yaml
import os
from anomaly_detector import Config, AudioConfig, ModelConfig, AccidentSoundAnomalyDetector, AudioMonitor

def load_config() -> Config:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    return Config(
        audio=AudioConfig(**cfg['audio']),
        model=ModelConfig(**cfg['model']),
        dataset_path=cfg['dataset_path'],
        model_path=cfg['model_path']
    )

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("예외 발생", exc_info=(exc_type, exc_value, exc_traceback))
    print(f"\n*** 오류 발생 ***")
    print(f"유형: {exc_type.__name__}")
    print(f"내용: {exc_value}")

sys.excepthook = handle_exception

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("debug.log")
        ]
    )
    
    cfg = load_config()
    
    detector = AccidentSoundAnomalyDetector(cfg)
    if not os.path.exists(cfg.model_path):
        logging.info("모델 학습 시작...")
        detector.train()
    else:
        logging.info("기존 모델 로드 중...")
        detector.load_model()

    monitor = AudioMonitor(detector, cfg)
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()

if __name__ == "__main__":
    main()