import logging
import sys
import yaml
import os
from typing import Optional
from anomaly_detector import Config, AudioConfig, ModelConfig, AccidentSoundAnomalyDetector, AudioMonitor

def load_config(config_path: str = "config.yaml") -> Config:
    """YAML 설정 파일을 로드하고 Config 객체를 반환합니다."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return Config(
            audio=AudioConfig(**cfg['audio']),
            model=ModelConfig(**cfg['model']),
            dataset_path=cfg['dataset_path'],
            model_path=cfg['model_path']
        )
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logging.error(f"설정 파일 로드 실패: {e}")
        raise

def handle_exception(exc_type, exc_value, exc_traceback):
    """전역 예외 핸들러"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("예외 발생", exc_info=(exc_type, exc_value, exc_traceback))
    print(f"\n*** 오류 발생 ***\n유형: {exc_type.__name__}\n내용: {exc_value}")

def configure_logging(log_file: str = "debug.log") -> None:
    """로깅 설정을 초기화합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

def main():
    sys.excepthook = handle_exception
    configure_logging()
    
    try:
        cfg = load_config()
        detector = AccidentSoundAnomalyDetector(cfg)
        
        if not os.path.exists(cfg.model_path):
            logging.info("모델 학습 시작...")
            detector.train()
        else:
            logging.info("기존 모델 로드 중...")
            detector.load_model()
        
        monitor = AudioMonitor(detector, cfg)
        monitor.start()
    except Exception as e:
        logging.critical(f"치명적 오류로 종료: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()