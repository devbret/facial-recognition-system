from __future__ import annotations

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def missing_models() -> list[Path]:
    return [p for p in (DETECTOR_PATH, RECOGNIZER_PATH) if not p.exists()]


def fetch_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            continue
        print(f"  {filename}: downloading ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except OSError as exc:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Download of {filename} failed ({exc}). Check your connection and rerun.") from exc
        print(f"  {filename}: done ({dest.stat().st_size / 1e6:.1f} MB)")


def check_models() -> None:
    if missing := missing_models():
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"Missing model file(s): {names}. Run `python3 app.py` to download them.")
