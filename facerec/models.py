from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_BUFFALO_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

DETECTOR_URL = f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"

DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "w600k_r50.onnx"


def missing_models() -> list[Path]:
    return [p for p in (DETECTOR_PATH, RECOGNIZER_PATH) if not p.exists()]


def _download(url: str, dest: Path) -> None:
    try:
        urllib.request.urlretrieve(url, dest)
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download of {dest.name} failed ({exc}). Check your connection and rerun."
        ) from exc


def fetch_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not DETECTOR_PATH.exists():
        print(f"  {DETECTOR_PATH.name}: downloading ...")
        _download(DETECTOR_URL, DETECTOR_PATH)
        print(f"  {DETECTOR_PATH.name}: done ({DETECTOR_PATH.stat().st_size / 1e6:.1f} MB)")
    if not RECOGNIZER_PATH.exists():
        zip_path = MODELS_DIR / "buffalo_l.zip"
        print(f"  {RECOGNIZER_PATH.name}: downloading buffalo_l.zip (~340 MB) ...")
        try:
            _download(_BUFFALO_ZIP, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                with archive.open(RECOGNIZER_PATH.name) as src, open(RECOGNIZER_PATH, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            RECOGNIZER_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download of {RECOGNIZER_PATH.name} failed ({exc}). "
                "Check your connection and rerun."
            ) from exc
        finally:
            zip_path.unlink(missing_ok=True)
        print(f"  {RECOGNIZER_PATH.name}: done ({RECOGNIZER_PATH.stat().st_size / 1e6:.1f} MB)")


def check_models() -> None:
    if missing := missing_models():
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"Missing model file(s): {names}. Run `python3 app.py` to download them.")
