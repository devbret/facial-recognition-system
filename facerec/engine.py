from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from facerec.models import DETECTOR_PATH, RECOGNIZER_PATH, check_models

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

DEFAULT_THRESHOLD = 0.363
MAX_DETECT_SIZE = 1024


@dataclass
class Face:
    box: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


class FaceEngine:
    def __init__(self, score_threshold: float = 0.8):
        check_models()
        self._detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), score_threshold, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")

    def extract_faces(self, image: np.ndarray) -> list[Face]:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        faces = self._extract(image)
        if not faces:
            pad = int(0.4 * max(image.shape[:2]))
            padded = cv2.copyMakeBorder(
                image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            img_h, img_w = image.shape[:2]
            for face in self._extract(padded):
                x, y, w, h = face.box
                x, y = max(0, x - pad), max(0, y - pad)
                w, h = min(w, img_w - x), min(h, img_h - y)
                if w > 0 and h > 0:
                    faces.append(Face(box=(x, y, w, h), score=face.score, embedding=face.embedding))
        return faces

    def _extract(self, image: np.ndarray) -> list[Face]:
        scale = min(1.0, MAX_DETECT_SIZE / max(image.shape[:2]))
        det_img = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )
        h, w = det_img.shape[:2]
        self._detector.setInputSize((w, h))
        _, rows = self._detector.detect(det_img)
        if rows is None:
            return []

        img_h, img_w = image.shape[:2]
        faces = []
        for row in rows:
            aligned = self._recognizer.alignCrop(det_img, row)
            feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
            feat /= np.linalg.norm(feat) + 1e-10
            x, y, bw, bh = (row[:4] / scale).round().astype(int)
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, img_w - x), min(bh, img_h - y)
            faces.append(Face(box=(x, y, bw, bh), score=float(row[-1]), embedding=feat))
        return faces


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
