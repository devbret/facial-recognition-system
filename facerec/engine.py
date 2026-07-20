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

DEFAULT_THRESHOLD = 0.25
DEFAULT_PROBABLE_THRESHOLD = 0.18
DEFAULT_SCORE_THRESHOLD = 0.6
EDGE_SCORE_THRESHOLD = 0.15
MAX_DETECT_SIZE = 1024
LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class Face:
    box: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray
    landmarks: dict[str, tuple[float, float]]

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


def equalize_lighting(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceEngine:
    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        check_models()
        self._score_threshold = score_threshold
        self._detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), score_threshold, 0.3, 5000
        )
        self._recognizer = cv2.dnn.readNet(str(RECOGNIZER_PATH))

    def extract_faces(self, image: np.ndarray) -> list[Face]:
        candidates = (image, equalize_lighting(image))
        for candidate in candidates:
            if faces := self._extract(candidate):
                return faces + self._extract_edge_faces(candidate, faces)
        for candidate in candidates:
            if faces := self._extract_padded(candidate):
                return faces
        return []

    def _extract_edge_faces(self, image: np.ndarray, existing: list[Face]) -> list[Face]:
        self._detector.setScoreThreshold(EDGE_SCORE_THRESHOLD)
        try:
            candidates = self._detect(image)
        finally:
            self._detector.setScoreThreshold(self._score_threshold)
        img_h, img_w = image.shape[:2]
        max_area = 1.5 * max(f.area for f in existing)
        edges: list[Face] = []
        for box, score, landmarks in candidates:
            x, y, w, h = box
            if w * h > max_area:
                continue
            on_border = x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2
            if on_border and all(_iou(box, f.box) <= 0.3 for f in existing + edges):
                edges.append(
                    Face(
                        box=box,
                        score=score,
                        embedding=self._embed(image, box, landmarks),
                        landmarks=landmarks,
                    )
                )
        return edges

    def _extract_padded(self, image: np.ndarray) -> list[Face]:
        pad = int(0.4 * max(image.shape[:2]))
        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
        img_h, img_w = image.shape[:2]
        faces = []
        for face in self._extract(padded):
            x, y, w, h = face.box
            x, y = max(0, x - pad), max(0, y - pad)
            w, h = min(w, img_w - x), min(h, img_h - y)
            if w > 0 and h > 0:
                faces.append(
                    Face(
                        box=(x, y, w, h),
                        score=face.score,
                        embedding=face.embedding,
                        landmarks={
                            name: (px - pad, py - pad)
                            for name, (px, py) in face.landmarks.items()
                        },
                    )
                )
        return faces

    def _detect(
        self, image: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], float, dict[str, tuple[float, float]]]]:
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
        detections = []
        for row in rows:
            x, y, bw, bh = (int(v) for v in (row[:4] / scale).round())
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, img_w - x), min(bh, img_h - y)
            points = (row[4:14] / scale).reshape(5, 2)
            landmarks = {
                name: (float(px), float(py)) for name, (px, py) in zip(LANDMARK_NAMES, points)
            }
            detections.append(((x, y, bw, bh), float(row[-1]), landmarks))
        return detections

    def _extract(self, image: np.ndarray) -> list[Face]:
        return [
            Face(
                box=box,
                score=score,
                embedding=self._embed(image, box, landmarks),
                landmarks=landmarks,
            )
            for box, score, landmarks in self._detect(image)
        ]

    def _embed(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
        landmarks: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        src = np.array([landmarks[name] for name in LANDMARK_NAMES], dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE)
        if matrix is not None:
            aligned = cv2.warpAffine(image, matrix, (112, 112))
        else:
            x, y, w, h = box
            crop = image[y : y + h, x : x + w]
            aligned = (
                cv2.resize(crop, (112, 112)) if crop.size else np.zeros((112, 112, 3), np.uint8)
            )
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        self._recognizer.setInput(blob)
        feat = self._recognizer.forward().flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-10)


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
