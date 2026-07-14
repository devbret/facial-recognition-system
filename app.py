from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _reexec_into_venv() -> None:
    if sys.prefix != sys.base_prefix:
        return
    for python in (ROOT / "venv/bin/python", ROOT / "venv/Scripts/python.exe"):
        if python.exists():
            os.execv(str(python), [str(python), *sys.argv])


_reexec_into_venv()

import argparse
import csv
from datetime import datetime

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit(
        "Dependencies are missing. Set them up with:\n"
        "  python3 -m venv venv\n"
        "  venv/bin/pip install -r requirements.txt"
    )

from facerec.engine import DEFAULT_THRESHOLD, Face, FaceEngine, load_image
from facerec.models import fetch_models, missing_models

KNOWN_DIR = ROOT / "known_faces"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

RED = (0, 0, 220)
GRAY = (140, 140, 140)


class Gallery:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.labels.append(name)
        self._embeddings.append(embedding)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
        if not self._embeddings:
            return None, 0.0
        sims = np.stack(self._embeddings) @ embedding
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (self.labels[best] if score >= threshold else None), score


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def enroll_known_faces(engine: FaceEngine, log) -> Gallery:
    gallery = Gallery()
    people = sorted(d for d in KNOWN_DIR.iterdir() if d.is_dir()) if KNOWN_DIR.is_dir() else []
    for person in people:
        enrolled = 0
        for path in list_images(person):
            try:
                faces = engine.extract_faces(load_image(path))
            except ValueError as exc:
                log(f"  WARNING: {exc}")
                continue
            if not faces:
                log(f"  WARNING: no face found in {path.relative_to(ROOT)}, skipped")
                continue
            if len(faces) > 1:
                log(f"  note: {len(faces)} faces in {path.relative_to(ROOT)}, using the largest")
            gallery.add(person.name, max(faces, key=lambda f: f.area).embedding)
            enrolled += 1
        log(f"  {person.name}: {enrolled} reference photo(s)")
    if not gallery.labels:
        log("")
        log("  WARNING: nobody is enrolled - every face below will be 'Unknown'.")
        log(f"  Add reference photos as {KNOWN_DIR.name}/<Person Name>/photo.jpg and rerun.")
    return gallery


def annotate(image: np.ndarray, face: Face, label: str, matched: bool) -> None:
    x, y, w, h = face.box
    color = RED if matched else GRAY
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    ty = y - 8 if y - th - baseline - 8 >= 0 else y + h + th + 8
    cv2.rectangle(image, (x, ty - th - baseline), (x + tw + 4, ty + baseline), color, -1)
    cv2.putText(image, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def run(images: list[Path], threshold: float) -> Path:
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    annotated_dir = run_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = []

    def log(line: str) -> None:
        print(line)
        report.append(line)

    log(f"Facial recognition run  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Threshold: {threshold}")
    log("")
    log("Enrolling known faces:")
    engine = FaceEngine()
    gallery = enroll_known_faces(engine, log)
    log(
        "Known people: "
        + (", ".join(f"{n} ({c})" for n, c in sorted(gallery.counts().items())) or "none")
    )
    log("")

    rows: list[dict] = []
    matched_total = 0

    for path in images:
        try:
            image = load_image(path)
        except ValueError as exc:
            log(f"WARNING: {exc}, skipped")
            continue
        faces = engine.extract_faces(image)
        log(f"{path.name}: {len(faces)} face(s)")
        for i, face in enumerate(faces, start=1):
            name, score = gallery.match(face.embedding, threshold)
            matched_total += name is not None
            x, y, w, h = face.box
            log(f"  face {i}: {name or 'Unknown'} (similarity {score:.3f}) at x={x} y={y} w={w} h={h}")
            rows.append(
                {
                    "image": path.name,
                    "face": i,
                    "name": name or "Unknown",
                    "similarity": f"{score:.4f}",
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "detector_confidence": f"{face.score:.4f}",
                }
            )
            annotate(image, face, f"{name or 'Unknown'} ({score:.3f})", matched=name is not None)
        cv2.imwrite(str(annotated_dir / path.name), image)

    log("")
    log(
        f"Summary: {len(images)} image(s), {len(rows)} face(s), "
        f"{matched_total} matched, {len(rows) - matched_total} unknown"
    )

    (run_dir / "results.txt").write_text("\n".join(report) + "\n")
    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "face", "name", "similarity",
                "x", "y", "width", "height", "detector_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Facial recognition over image files.")
    parser.add_argument(
        "images", nargs="*", help=f"images to analyse (default: everything in {INPUT_DIR.name}/)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"cosine similarity needed for a match (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    KNOWN_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)

    if missing_models():
        print("First run: downloading face models (~39 MB) ...")
        fetch_models()

    images = [Path(p) for p in args.images] or list_images(INPUT_DIR)
    if not images:
        print(
            f"No images to analyse. Drop photos into {INPUT_DIR.name}/ "
            f"(and reference photos into {KNOWN_DIR.name}/<Person Name>/), then rerun."
        )
        return 1

    run_dir = run(images, args.threshold)
    print(f"\nResults written to {run_dir.relative_to(ROOT)}/ (results.txt, results.csv, annotated/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
