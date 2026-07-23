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
import functools
import http.server
import json
import webbrowser
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

from facerec.engine import (
    DEFAULT_PROBABLE_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_THRESHOLD,
    Face,
    FaceEngine,
    load_image,
)
from facerec.models import fetch_models, missing_models

KNOWN_DIR = ROOT / "known_faces"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

RED = (0, 0, 220)
ORANGE = (0, 140, 230)
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

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float, str | None]:
        if not self._embeddings:
            return None, 0.0, None
        sims = np.stack(self._embeddings) @ embedding
        best = int(np.argmax(sims))
        score = float(sims[best])
        best_label = self.labels[best]
        return (best_label if score >= threshold else None), score, best_label


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
            confident = [f for f in faces if f.score >= 0.5] or faces
            gallery.add(person.name, max(confident, key=lambda f: f.area).embedding)
            enrolled += 1
        log(f"  {person.name}: {enrolled} reference photo(s)")
    if not gallery.labels:
        log("")
        log("  WARNING: nobody is enrolled - every face below will be 'Unknown'.")
        log(f"  Add reference photos as {KNOWN_DIR.name}/<Person Name>/photo.jpg and rerun.")
    return gallery


METRIC_FIELDS = [
    "interocular_distance", "eye_to_mouth_distance", "nose_to_eye_midpoint",
    "nose_to_mouth_midpoint", "mouth_width", "right_eye_to_nose", "left_eye_to_nose",
    "mouth_right_to_nose", "mouth_left_to_nose", "roll_degrees", "nose_offset_ratio",
    "bilateral_symmetry", "mouth_to_interocular_ratio", "eye_mouth_to_interocular_ratio",
    "nose_position_ratio", "box_aspect_ratio", "interocular_to_face_width",
    "mouth_to_face_width", "face_area", "face_area_ratio", "face_center_x_ratio",
    "face_center_y_ratio", "brightness", "contrast", "sharpness",
]

EPS = 1e-10


def face_metrics(face: Face, image: np.ndarray) -> dict[str, float]:
    lm = {name: np.array(point) for name, point in face.landmarks.items()}
    x, y, w, h = face.box
    img_h, img_w = image.shape[:2]
    eye_mid = (lm["right_eye"] + lm["left_eye"]) / 2
    mouth_mid = (lm["mouth_right"] + lm["mouth_left"]) / 2
    eye_vec = lm["left_eye"] - lm["right_eye"]
    interocular = float(np.linalg.norm(eye_vec))
    eye_to_mouth = float(np.linalg.norm(mouth_mid - eye_mid))
    mouth_width = float(np.linalg.norm(lm["mouth_left"] - lm["mouth_right"]))
    nose_to_eye = float(np.linalg.norm(lm["nose"] - eye_mid))
    right_eye_nose = float(np.linalg.norm(lm["nose"] - lm["right_eye"]))
    left_eye_nose = float(np.linalg.norm(lm["nose"] - lm["left_eye"]))
    mouth_right_nose = float(np.linalg.norm(lm["nose"] - lm["mouth_right"]))
    mouth_left_nose = float(np.linalg.norm(lm["nose"] - lm["mouth_left"]))
    nose_lateral = float(np.dot(lm["nose"] - eye_mid, eye_vec / (interocular + EPS)))
    crop = image[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.zeros((1, 1), np.uint8)
    values = {
        "interocular_distance": interocular,
        "eye_to_mouth_distance": eye_to_mouth,
        "nose_to_eye_midpoint": nose_to_eye,
        "nose_to_mouth_midpoint": float(np.linalg.norm(lm["nose"] - mouth_mid)),
        "mouth_width": mouth_width,
        "right_eye_to_nose": right_eye_nose,
        "left_eye_to_nose": left_eye_nose,
        "mouth_right_to_nose": mouth_right_nose,
        "mouth_left_to_nose": mouth_left_nose,
        "roll_degrees": float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))),
        "nose_offset_ratio": nose_lateral / (interocular / 2 + EPS),
        "bilateral_symmetry": 1
        - 0.5
        * (
            abs(right_eye_nose - left_eye_nose) / (right_eye_nose + left_eye_nose + EPS)
            + abs(mouth_right_nose - mouth_left_nose) / (mouth_right_nose + mouth_left_nose + EPS)
        ),
        "mouth_to_interocular_ratio": mouth_width / (interocular + EPS),
        "eye_mouth_to_interocular_ratio": eye_to_mouth / (interocular + EPS),
        "nose_position_ratio": nose_to_eye / (eye_to_mouth + EPS),
        "box_aspect_ratio": w / (h + EPS),
        "interocular_to_face_width": interocular / (w + EPS),
        "mouth_to_face_width": mouth_width / (w + EPS),
        "face_area": w * h,
        "face_area_ratio": w * h / (img_w * img_h),
        "face_center_x_ratio": (x + w / 2) / img_w,
        "face_center_y_ratio": (y + h / 2) / img_h,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    return {k: (v if isinstance(v, int) else round(v, 4)) for k, v in values.items()}


def annotate(image: np.ndarray, face: Face, label: str, tier: str) -> None:
    x, y, w, h = face.box
    color = {"matched": RED, "probable": ORANGE}.get(tier, GRAY)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    ty = y - 8 if y - th - baseline - 8 >= 0 else y + h + th + 8
    cv2.rectangle(image, (x, ty - th - baseline), (x + tw + 4, ty + baseline), color, -1)
    cv2.putText(image, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def run(
    images: list[Path], threshold: float, probable_threshold: float, detect_threshold: float
) -> Path:
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    annotated_dir = run_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = []

    def log(line: str) -> None:
        print(line)
        report.append(line)

    log(f"Facial recognition run  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Match threshold: {threshold}  (probable from {probable_threshold})")
    log(f"Detection threshold: {detect_threshold}")
    log("")
    log("Enrolling known faces:")
    engine = FaceEngine(score_threshold=detect_threshold)
    gallery = enroll_known_faces(engine, log)
    log(
        "Known people: "
        + (", ".join(f"{n} ({c})" for n, c in sorted(gallery.counts().items())) or "none")
    )
    log("")

    rows: list[dict] = []
    face_records: list[dict] = []
    matched_total = 0
    probable_total = 0

    for path in images:
        try:
            image = load_image(path)
        except ValueError as exc:
            log(f"WARNING: {exc}, skipped")
            continue
        faces = engine.extract_faces(image)
        log(f"{path.name}: {len(faces)} face(s)")
        for i, face in enumerate(faces, start=1):
            name, score, best_label = gallery.match(face.embedding, threshold)
            if name is not None:
                tier, shown = "matched", name
            elif best_label is not None and score >= probable_threshold:
                tier, shown = "probable", f"{best_label}?"
            else:
                tier, shown = "unknown", "Unknown"
            matched_total += tier == "matched"
            probable_total += tier == "probable"
            x, y, w, h = face.box
            metrics = face_metrics(face, image)
            log(f"  face {i}: {shown} (similarity {score:.3f}) at x={x} y={y} w={w} h={h}")
            rows.append(
                {
                    "image": path.name,
                    "face": i,
                    "name": shown,
                    "tier": tier,
                    "best_match": best_label or "",
                    "similarity": f"{score:.4f}",
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "detector_confidence": f"{face.score:.4f}",
                    **metrics,
                }
            )
            face_records.append(
                {
                    "image": path.name,
                    "image_size": {"width": image.shape[1], "height": image.shape[0]},
                    "face": i,
                    "name": shown,
                    "tier": tier,
                    "best_match": best_label,
                    "matched": tier == "matched",
                    "similarity": round(score, 4),
                    "detector_confidence": round(face.score, 4),
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "landmarks": {
                        k: [round(px, 1), round(py, 1)] for k, (px, py) in face.landmarks.items()
                    },
                    "metrics": metrics,
                    "embedding": [round(float(v), 6) for v in face.embedding],
                }
            )
            annotate(image, face, f"{shown} ({score:.3f})", tier)
        cv2.imwrite(str(annotated_dir / path.name), image)

    log("")
    log(
        f"Summary: {len(images)} image(s), {len(rows)} face(s), "
        f"{matched_total} matched, {probable_total} probable, "
        f"{len(rows) - matched_total - probable_total} unknown"
    )

    (run_dir / "results.txt").write_text("\n".join(report) + "\n")
    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "face", "name", "tier", "best_match", "similarity",
                "x", "y", "width", "height", "detector_confidence",
                *METRIC_FIELDS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    biometrics = {
        "run": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "threshold": threshold,
        "probable_threshold": probable_threshold,
        "detection_threshold": detect_threshold,
        "images_analyzed": len(images),
        "faces_found": len(face_records),
        "faces_matched": matched_total,
        "faces_probable": probable_total,
        "known_people": gallery.counts(),
        "faces": face_records,
    }
    (run_dir / "biometrics.json").write_text(json.dumps(biometrics, indent=2) + "\n")
    runs = sorted(
        (d.name for d in OUTPUT_DIR.iterdir() if (d / "biometrics.json").is_file()),
        reverse=True,
    )
    (OUTPUT_DIR / "runs.json").write_text(json.dumps(runs) + "\n")
    return run_dir


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def fetch_reference_photos() -> int:
    KNOWN_DIR.mkdir(exist_ok=True)
    try:
        from facerec.fetch import fetch_reference_faces
    except ImportError as exc:
        sys.exit(
            "The --fetch feature needs extra packages (requests, python-dotenv).\n"
            "Install them with:  venv/bin/pip install -r requirements.txt\n"
            f"({exc})"
        )

    summary = fetch_reference_faces()
    faces_root = Path(summary["faces_root"])
    print(
        f"\nDownloaded {summary['downloaded']} reference photo(s) for "
        f"{len(summary['names'])} name(s) into {faces_root.name}/ "
        f"(skipped {summary['skipped']}, {summary['search_errors']} search error(s))."
    )
    print(
        "Review each folder, remove any photos that show the wrong person, then "
        "add images to analyse in "
        f"{INPUT_DIR.name}/ and run: python3 app.py"
    )
    return 0


def serve_dashboard() -> int:
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/"
        print(f"Dashboard running at {url} (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


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
    parser.add_argument(
        "--probable-threshold",
        type=float,
        default=DEFAULT_PROBABLE_THRESHOLD,
        help=(
            "similarity from which a below-threshold face is labeled as a tentative "
            f"'Name?' match instead of Unknown (default: {DEFAULT_PROBABLE_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--detect-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=(
            "detector confidence needed to count as a face "
            f"(default: {DEFAULT_SCORE_THRESHOLD}); lower it to find more faces"
        ),
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="open the visualization dashboard in your browser",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=(
            "download reference photos for the names listed in .env into "
            f"{KNOWN_DIR.name}/<name>/, then exit (see .env.template)"
        ),
    )
    args = parser.parse_args()

    if args.dashboard:
        return serve_dashboard()

    if args.fetch:
        return fetch_reference_photos()

    KNOWN_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)

    if missing_models():
        print("First run: downloading face models (~340 MB) ...")
        fetch_models()

    images = [Path(p) for p in args.images] or list_images(INPUT_DIR)
    if not images:
        print(
            f"No images to analyse. Drop photos into {INPUT_DIR.name}/ "
            f"(and reference photos into {KNOWN_DIR.name}/<Person Name>/), then rerun."
        )
        return 1

    run_dir = run(images, args.threshold, args.probable_threshold, args.detect_threshold)
    print(
        f"\nResults written to {run_dir.relative_to(ROOT)}/ "
        "(results.txt, results.csv, biometrics.json, annotated/)"
    )
    print("View them with: python3 app.py --dashboard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
