# Facial Recognition System

Automated facial recognition system which identifies known people in photographs with a single command.

## Overview

First teach the application who people are through plain file management; each person has a folder of reference photos inside `known_faces/` and any images you want analysed go into `input/`. Running `python3 app.py` does everything else. All processing happens locally on your machine. No photo or biometric data is ever sent anywhere.

Under the hood, the system uses `OpenCV`'s `YuNet` model to detect faces and their landmarks, then `SFace` to convert each face into an embedding, matching faces by similarity against the references. To help with accuracy images are normalized as grayscale before embedding and close-ups are automatically retried with padding.

Each run writes its results to a timestamped folder in `output/`, containing a `results.txt`, `results.csv` and annotated copies of every image with a red box around each identified person and a gray box around unknown faces.
