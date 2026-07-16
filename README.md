# Facial Recognition System

![Screenshot from the facial biometric data dashboard.](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/22490fe4-4839-435b-abcc-696f025239e9.png)

Facial recognition system which identifies known people in your photographs with a single local command by matching every detected face against your own reference photos.

## Overview

Teach the application who people are through file management; give each person a folder of reference photos inside `known_faces/`, drop the images you want analysed into `input` and run `python3 app.py`. All processing happens locally on your machine. So no photo or biometric data ever leaves.

Under the hood, OpenCV's `YuNet` model detects each face and its landmarks, then `SFace` converts the face into an embedding to be matched against your references. Two refinements keep accuracy high on real-world photos. First, images are normalized to grayscale before embedding so color grading and filters cannot skew a match. And secondly, tight close-ups are automatically retried with padding so the detector can find faces that fill the frame.

Each run writes its results to a folder inside `output`, containing a human-readable `results.txt`, `results.csv`, `biometrics.json` and an `annotated` folder holding a copy of every image with a red box around each identified person and a gray box around unknown faces. Running `python3 -m http.server` opens the results in a visual dashboard in your browser.

## Basic Setup Instructions

Below are the required software programs and instructions for installing and using this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps For Use

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/facial-recognition-system.git`

4. Navigate to the repo's directory: `cd facial-recognition-system`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Create a folder of reference photos for each face you would like identified, for example: `known_faces/Jane Doe/photo.jpg`

9. Place the images you want analysed into the `input` folder

10. Run the application: `python3 app.py`

11. Results will be returned to the `output` directory

12. Launch an HTTP server to explore analysis of results: `python3 -m http.server`

13. Access the frontend dashboard in a browser: `http://localhost:8000`

14. When finished, close the HTTP server: `CTRL + C`

15. Exit the virtual environment: `deactivate`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Identify known people in any batch of photographs with a single terminal command

- Run a modern facial recognition workflow entirely on the local machine

- Normalize images to grayscale and automatically retry tight crops with padding

- Produce records of every run to power the visual dashboard

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
