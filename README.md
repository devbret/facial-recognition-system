# Facial Recognition System

![Screenshot from the facial biometric data dashboard.](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/22490fe4-4839-435b-abcc-696f025239e9.png)

Facial recognition system which identifies known people in photographs with a single local command by matching every detected face against reference photos.

## Application Overview

Teach the application who people are through file management; give each person a folder of reference photos inside `known_faces/`, drop the images you want analysed into `input` and run `python3 app.py`. All processing happens locally on your machine. So no photo or biometric data ever leaves.

Under the hood, OpenCV's `YuNet` model detects each face and its landmarks, then an `ArcFace` recognition model converts each face into an embedding to be matched against your references. Both models are fed full-color images so detection and matching work consistently across all lighting conditions.

Each run writes its results to a folder inside `output`, containing a human-readable `results.txt`, `results.csv`, `biometrics.json` and an `annotated` folder holding a copy of each analyzed image with a red box around each identified person, an orange box around tentative matches and a gray box around unknown faces.

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

8. Create a folder of reference photos for each face you would like identified in `known_faces`

9. Place the images you want analysed into the `input` folder

10. Run the application: `python3 app.py`

11. Results will be returned to the `output` directory

12. Launch the dashboard to explore analysis of results, which opens in your browser: `python3 app.py --dashboard`

13. When finished, close the dashboard: `CTRL + C`

14. Exit the virtual environment: `deactivate`

### Fetching Reference Photos Automatically

Instead of manually gathering reference photos, you can name the people you want to enroll and let the app collect their photos for you:

1. Copy `.env.template` to `.env`: `cp .env.template .env`

2. Add your details to the `.env` file

3. Run the application: `python3 app.py --fetch`

## Other Considerations

Below you will find information not covered in the installation and use sections above. Including the abilities this repo is intended to demonstrate. As well as an overview of the license this code is made available with. And a way to contact the maintainer with questions, suggestions and collaboration opportunities.

### Abilities Demonstrated

This project repo is intended to demonstrate an ability to do the following:

- Identify known people in any batch of photographs with a single terminal command

- Run a modern facial recognition workflow entirely on the local machine

- Automatically retry unrecognized photos with lighting equalization and padded crops

- Produce records of every run to power the visual dashboard

### License Information

This repository is distributed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense and sell copies of this software, including as part of proprietary or commercial work. The single condition is the copyright and permission notices contained in the LICENSE file must be included with any copy or substantial portion of the software that you redistribute. The software is provided "as is", without warranty of any kind, and the copyright holder is not liable for any claim or damages arising from its use.

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
