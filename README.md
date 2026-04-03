# Face Detection System – Real-Time Face Capture with Multi-Format Output#

A real-time face detection tool built using Python and OpenCV. This application captures faces from your webcam and generates outputs in **grayscale**, **color**, and **ASCII art** formats. It also plays a **sound alert** every time a face is detected.

##  Features

-  **Real-time face detection** using Haar Cascade classifier
-  Saves detected face in:
  - Grayscale (`.png`)
  - Color (`.png`)
  - ASCII Art (`.html`)
-  **Sound notification** using Pygame when a face is detected
- Automatically saves all output formats to structured folders
- Educational tool to understand computer vision basics

## Tech Stack

- **Python**
- **OpenCV** – For video capture and image processing
- **NumPy** – For image matrix manipulation
- **PIL (Pillow)** – For ASCII image conversion
- **Pygame** – For audio alert
- **Tkinter** (optional) – If you plan to add a GUI



## Installation

```bash
git clone https://github.com/yourusername/face-detection-system.git
cd face-detection-system
pip install -r requirements.txt
python main.py
