# BadmintonAI: Real-time Shot Classification System

![Badminton Shot Classification Demo](hyper-paramter-tunning.png)

A hybrid deep learning system that classifies badminton shots from video sequences using CNN-LSTM architecture with real-time pose estimation capabilities.

## Features

- **Hybrid CNN-LSTM Model**: Processes temporal sequences of video frames for accurate shot classification
- **Real-time Pose Estimation**: MediaPipe-based shot recognition from live camera feed
- **Web Application**: Flask-based interface with user authentication
- **High Accuracy**: 100% validation accuracy on 5 shot types (clears, drives, lifts, net shots)
- **Video Processing**: Frame extraction, normalization, and sequence generation pipeline

## Tech Stack

**Deep Learning:**  
TensorFlow, Keras, MediaPipe, OpenCV, scikit-learn  

**Web Framework:**  
Flask, Bootstrap  

**Data Processing:**  
NumPy, Pandas, Joblib  

## Installation

1. Clone repository:

git clone https://github.com/vinukaveesha/badminton-shot-classifier.git
cd badminton-shot-classifier 

2. Create virtual environment

python -m venv badminton-env
source badminton-env/bin/activate  # Linux/Mac
badminton-env\Scripts\activate    # Windows

3. Install dependencies

pip install -r requirements.txt

## Usage

**Web Application**

Visit http://localhost:5000 in your browser (use admin@badmintonflex.com / adminpass)