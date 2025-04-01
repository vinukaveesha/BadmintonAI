import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model parameters
SEQUENCE_LENGTH = 20
IMG_SIZE = (224, 224)

# Dummy user for authentication
users = {
    "admin@badmintonflex.com": "adminpass"
}

# Load hybrid model
model = tf.keras.models.load_model("epoch_10_valacc_1.00.keras")

# Shot labels
class_labels = ['backhand_drive', 'forehand_clear', 'forehand_drive', 'forehand_lift', 'forehand_net_shot']


# Decorator for login required
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def extract_sequence(video_path):
    """Extract SEQUENCE_LENGTH frames evenly spaced from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, SEQUENCE_LENGTH, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Preprocess frame
                frame = cv2.resize(frame, IMG_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                # Pad with black frame if missing
                frames.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
    finally:
        cap.release()

    return np.array(frames)


def preprocess_sequence(frames):
    """Process frame sequence for model input"""
    # Ensure exactly SEQUENCE_LENGTH frames
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
    
    return np.expand_dims(frames[:SEQUENCE_LENGTH], axis=0)

@app.route("/predict", methods=["POST"])
@login_required
def predict_shot():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Save uploaded video
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    try:
        # Extract and preprocess sequence
        frames = extract_sequence(video_path)
        sequence = preprocess_sequence(frames)
        
        # Make prediction
        pred = model.predict(sequence, verbose=0)[0]
        class_index = np.argmax(pred)
        confidence = float(round(pred[class_index] * 100, 2))
        class_name = class_labels[class_index]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})
    finally:
        # Clean up video file
        try:
            os.remove(video_path)
        except:
            pass

    print(f"Predicted class: {class_name}, Confidence: {confidence}%")

    return jsonify({
        "prediction": class_name,
        "confidence": confidence,
        "message": "Shot classification successful!"
    })


# Authentication routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        if email in users and users[email] == password:
            session["user"] = email
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# Main application route
@app.route("/")
@login_required
def home():
    return render_template("home.html", user=session.get("user"))


if __name__ == "__main__":
    app.run(debug=True)