import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Dummy user for authentication
users = {
    "admin@badmintonflex.com": "adminpass"
}

# Load CNN model
model_cnn = tf.keras.models.load_model("badminton_shot_classification_model.h5", compile=False)

# Shot labels
class_labels = ['backhand_drive','backhand_net_shot', 'forehand_clear', 'forehand_drive', 'forehand_lift', 'forehand_net_shot']


# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Decorator for login required
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

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

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    # Resize and convert color space
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions and normalize
    frame = np.expand_dims(frame, axis=0)
    return next(test_datagen.flow(frame, batch_size=1))

def extract_frames(video_path):
    """Extract every 5th frame from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 5 == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames

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

    # Extract frames
    frames = extract_frames(video_path)
    if not frames:
        return jsonify({"error": "Could not extract frames from video."})

    # Process frames and collect predictions
    predictions = []
    confidences = []
    
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        pred = model_cnn.predict(processed_frame, verbose=0)[0]
        confidences.append(pred)
        predictions.append(np.argmax(pred))

    # Aggregate results
    avg_confidence = np.mean(confidences, axis=0)
    final_class_index = np.argmax(avg_confidence)
    final_class = class_labels[final_class_index]
    final_confidence = float(round(avg_confidence[final_class_index] * 100, 2))

    # Clean up video file
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"Error deleting video file: {e}")

    return jsonify({
        "prediction": final_class,
        "confidence": final_confidence,
        "message": "Shot classification successful!"
    })

if __name__ == "__main__":
    app.run(debug=True)