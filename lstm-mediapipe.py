import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

SEQUENCE_LENGTH = 20
FEATURE_DIM = 195  # 33*4 + 21*3

model = tf.keras.models.load_model('models/badminton_shot_model_large_dataset.keras')

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

LABELS = ['backhand_drive', 'backhand_net_shot', 'forehand_clear', 'forehand_drive', 'forehand_lift', 'forehand_net_shot']

print(LABELS)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Buffer to store sequences
sequence = []
predicted_class = 'Waiting...'
confidence = 0

MEAN = np.load('large-dataset-keypoints/mean.npy')  
STD = np.load('large-dataset-keypoints/std.npy')

def extract_keypoints(results_pose, results_hands):
    pose_keypoints = np.zeros(33 * 4)  # 33 landmarks with x,y,z,visibility
    if results_pose.pose_landmarks:
        for i, lm in enumerate(results_pose.pose_landmarks.landmark):
            pose_keypoints[i*4:i*4+4] = [lm.x, lm.y, lm.z, lm.visibility]

    # Just get right hand (or left, if needed)
    hand_keypoints = np.zeros(21 * 3)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                hand_keypoints[i*3:i*3+3] = [lm.x, lm.y, lm.z]
            break  # only one hand for simplicity
    return np.concatenate([pose_keypoints, hand_keypoints])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)
    
    keypoints = extract_keypoints(results_pose, results_hands)
    
    # Add to sequence buffer
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]
    
    if len(sequence) == SEQUENCE_LENGTH:
        
        normalized_sequence = (np.array(sequence) - MEAN) / STD
        
        prediction = model.predict(np.expand_dims(normalized_sequence, axis=0))
        
        predicted_index = np.argmax(prediction)
        predicted_class = LABELS[predicted_index]
        confidence = np.max(prediction)
    
    cv2.putText(frame, f"Shot: {predicted_class}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Badminton Shot Recognition', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()