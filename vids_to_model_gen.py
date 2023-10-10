import mediapipe as mp
import cv2
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Constants
DATASET_PATH = "/content/drive/MyDrive/Exercise Prediction"
OUTPUT_CSV_FILE = "dataset.csv"
SAVED_MODEL = 'Exercise_pred_model.pkl'

# Initialize mediapipe pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = mpPose.PoseLandmark

# Function to extract pose landmarks
def extract_landmarks_from_video(video_path, class_label):
    data = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp = []
        imageWidth, imageHeight = frame.shape[:2]
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, j in zip(points, landmarks):
                temp.extend([j.x, j.y, j.z, j.visibility])
        temp.append(class_label)
        data.append(temp)
    cap.release()
    return data

# Extract landmarks and create dataframe
data = []
for class_folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_folder)
    class_label = int(class_folder)  # Assuming folder names are numbers. Adjust if they are not
    for video_file in os.listdir(class_path):
        video_path = os.path.join(class_path, video_file)
        data.extend(extract_landmarks_from_video(video_path, class_label))

columns = [f"{str(p)[13:]}_{dim}" for p in points for dim in ["x", "y", "z", "vis"]] + ["Label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV_FILE, index=False)

# Train a model
df = pd.read_csv(OUTPUT_CSV_FILE)
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model
joblib.dump(model, SAVED_MODEL)
