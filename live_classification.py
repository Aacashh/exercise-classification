import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import joblib
import time

# Global variables
labels_dic = {0: "Back Lunges", 1: "Box Jumps", 2: "Glute Bridges", 3: "Overhead Squat"}
exercise_muscles_dict = {
    "Overhead Squat": ["Quadriceps", "Hamstrings", "Glutes", "Lower Back", "Core"],
    "Box Jumps": ["Quadriceps", "Hamstrings", "Calves", "Glutes"],
    "Glute Bridges": ["Glutes", "Hamstrings", "Lower Back"],
    "Back Lunges": ["Quadriceps", "Hamstrings", "Glutes", "Calves"]
}

def initialize():
    rf_model = joblib.load('Exercise_pred_model.pkl')
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return rf_model, pose

def process_landmarks(frame, pose):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    return results
    # temp = []
    
    # if results and results.pose_landmarks:
    #     for landmark in results.pose_landmarks.landmark:
    #         temp += [landmark.x, landmark.y, landmark.z, landmark.visibility]
    #     temp = np.array(temp).reshape(1, -1)
    #     return temp
    # else:
    #     return None

def classify_exercise(results, rf_model):
    temp = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            temp += [landmark.x, landmark.y, landmark.z, landmark.visibility]

        temp = np.array(temp).reshape(1, -1)
        
        class_label = rf_model.predict(temp)[0]
        return class_label


def display_results(frame, class_label, rep_count):
    class_name = labels_dic[class_label]
    targeted_muscle = exercise_muscles_dict[class_name]
    muscles_string = ", ".join(targeted_muscle)
    text = f'Class: {class_name}'
    muscle_text = f'Muscles: {muscles_string}'
    rep_text = f'Reps: {rep_count}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    muscle_text_size = cv2.getTextSize(muscle_text, font, font_scale, font_thickness)[0]
    rep_text_size = cv2.getTextSize(rep_text, font, font_scale, font_thickness)[0]
    frame[5:5+text_size[1]+muscle_text_size[1]+rep_text_size[1]+40, 5:5+max(text_size[0], muscle_text_size[0] + rep_text_size[1])+10] = [50, 50, 50]

    cv2.putText(frame, text, (10, text_size[1]+10), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(frame, muscle_text, (10, text_size[1]+muscle_text_size[1]+20), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(frame, rep_text, (10, text_size[1]+muscle_text_size[1]+rep_text_size[1]+40), font, font_scale, (0, 255, 0), font_thickness)

    cv2.imshow('Live Exercise Classification', frame)

def calculate_angle(landmark_a, landmark_b, landmark_c):
    """
    Calculate the angle between three landmarks.
    """
    a = np.array([landmark_a.x, landmark_a.y])
    b = np.array([landmark_b.x, landmark_b.y])
    c = np.array([landmark_c.x, landmark_c.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def count_reps(class_label, results, prev_hip_y, state, rep_count, cumulative_motion):
    if results is not None:
        landmarks = results.pose_landmarks.landmark
        
        # Common landmarks
        left_hip_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
        avg_hip_y = (left_hip_y + right_hip_y) / 2.0
        nose_y = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y

        if prev_hip_y:
            difference = abs(avg_hip_y - prev_hip_y)
            cumulative_motion += difference

        motion_threshold = 0.1  # default value, adjust based on testing
        
        # Overhead Squat
        if class_label == "Overhead Squat":
            motion_threshold = 0.2
            if state == "start" and avg_hip_y < prev_hip_y:
                state = "down"
            elif state == "down" and avg_hip_y > prev_hip_y:
                state = "up"
            elif state == "up" and cumulative_motion > motion_threshold:
                rep_count += 1
                state = "start"
                cumulative_motion = 0

        # Box Jumps
        elif class_label == "Box Jumps":
            motion_threshold = 0.5  # Jumping might have a larger motion
            if state == "start" and nose_y < prev_hip_y:
                state = "up"
            elif state == "up" and nose_y > prev_hip_y:
                state = "down"
            elif state == "down" and cumulative_motion > motion_threshold:
                rep_count += 1
                state = "start"
                cumulative_motion = 0

        # Glute Bridges
        elif class_label == "Glute Bridges":
            motion_threshold = 0.1
            if state == "start" and avg_hip_y > prev_hip_y:
                state = "up"
            elif state == "up" and avg_hip_y < prev_hip_y:
                state = "down"
            elif state == "down" and cumulative_motion > motion_threshold:
                rep_count += 1
                state = "start"
                cumulative_motion = 0

        # Back Lunges
        elif class_label == "Back Lunges":
            left_foot_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
            right_foot_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
            foot_difference = abs(left_foot_y - right_foot_y)
            if state == "start" and foot_difference > motion_threshold:
                state = "lunge"
            elif state == "lunge" and foot_difference < motion_threshold:
                rep_count += 1
                state = "start"
                cumulative_motion = 0

    return rep_count, state, cumulative_motion


def count_reps_using_angles(class_label, results, state, rep_count):
    if results is not None:
        landmarks = results.pose_landmarks.landmark

        # Overhead Squat
        if class_label == "Overhead Squat":
            # Monitoring the angle at the hips
            angle_hips = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value])

            if state == "start" and angle_hips < 160:
                state = "down"
            elif state == "down" and angle_hips > 175:
                state = "up"
                rep_count += 1
                state = "start"

        # Box Jumps
        elif class_label == "Box Jumps":
            # Monitoring the angle at the knees
            angle_knees = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value])
            
            if state == "start" and angle_knees < 150:
                state = "up"
            elif state == "up" and angle_knees > 160:
                state = "down"
                rep_count += 1
                state = "start"

        # Glute Bridges
        elif class_label == "Glute Bridges":
            # Monitoring the angle at the hips
            angle_hips = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value])
            
            if state == "start" and angle_hips < 160:
                state = "up"
            elif state == "up" and angle_hips > 175:
                state = "down"
                rep_count += 1
                state = "start"

        # Back Lunges
        elif class_label == "Back Lunges":
            # Monitoring the angle at the forward knee
            angle_knee = calculate_angle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value])
            
            if state == "start" and angle_knee > 155:
                state = "lunge"
            elif state == "lunge" and angle_knee < 145:
                state = "up"
                rep_count += 1
                state = "start"

    return rep_count, state
                
import time

def main():
    rf_model, pose = initialize()
    cap = cv2.VideoCapture(0)
    predicted_classes = []
    
    rep_count = 0
    state = "start"
    prev_hip_y = None
    cumulative_motion = float(0)
    # Initialize class label and timers
    current_time = time.time()
    last_classification_time = 0
    last_class_label = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = process_landmarks(frame, pose)

        if results and results.pose_landmarks:
            # Classify every 3 seconds
            if time.time() - last_classification_time > 2:
                class_label = classify_exercise(results, rf_model)
                if class_label != last_class_label:  # Only append if classification changes
                    predicted_classes.append(class_label)
                last_class_label = class_label
                last_classification_time = time.time()
            else:
                class_label = last_class_label

            # left_hip_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
            # right_hip_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
            # avg_hip_y = (left_hip_y + right_hip_y) / 2.0

            # difference = abs(avg_hip_y - prev_hip_y) if prev_hip_y is not None else 0.0

            rep_count, state = count_reps_using_angles(class_label, results, state, rep_count)
            # prev_hip_y = avg_hip_y
                
            display_results(frame, class_label, rep_count)

        if cv2.getWindowProperty('Live Exercise Classification', cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    majority_class = Counter(predicted_classes).most_common(1)[0][0]
    print(f"Majority Exercise: {majority_class}")


if __name__ == "__main__":
    main()
