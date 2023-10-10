import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import joblib

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

def count_reps(class_label, results, avg_hip_y, difference, prev_hip_y, state, rep_count):
    if prev_hip_y is not None and results is not None:
        # left_hip_y = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
        # right_hip_y = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
        # avg_hip_y = (left_hip_y + right_hip_y) / 2.0

        difference = abs(avg_hip_y - prev_hip_y)
        dynamic_threshold = prev_hip_y * 0.02  # 2% of the previous hip position

        if class_label == "Overhead Squat":
            if state == "start" and difference > dynamic_threshold and avg_hip_y < prev_hip_y:
                state = "down"
            elif state == "down" and difference > dynamic_threshold and avg_hip_y > prev_hip_y:
                state = "up"
            elif state == "up":
                rep_count += 1
                state = "start"

        elif class_label == "Box Jumps":
            nose_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE.value].y
            if state == "start" and difference > dynamic_threshold and nose_y < prev_hip_y:
                state = "up"
            elif state == "up" and difference > dynamic_threshold and nose_y > prev_hip_y:
                state = "down"
            elif state == "down":
                rep_count += 1
                state = "start"

        elif class_label == "Glute Bridges":
            if state == "start" and difference > dynamic_threshold and avg_hip_y > prev_hip_y:
                state = "up"
            elif state == "up" and difference > dynamic_threshold and avg_hip_y < prev_hip_y:
                state = "down"
            elif state == "down":
                rep_count += 1
                state = "start"

        elif class_label == "Back Lunges":
            left_foot_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
            right_foot_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
            foot_difference = abs(left_foot_y - right_foot_y)
            
            if state == "start" and foot_difference > dynamic_threshold:
                state = "lunge"
            elif state == "lunge" and foot_difference < dynamic_threshold:
                rep_count += 1
                state = "start"

    return rep_count, state


                
def main():
    rf_model, pose = initialize()
    cap = cv2.VideoCapture(0)
    predicted_classes = []
    
    rep_count = 0
    state = "start"
    prev_hip_y = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = process_landmarks(frame, pose)
        if results is not None:
            class_label = classify_exercise(results, rf_model)
            predicted_classes.append(class_label)
            
            if results.pose_landmarks is not None:
                left_hip_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
                right_hip_y = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
                avg_hip_y = (left_hip_y + right_hip_y) / 2.0
                if prev_hip_y is not None:
                    
                    difference = abs(avg_hip_y - prev_hip_y)
                else:
                    
                    difference = float(0)
                rep_count, state = count_reps(class_label, results, avg_hip_y, difference, prev_hip_y, state, rep_count)
                prev_hip_y = avg_hip_y
                
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
