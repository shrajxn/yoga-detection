import cv2
import os
import mediapipe as mp
import threading
import queue

os.environ["TF_XLA_FLAGS"] = "--tf_xla_disable_xnnpack=true"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


poses = {
    'tadasana': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.5, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.5, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.5, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.5, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.5, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.5, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.5, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.5, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.5, 0.1), mp_pose.PoseLandmark.RIGHT_WRIST: (0.5, 0.1)
    },
    'adho_mukha_svanasana': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.5, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.5, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.5, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.5, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.5, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.5, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.5, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.5, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.5, 0.1), mp_pose.PoseLandmark.RIGHT_WRIST: (0.5, 0.1)
    },
    'balasana': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.5, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.5, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.5, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.5, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.5, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.5, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.5, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.5, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.5, 0.1), mp_pose.PoseLandmark.RIGHT_WRIST: (0.5, 0.1)
    },
    'virabhadrasana_i': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.4, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.6, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.4, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.6, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.4, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.6, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.4, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.6, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.4, 0.1), mp_pose.PoseLandmark.RIGHT_WRIST: (0.6, 0.1)
    },
    'virabhadrasana_ii': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.4, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.6, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.4, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.6, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.4, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.6, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.4, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.6, 0.3),
        mp_pose.PoseLandmark.LEFT_ELBOW: (0.35, 0.3), mp_pose.PoseLandmark.RIGHT_ELBOW: (0.65, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.3, 0.4), mp_pose.PoseLandmark.RIGHT_WRIST: (0.7, 0.4)
    },
    'virabhadrasana_iii': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.4, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.6, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.4, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.6, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.4, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.6, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.4, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.6, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.4, 0.1), mp_pose.PoseLandmark.RIGHT_WRIST: (0.6, 0.1)
    },
    'vrksasana': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.5, 0.8), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.5, 0.8),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.5, 0.6), mp_pose.PoseLandmark.RIGHT_KNEE: (0.5, 0.6),
        mp_pose.PoseLandmark.LEFT_HIP: (0.5, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.5, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.5, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.5, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.5, 0.1), mp_pose.PoseLandmark.RIGHT_WRIST: (0.5, 0.1)
    },
    'trikonasana': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.4, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.6, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.4, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.6, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.4, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.6, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.4, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.6, 0.3),
        mp_pose.PoseLandmark.LEFT_ELBOW: (0.35, 0.3), mp_pose.PoseLandmark.RIGHT_ELBOW: (0.65, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.3, 0.4), mp_pose.PoseLandmark.RIGHT_WRIST: (0.7, 0.4)
    },
    'utthita_parsvakonasana': {
        mp_pose.PoseLandmark.LEFT_ANKLE: (0.4, 0.9), mp_pose.PoseLandmark.RIGHT_ANKLE: (0.6, 0.9),
        mp_pose.PoseLandmark.LEFT_KNEE: (0.4, 0.7), mp_pose.PoseLandmark.RIGHT_KNEE: (0.6, 0.7),
        mp_pose.PoseLandmark.LEFT_HIP: (0.4, 0.5), mp_pose.PoseLandmark.RIGHT_HIP: (0.6, 0.5),
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0.4, 0.3), mp_pose.PoseLandmark.RIGHT_SHOULDER: (0.6, 0.3),
        mp_pose.PoseLandmark.LEFT_ELBOW: (0.35, 0.3), mp_pose.PoseLandmark.RIGHT_ELBOW: (0.65, 0.3),
        mp_pose.PoseLandmark.LEFT_WRIST: (0.3, 0.4), mp_pose.PoseLandmark.RIGHT_WRIST: (0.7, 0.4)
    }}

def extract_landmarks(pose_landmarks, relevant_parts):
    landmarks = {}
    for part in relevant_parts:
        landmark = pose_landmarks.landmark[part]
        landmarks[part] = (landmark.x, landmark.y)
    return landmarks

def compare_pose(detected_landmarks, poses):
    min_distance = float('inf')
    best_pose = None

    for pose_name, reference_landmarks in poses.items():
        distance = sum(
            ((detected_landmarks[part][0] - reference_landmarks[part][0]) ** 2 + 
             (detected_landmarks[part][1] - reference_landmarks[part][1]) ** 2) ** 0.5
            for part in detected_landmarks if part in reference_landmarks
        )
        if distance < min_distance:
            min_distance = distance
            best_pose = pose_name

    return best_pose


def process_frames():
    global detected_pose, results
    while running:
        if not frames_queue.empty():
            frame = frames_queue.get()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(frame_rgb)
            if results.pose_landmarks:

                detected_landmarks = extract_landmarks(results.pose_landmarks, relevant_parts)

                detected_pose = compare_pose(detected_landmarks, poses)
            else:
                detected_pose = None

detected_pose = None
running = True

# Define the relevant parts for landmark extraction
relevant_parts = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                  mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                  mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]

# Start video capture
cap = cv2.VideoCapture(0)

frames_queue = queue.Queue()

frame_thread = threading.Thread(target=process_frames)
frame_thread.start()

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frames_queue.put(frame)

    if detected_pose:
        
        green = (0, 255, 0)
        red = (0, 0, 255)

        # Overlay correct poses with green text and incorrect poses with red text
        if detected_pose in poses:
            cv2.putText(frame, f'Pose: {detected_pose}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Unknown Pose', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2, cv2.LINE_AA)
    
    if 'results' in globals() and results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break

cap.release()
cv2.destroyAllWindows()


frame_thread.join()
