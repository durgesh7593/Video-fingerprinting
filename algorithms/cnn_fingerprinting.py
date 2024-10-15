import cv2
import numpy as np
import time
import os
import tensorflow as tf

MODEL_PATH = 'local_pretrained_cnn_model.h5'

# Function to load or save the model locally
def load_or_save_model():
    if os.path.exists(MODEL_PATH):
        print("Loading model from local storage...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Downloading and saving model to local storage...")
        model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
        model.save(MODEL_PATH)
    return model

# Load the pre-trained model
model = load_or_save_model()

def generate_fingerprints(video_path, step=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / 5)
    fingerprints = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % step == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_normalized = frame_resized / 255.0
            frame_input = np.expand_dims(frame_normalized, axis=0)
            fingerprint = model.predict(frame_input)
            fingerprints.append({"frame_number": frame_number, "fingerprint": fingerprint})
        frame_number += 1

    cap.release()
    return fingerprints

def compute_video_fingerprints(video_path, step=1):
    start_time = time.time()
    fingerprints = generate_fingerprints(video_path, step)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end_time = time.time()

    return {
        "algorithm": "CNN-Based Fingerprinting",
        # "fingerprints": fingerprints,
        "execution_time": end_time - start_time,
        "total_frames_processed": len(fingerprints),
        "total_frames_in_video": total_frames,
        "time_complexity": f"O(n/{step})",
        "space_complexity": "O(n)"
    }

def check_determinism(video_path, iterations= 2, step=1):
    # Ensure step is an integ
    step = int(step)
    all_fingerprints = []
    
    for _ in range(iterations):
        fingerprints = generate_fingerprints(video_path, step)
        all_fingerprints.append(fingerprints)
    
    # Take the first set of fingerprints as reference
    first_fingerprints = all_fingerprints[0]
    
    for i in range(1, iterations):
        # Check if the number of fingerprints are the same
        if len(all_fingerprints[i]) != len(first_fingerprints):
            return False
        
        # Check if each fingerprint is identical
        for f1, f2 in zip(all_fingerprints[i], first_fingerprints):
            # Use np.array_equal to compare the fingerprint arrays
            if not np.array_equal(f1['fingerprint'], f2['fingerprint']):
                return False
                
    return True


def rotate_video(video_path, angle):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'rotated_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(frame)

    cap.release()
    out.release()
    return output_path

def check_transformation_resilience(video_path, transformations, step=1):
    original_fingerprints = generate_fingerprints(video_path, step)
    for transform in transformations:
        transformed_video = transform(video_path)
        transformed_fingerprints = generate_fingerprints(transformed_video, step)
        if len(original_fingerprints) != len(transformed_fingerprints):
            return False
        for f1, f2 in zip(original_fingerprints, transformed_fingerprints):
            if not np.array_equal(f1['fingerprint'], f2['fingerprint']):
                return False
    return True

def check_robustness_and_versatility(video_path):
    return {
        "determinism": check_determinism(video_path),
        "rotation_resilience": check_transformation_resilience(video_path, 
            [lambda path: rotate_video(path, 90), 
             lambda path: rotate_video(path, 180),
             lambda path: rotate_video(path, 270)]),
    }

def isSame(video_path1, video_path2, step=1):
    # Generate fingerprints for both videos
    fingerprints1 = generate_fingerprints(video_path1, step)
    fingerprints2 = generate_fingerprints(video_path2, step)

    # Check if both have the same number of fingerprints
    if len(fingerprints1) != len(fingerprints2):
        return False

    # Compare the fingerprints of both videos
    for f1, f2 in zip(fingerprints1, fingerprints2):
        if not np.array_equal(f1['fingerprint'], f2['fingerprint']):
            return False
            
    return True
