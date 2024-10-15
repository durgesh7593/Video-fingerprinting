import time
import numpy as np
import cv2
import os

# Generate DCT-based fingerprints for each frame in the video
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
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(gray_frame))
            fingerprints.append({"frame_number": frame_number, "dct": dct.flatten()})
        frame_number += 1
    cap.release()
    return fingerprints

# Compute video fingerprints and return metadata
def compute_video_fingerprints(video_path, step=1):
    start_time = time.time()
    fingerprints = generate_fingerprints(video_path, step)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end_time = time.time()
    return {
        "algorithm": "DCT-based Fingerprinting",
        "execution_time": end_time - start_time,
        "total_frames_processed": len(fingerprints),
        "total_frames_in_video": total_frames,
        "time_complexity": f"O(n/{step})",
        "space_complexity": "O(n)"
    }

# Check if fingerprints generated over multiple iterations are deterministic
def check_determinism(video_path, iterations=10, step=1):
    all_fingerprints = []
    
    # Generate fingerprints over multiple iterations
    for _ in range(iterations):
        fingerprints = generate_fingerprints(video_path, step)
        all_fingerprints.append(fingerprints)
    
    # Get the first set of fingerprints for comparison
    first_fingerprints = all_fingerprints[0]
    
    # Compare each subsequent set of fingerprints with the first set
    for i in range(1, iterations):
        for f1, f2 in zip(first_fingerprints, all_fingerprints[i]):
            if not np.array_equal(f1['dct'], f2['dct']):
                return False
    
    return True

def check_transformation_resilience(video_path, video_path2, step=1):
    original_fingerprints = generate_fingerprints(video_path, step)
    transformed_fingerprints = generate_fingerprints(video_path2, step)
        
    if len(original_fingerprints) != len(transformed_fingerprints):
        return False
        
    for f1, f2 in zip(original_fingerprints, transformed_fingerprints):
        # Compare DCT coefficients instead of difference
        if not np.array_equal(f1['dct'], f2['dct']):
            return False
            
    return True


def check_robustness_and_versatility(video_path,video_path2):
    return {
        "determinism": check_determinism(video_path),
        "rotation_resilience": check_transformation_resilience(video_path, video_path2)
    }
    
def isSame(video_path1, video_path2, step=1):
    # Generate fingerprints for both videos
    fingerprints1 = generate_fingerprints(video_path1, step)
    fingerprints2 = generate_fingerprints(video_path2, step)

    # Check if both have the same number of fingerprints
    if len(fingerprints1) != len(fingerprints2):
        return False

    # Compare the DCT coefficients of both videos
    for f1, f2 in zip(fingerprints1, fingerprints2):
        if not np.array_equal(f1['dct'], f2['dct']):
            return False

    return True

