import cv2
import numpy as np
import time

def generate_fingerprints(video_path, step=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / 5) 
    fingerprints = []
    frame_number = 0
    ret, previous_frame = cap.read()

    # Ensure that we have a valid frame to start
    if not ret:
        print(f"Error: Could not read the first frame from {video_path}. Please check the file.")
        return []

    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
        if frame_number % step == 0:
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            difference = cv2.absdiff(previous_frame, current_frame_gray)
            fingerprints.append({"frame_number": frame_number, "difference": difference})
            previous_frame = current_frame_gray
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
        "algorithm": "Spatio-Temporal Features",
        "execution_time": end_time - start_time,
        "total_frames_processed": len(fingerprints),
        "total_frames_in_video": total_frames,
        "time_complexity": f"O(n/{step})",
        "space_complexity": "O(n)"
    }

def check_determinism(video_path, iterations=10, step=1):
    all_fingerprints = []
    for _ in range(iterations):
        fingerprints = generate_fingerprints(video_path, step)
        all_fingerprints.append(fingerprints)
    
    first_fingerprints = all_fingerprints[0]
    
    # Compare each subsequent set of fingerprints with the first set
    for i in range(1, iterations):
        for f1, f2 in zip(first_fingerprints, all_fingerprints[i]):
            # Use np.array_equal to compare difference images
            if not np.array_equal(f1['difference'], f2['difference']):
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

def check_transformation_resilience(video_path, video_path2, step=1):
    original_fingerprints = generate_fingerprints(video_path)
    transformed_fingerprints = generate_fingerprints(video_path2, step)
        
    if len(original_fingerprints) != len(transformed_fingerprints):
        return False
        
    for f1, f2 in zip(original_fingerprints, transformed_fingerprints):
            # Use np.array_equal to compare difference images
        if not np.array_equal(f1['difference'], f2['difference']):
            return False
    return True

def check_robustness_and_versatility(video_path,video_path2):
    return {
        "determinism": check_determinism(video_path),
        "rotation_resilience": check_transformation_resilience(video_path, video_path2)
    }

import cv2
import numpy as np

def isSame(video_path1, video_path2, step=1):
    # Generate fingerprints for both videos
    fingerprints1 = generate_fingerprints(video_path1, step)
    fingerprints2 = generate_fingerprints(video_path2, step)

    # Check if both videos have the same number of frames processed
    if len(fingerprints1) != len(fingerprints2):
        return False

    # Compare each pair of frames (frame differences) from both videos
    for f1, f2 in zip(fingerprints1, fingerprints2):
        # Compare difference matrices of both videos using np.array_equal
        if not np.array_equal(f1['difference'], f2['difference']):
            return False

    # If all frames are the same, return True
    return True
