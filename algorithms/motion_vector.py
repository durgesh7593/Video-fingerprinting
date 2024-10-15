import cv2
import numpy as np
import time

def generate_fingerprints(video_path, step=1):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / 5) 

    
    # Ensure that we have a valid frame to start
    if not ret:
        print("Error: Could not read the first frame.")
        return []
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    fingerprints = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % step == 0:
            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fingerprints.append({"frame_number": frame_number, "flow": flow})
            old_gray = new_gray
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
        "algorithm": "Optical Flow",
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
            # Use np.array_equal to compare flow data
            if not np.array_equal(f1['flow'], f2['flow']):
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
            # Use np.array_equal to compare flow data
            if not np.array_equal(f1['flow'], f2['flow']):
                return False
    return True

def check_robustness_and_versatility(video_path):
    return {
        "determinism": check_determinism(video_path),
        "rotation_resilience": check_transformation_resilience(video_path, 
            [lambda path: rotate_video(path, 90), 
             lambda path: rotate_video(path, 180),
             lambda path: rotate_video(path, 270)])
    }

import numpy as np

def isSame(video_path1, video_path2, step=1, tolerance=1e-5):
    # Generate fingerprints for both videos
    fingerprints1 = generate_fingerprints(video_path1, step)
    fingerprints2 = generate_fingerprints(video_path2, step)

    # Check if both have the same number of fingerprints
    if len(fingerprints1) != len(fingerprints2):
        return False

    # Compare the optical flow between corresponding frames from both videos
    for f1, f2 in zip(fingerprints1, fingerprints2):
        flow1 = f1['flow']
        flow2 = f2['flow']

        # Compute the difference between the two flow arrays
        flow_diff = np.linalg.norm(flow1 - flow2)

        # If the difference is greater than the tolerance, the videos are not the same
        if flow_diff > tolerance:
            return False

    return True
