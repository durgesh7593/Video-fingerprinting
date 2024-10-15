import time
import numpy as np
import cv2

# Generate color histogram-based fingerprints for each frame in the video
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
            # Compute color histogram in 3D (for R, G, and B channels)
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            # Normalize the histogram
            hist = cv2.normalize(hist, hist).flatten()
            fingerprints.append({"frame_number": frame_number, "histogram": hist})
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
        "algorithm": "Color Histogram",
        "execution_time": end_time - start_time,
        "total_frames_processed": len(fingerprints),
        "total_frames_in_video": total_frames,
        "time_complexity": f"O(n/{step})",
        "space_complexity": "O(n)"
    }

# Check if fingerprints generated over multiple iterations are deterministic
def check_determinism(video_path, iterations=10, step=1):
    all_fingerprints = []
    for _ in range(iterations):
        fingerprints = generate_fingerprints(video_path, step)
        all_fingerprints.append(fingerprints)
    
    first_fingerprints = all_fingerprints[0]
    
    # Compare the histograms of all iterations with the first iteration
    for i in range(1, iterations):
        for f1, f2 in zip(first_fingerprints, all_fingerprints[i]):
            if not np.array_equal(f1['histogram'], f2['histogram']):
                return False
    return True

# Check resilience to video transformations (rotation)
def check_transformation_resilience(video_path, transformations, step=1):
    original_fingerprints = generate_fingerprints(video_path, step)
    
    for transform in transformations:
        transformed_video = transform(video_path)
        transformed_fingerprints = generate_fingerprints(transformed_video, step)
        
        if len(original_fingerprints) != len(transformed_fingerprints):
            return False
        
        for f1, f2 in zip(original_fingerprints, transformed_fingerprints):
            if not np.array_equal(f1['histogram'], f2['histogram']):
                return False
    return True

# Rotate video by a specified angle and return the new video path
def rotate_video(video_path, angle):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f'rotated_{angle}_video.mp4'
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

# Perform robustness checks (determinism and rotation resilience)
def check_robustness_and_versatility(video_path):
    results = {
        "determinism": check_determinism(video_path),
        "rotation_resilience": check_transformation_resilience(video_path, 
            [lambda path: rotate_video(path, 90), 
             lambda path: rotate_video(path, 180),
             lambda path: rotate_video(path, 270)])
    }
    return results

def isSame(video_path1, video_path2, step=1):
    # Generate fingerprints for both videos
    fingerprints1 = generate_fingerprints(video_path1, step)
    fingerprints2 = generate_fingerprints(video_path2, step)

    # Check if both have the same number of fingerprints
    if len(fingerprints1) != len(fingerprints2):
        return False

    # Compare the histograms of both videos
    for f1, f2 in zip(fingerprints1, fingerprints2):
        if not np.array_equal(f1['histogram'], f2['histogram']):
            return False

    return True

