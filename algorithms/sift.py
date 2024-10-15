import time
import cv2

def generate_fingerprints(video_path, step=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps / 5) 
    fingerprints = []
    frame_number = 0
    sift = cv2.SIFT_create()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % step == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
            fingerprints.append({"frame_number": frame_number, "keypoints": keypoints, "descriptors": descriptors})
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
        "algorithm": "SIFT",
        # "fingerprints": fingerprints,
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
    for i in range(1, iterations):
        if all_fingerprints[i] != first_fingerprints:
            return False
    return True

def check_transformation_resilience(video_path, transformations, step=1):
    original_fingerprints = generate_fingerprints(video_path, step)
    for transform in transformations:
        transformed_video = transform(video_path)
        transformed_fingerprints = generate_fingerprints(transformed_video, step)
        if len(original_fingerprints) != len(transformed_fingerprints):
            return False
        for f1, f2 in zip(original_fingerprints, transformed_fingerprints):
            if f1['keypoints'] != f2['keypoints']:
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

def check_robustness_and_versatility(video_path):
    results = {
        "determinism": check_determinism(video_path),
        "rotation_resilience": check_transformation_resilience(video_path, 
            [lambda path: rotate_video(path, 90), 
             lambda path: rotate_video(path, 180),
             lambda path: rotate_video(path, 270)])
    }
    return results

import cv2
import numpy as np

def isSame(video_path1, video_path2, step=1, ratio_threshold=0.75):
    # Generate fingerprints for both videos
    fingerprints1 = generate_fingerprints(video_path1, step)
    fingerprints2 = generate_fingerprints(video_path2, step)

    # Check if both videos have the same number of frames
    if len(fingerprints1) != len(fingerprints2):
        return False

    # Create a BFMatcher object for matching descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    for f1, f2 in zip(fingerprints1, fingerprints2):
        # Get descriptors for corresponding frames
        descriptors1 = f1['descriptors']
        descriptors2 = f2['descriptors']

        # If either frame has no descriptors, skip comparison
        if descriptors1 is None or descriptors2 is None:
            if descriptors1 != descriptors2:  # If one is None and the other isn't, return False
                return False
            continue

        # Match descriptors using KNN (k-nearest neighbors)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test as per Lowe's paper to identify good matches
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        # Define a threshold for the number of good matches to consider the frames as "same"
        min_matches_threshold = 0.8 * min(len(descriptors1), len(descriptors2))

        if len(good_matches) < min_matches_threshold:
            return False

    return True

