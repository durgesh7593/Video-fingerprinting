import time
from PIL import Image
import imagehash
import cv2

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
            img = Image.fromarray(frame)
            computed_hash = imagehash.average_hash(img)
            fingerprints.append({"frame_number": frame_number, "hash": str(computed_hash)})
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
        "algorithm": "Average Hashing (aHash)",
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

def isSame(video_path1, video_path2):
    fingerprints1 = generate_fingerprints(video_path1)
    fingerprints2 = generate_fingerprints(video_path2)
    if fingerprints1 == fingerprints2:
        return True
    return False

def check_transformation_resilience(video_path, transformations, step=1):
    original_fingerprints = generate_fingerprints(video_path, step)
    for transform in transformations:
        transformed_video = transform(video_path)
        transformed_fingerprints = generate_fingerprints(transformed_video, step)
        if len(original_fingerprints) != len(transformed_fingerprints):
            return False
        for f1, f2 in zip(original_fingerprints, transformed_fingerprints):
            if f1['hash'] != f2['hash']:
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
