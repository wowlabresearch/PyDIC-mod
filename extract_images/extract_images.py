import cv2
import os
import glob


def find_video_candidates(base_dir):
    # Prefer common default names, then any video file in the folder.
    defaults = ["exp.mp4", "exp3.mp4"]
    candidates = []

    for name in defaults:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            candidates.append(path)

    patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(base_dir, pattern))):
            if path not in candidates:
                candidates.append(path)

    return candidates


def open_video_with_fallback(candidates, base_dir):
    tried = []
    for abs_path in candidates:
        rel_path = os.path.relpath(abs_path, base_dir)

        # Try relative path first to avoid some OpenCV unicode-path issues on Windows.
        for candidate in [rel_path, abs_path]:
            cap = cv2.VideoCapture(candidate)
            ok = cap.isOpened()
            tried.append(candidate)
            if ok:
                return cap, abs_path, tried
            cap.release()

    return None, None, tried

# ---------------------------------------------------------
# 1. Input video path
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

output_dir = os.path.join(PROJECT_ROOT, "uniaxial_tension", "img")
target_count = 6

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Open video
# ---------------------------------------------------------
video_candidates = find_video_candidates(BASE_DIR)
if not video_candidates:
    raise FileNotFoundError(
        f"No video file found in '{BASE_DIR}'. Place a video file there (e.g., exp.mp4)."
    )

original_cwd = os.getcwd()
os.chdir(BASE_DIR)
cap, video_path, tried_paths = open_video_with_fallback(video_candidates, BASE_DIR)
os.chdir(original_cwd)

if cap is None:
    raise RuntimeError(
        "Cannot open the video file with OpenCV. "
        f"Tried: {tried_paths}. "
        "If the file exists, this is often a codec/path issue. "
        "Try converting the video to AVI (MJPG) or placing/running this project in an ASCII-only path."
    )

print(f"Using video: {video_path}")

# ---------------------------------------------------------
# 3. Get video information
# ---------------------------------------------------------
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

# ---------------------------------------------------------
# 4. Extract exactly target_count frames (evenly spaced)
# ---------------------------------------------------------
if frame_count <= 0:
    cap.release()
    raise RuntimeError("Video has no readable frames.")

if target_count <= 0:
    cap.release()
    raise ValueError("target_count must be > 0")

if target_count == 1:
    target_indices = [0]
else:
    # Include first and last frame, and distribute the rest uniformly.
    target_indices = [
        round(i * (frame_count - 1) / (target_count - 1))
        for i in range(target_count)
    ]
target_set = set(target_indices)

frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx in target_set:
        timestamp_sec = frame_idx / fps
        m = int(timestamp_sec // 60)
        s = timestamp_sec % 60
        timestamp_str = f"{m:02d}m{s:05.2f}s"

        output_path = os.path.join(
            output_dir,
            f"{saved_idx:02d}_{timestamp_str}.jpg"
        )

        cv2.imwrite(output_path, frame)
        print(f"  [{saved_idx:02d}] frame {frame_idx:6d}  ->  {timestamp_str}  ->  {os.path.basename(output_path)}")
        saved_idx += 1

        if saved_idx == target_count:
            break

    frame_idx += 1

cap.release()

print(f"\nSaved {saved_idx} frames to '{output_dir}'")
