import cv2
import os

# ---------------------------------------------------------
# 1. Input video path
# ---------------------------------------------------------
video_path = "exp3.mp4"
output_dir = "extracted_frames"

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Open video
# ---------------------------------------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open video file. Check the video_path.")

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
# 4. Extract frames
# ---------------------------------------------------------
interval_sec = 1.0
frame_interval = int(fps * interval_sec)

frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save one frame every interval_sec seconds
    if frame_idx % frame_interval == 0:
        output_path = os.path.join(
            output_dir,
            f"frame_{saved_idx:04d}.jpg"
        )

        cv2.imwrite(output_path, frame)
        saved_idx += 1

    frame_idx += 1

cap.release()

print(f"Saved {saved_idx} frames to '{output_dir}'")