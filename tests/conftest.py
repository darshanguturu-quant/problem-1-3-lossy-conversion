import os
import sys
import shutil
import tempfile
import pytest
import numpy as np
import cv2
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

@pytest.fixture(scope="session")
def test_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix="aintropy_task3_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def sample_sports_video(test_temp_dir):
    video_path = os.path.join(test_temp_dir, "test_sports.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 15, (320, 240))
    for i in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)
        out.write(frame)
    out.release()
    return video_path

@pytest.fixture(scope="session")
def sample_lecture_video(test_temp_dir):
    video_path = os.path.join(test_temp_dir, "test_lecture.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 15, (320, 240))
    for i in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (240, 240, 240)
        out.write(frame)
    out.release()
    return video_path

@pytest.fixture(scope="session")
def invalid_video_file(test_temp_dir):
    corrupt_path = os.path.join(test_temp_dir, "corrupt_video.mp4")
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_RANDOM_BYTES_1234567890")
    return corrupt_path

@pytest.fixture(scope="session")
def dummy_frames():
    frames = []
    for i in range(4):
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        f[:, :] = (i * 40, 100, 150)
        frames.append(f)
    return frames
