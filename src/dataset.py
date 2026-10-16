"""
Dataset loader, label validator, and synthetic video test generator.
Provides reproducible dataset splits and validates label alignment with assignment taxonomy.
"""

import os
import json
import csv
import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from src.classifier.stage1 import DEFAULT_CLASSES

logger = logging.getLogger(__name__)


def inspect_videomme_dataset(jsonl_path: str = "test.jsonl") -> Dict[str, Any]:
    """
    Inspects VideoMME sample annotations and checks available labels.
    Explicitly reports domain/sub_category presence and the absence of native routing labels.
    """
    if not os.path.exists(jsonl_path):
        return {
            "error": f"Dataset file not found at {jsonl_path}",
            "available": False,
        }

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    unique_videos = {}
    for r in records:
        vid = r.get("video_id")
        if vid and vid not in unique_videos:
            unique_videos[vid] = {
                "domain": r.get("domain"),
                "sub_category": r.get("sub_category"),
                "duration": r.get("duration"),
            }

    domains = {}
    sub_categories = {}
    for v in unique_videos.values():
        d = v["domain"] or "unknown"
        s = v["sub_category"] or "unknown"
        domains[d] = domains.get(d, 0) + 1
        sub_categories[s] = sub_categories.get(s, 0) + 1

    return {
        "available": True,
        "total_qa_records": len(records),
        "unique_videos": len(unique_videos),
        "domains": domains,
        "sub_categories": sub_categories,
        "native_routing_labels_present": False,
        "limitation_note": (
            "VideoMME provides 'domain' and 'sub_category' for QA reasoning, "
            "but does NOT natively provide chunking-oriented video-type labels "
            "(gameplay, podcast, news_lecture, sports, surveillance, meetings, etc.). "
            "Priors or verified manifests must be applied."
        ),
    }


def generate_synthetic_video(
    output_path: str,
    video_type: str,
    duration_s: float = 1.0,
    fps: int = 15,
    width: int = 320,
    height: int = 240,
) -> str:
    """
    Generates a deterministic, tiny synthetic MP4 video tailored to the target class
    to enable fully offline, self-contained testing and reproducible benchmarking.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = max(5, int(duration_s * fps))

    for i in range(total_frames):
        # Base canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if video_type == "gameplay":
            # Dark background, bright colorful HUD and moving character box
            frame[:, :] = (30, 20, 20)
            cv2.rectangle(frame, (10, 10), (100, 30), (0, 255, 0), -1)  # HUD bar
            pos_x = (i * 20) % (width - 40)
            cv2.rectangle(frame, (pos_x, 100), (pos_x + 30, 150), (0, 0, 255), -1)  # Red sprite
        elif video_type == "sports":
            # Vibrant green grass pitch, high motion white ball
            frame[:, :] = (34, 139, 34)  # Forest Green
            ball_x = int((i / float(total_frames)) * width)
            ball_y = int(120 + 40 * np.sin(i * 0.5))
            cv2.circle(frame, (ball_x, ball_y), 12, (255, 255, 255), -1)
        elif video_type == "news_lecture":
            # White slide background with blue banner and black text bars
            frame[:, :] = (245, 245, 245)
            cv2.rectangle(frame, (0, 0), (width, 40), (200, 50, 0), -1)  # Top banner
            cv2.line(frame, (30, 80), (width - 30, 80), (40, 40, 40), 4)  # Heading
            cv2.line(frame, (30, 120), (width - 60, 120), (80, 80, 80), 3)  # Bullet 1
            cv2.line(frame, (30, 150), (width - 90, 150), (80, 80, 80), 3)  # Bullet 2
        elif video_type == "podcast":
            # Two static portrait boxes (interview/talking head)
            frame[:, :] = (50, 50, 60)
            cv2.rectangle(frame, (30, 50), (130, 180), (180, 150, 120), -1)  # Speaker 1
            cv2.rectangle(frame, (190, 50), (290, 180), (120, 150, 180), -1)  # Speaker 2
        elif video_type == "concert":
            # Very dark background with pulsing circular stage lights
            frame[:, :] = (10, 5, 15)
            radius = int(25 + 15 * np.sin(i * 0.8))
            cv2.circle(frame, (width // 2, height // 2), radius, (255, 0, 200), -1)
        elif video_type == "wildlife":
            # Earthy greens and browns with slow horizontal pan
            frame[:, :] = (60, 100, 50)
            cv2.rectangle(frame, (0, 160), (width, height), (30, 60, 20), -1)
            cv2.circle(frame, (80 + i * 2, 100), 20, (50, 80, 120), -1)
        elif video_type == "film_animation":
            # Saturated cartoon colors, alternating shots
            shot = (i // 5) % 2
            if shot == 0:
                frame[:, :] = (255, 105, 180)  # Bright pink
                cv2.circle(frame, (160, 120), 40, (0, 255, 255), -1)
            else:
                frame[:, :] = (0, 215, 255)  # Bright yellow/gold
                cv2.rectangle(frame, (80, 60), (240, 180), (255, 0, 100), -1)
        elif video_type == "surveillance":
            # High-angle monochrome / grayscale street corner
            gray_val = 120 + int(10 * np.sin(i * 0.1))
            frame[:, :] = (gray_val, gray_val, gray_val)
            cv2.line(frame, (0, height), (width, 100), (80, 80, 80), 2)  # Curb
        elif video_type == "meetings":
            # 2x2 grid of webcam attendees
            frame[:, :] = (40, 40, 40)
            cv2.rectangle(frame, (10, 10), (150, 110), (140, 130, 120), -1)
            cv2.rectangle(frame, (170, 10), (310, 110), (120, 140, 130), -1)
            cv2.rectangle(frame, (10, 130), (150, 230), (130, 120, 140), -1)
            cv2.rectangle(frame, (170, 130), (310, 230), (150, 140, 110), -1)
        elif video_type == "egocentric":
            # Tilting horizon with shaky camera motion
            tilt = int(15 * np.sin(i * 1.2))
            frame[:, :] = (100, 120, 140)
            cv2.line(frame, (0, 120 + tilt), (width, 120 - tilt), (60, 80, 100), 5)
        else:  # general
            frame[:, :] = (128, 128, 128)
            cv2.circle(frame, (width // 2, height // 2), 30, (200, 200, 200), -1)

        out.write(frame)

    out.release()
    return output_path


def create_synthetic_benchmark_dataset(
    base_dir: str = "artifacts/synthetic_dataset",
    samples_per_class: int = 3,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Creates a train/test synthetic video dataset across assignment classes.
    Returns: (train_paths, train_labels, test_paths, test_labels).
    """
    os.makedirs(base_dir, exist_ok=True)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    target_classes = [
        "gameplay",
        "sports",
        "news_lecture",
        "podcast",
        "concert",
        "wildlife",
        "film_animation",
        "surveillance",
        "meetings",
        "egocentric",
        "general",
    ]

    for cls_name in target_classes:
        # Create training samples
        for s in range(samples_per_class):
            dur = 1.0 + (s * 0.5)
            p = os.path.join(base_dir, "train", f"{cls_name}_{s}.mp4")
            generate_synthetic_video(p, video_type=cls_name, duration_s=dur)
            train_paths.append(p)
            train_labels.append(cls_name)

        # Create test samples
        for s in range(2):
            dur = 1.2 + (s * 0.4)
            p = os.path.join(base_dir, "test", f"{cls_name}_test_{s}.mp4")
            generate_synthetic_video(p, video_type=cls_name, duration_s=dur)
            test_paths.append(p)
            test_labels.append(cls_name)

    return train_paths, train_labels, test_paths, test_labels


def load_manifest_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """Loads video paths and labels from a CSV manifest."""
    paths, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["video_path"])
            labels.append(row["label"])
    return paths, labels
