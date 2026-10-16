"""
Dataset Preparation, Labeling Strategy, and Anti-Leakage Split Partitioning for Task 2.
Inspects VideoMME questions, applies documented deterministic labeling rules,
synthesizes out-of-domain challenge queries, and creates video-grouped train/val/test splits.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Canonical intent categories
INTENT_CLASSES = ["whole_video", "frame_level", "clip_level"]

# Deterministic task_type priors based on minimum sufficient temporal evidence
TASK_TYPE_TO_INTENT = {
    "Information Synopsis": "whole_video",
    "OCR Problems": "frame_level",
    "Attribute Perception": "frame_level",
    "Action Recognition": "clip_level",
    "Action Reasoning": "clip_level",
    "Temporal Reasoning": "clip_level",
    "Temporal Perception": "clip_level",
    "Counting Problem": "clip_level",
    "Spatial Reasoning": "frame_level",
    "Spatial Perception": "frame_level",
    "Object Recognition": "frame_level",
    "Object Reasoning": "clip_level",
}


def label_videomme_question(item: Dict[str, Any]) -> str:
    """Assign reproducible intent label to VideoMME question based on task_type and lexical cues."""
    q_text = item.get("question", "").lower()
    task_type = item.get("task_type", "")

    # Priority 1: High-confidence lexical overrides
    if any(k in q_text for k in ["what does the video depict", "focus of the video", "main topic", "summarize", "entire video", "what is this video about"]):
        return "whole_video"
    if any(k in q_text for k in ["scoreboard", "what text", "what is written", "what word", "font", "sign say", "slide"]):
        return "frame_level"
    if any(k in q_text for k in ["how did", "when does", "what happens after", "moment when", "action does", "how do the"]):
        return "clip_level"

    # Priority 2: Task-type mapping
    return TASK_TYPE_TO_INTENT.get(task_type, "clip_level")


def generate_domain_queries() -> List[Dict[str, Any]]:
    """Synthesize structured in-domain and out-of-domain query templates across all video genres."""
    queries = []

    # 1. whole_video intent examples (summaries, overarching themes, global narrative)
    whole_templates = [
        "Can you summarize the main discussion points of this corporate meeting?",
        "Provide a comprehensive synopsis of the entire football match.",
        "What is the overall topic presented in this university lecture?",
        "Summarize the storyline and key events of this animation short.",
        "Give me an executive summary of this quarterly earnings call.",
        "What was the primary conclusion reached by the panel at the end of the video?",
        "Provide a high-level overview of everything that occurred in this CCTV stream.",
        "What is the primary message or moral conveyed by this short film?",
        "Can you outline all the chapters covered in this programming tutorial?",
        "What was the general sentiment and outcome of the press conference?",
        "Summarize the entire gameplay playthrough from start to finish.",
        "Give me a bullet-point digest of this podcast episode.",
        "What are the major themes discussed across this entire documentary?",
        "Explain the whole video and its main takeaways.",
        "What is the full narrative arc depicted in this concert recording?"
    ]
    for i, t in enumerate(whole_templates):
        queries.append({
            "query_id": f"syn_wv_{i}",
            "video_id": f"synthetic_vid_wv_{i % 5}",
            "query": t,
            "intent": "whole_video",
            "source": "synthetic_domain_template"
        })

    # 2. frame_level intent examples (OCR, pinpoint visual detail, single moment)
    frame_templates = [
        "What text is written on the whiteboard at 14:22?",
        "What is the current score displayed on the top-right scoreboard?",
        "What color shirt is the speaker wearing in the first slide?",
        "What is the brand logo on the laptop screen in the opening frame?",
        "Identify the license plate number of the red car shown in the driveway.",
        "What does the traffic sign say on the right side of the street?",
        "What error message is visible in the console terminal at 08:31?",
        "What book title is the professor holding in the video thumbnail?",
        "What icon is shown next to the user's name on the screen?",
        "Which stock ticker symbol is shown on the financial ticker banner?",
        "What color are the player's gloves during the opening sequence?",
        "What is written on the runner's bib number at the finish line?",
        "What phone model is advertised on the billboard at 02:10?",
        "What is the date stamped at the bottom of the surveillance footage?",
        "What badge is pinned to the security officer's uniform?"
    ]
    for i, t in enumerate(frame_templates):
        queries.append({
            "query_id": f"syn_fl_{i}",
            "video_id": f"synthetic_vid_fl_{i % 5}",
            "query": t,
            "intent": "frame_level",
            "source": "synthetic_domain_template"
        })

    # 3. clip_level intent examples (temporal events, actions, continuous intervals)
    clip_templates = [
        "Show me the moment the referee issues a red card to the player.",
        "Find the segment where the speaker explains gradient descent optimization.",
        "When does the lion begin chasing the gazelle across the river?",
        "Show the sequence where the musician plays the guitar solo.",
        "Find the clip where the presenter demonstrates the software installation steps.",
        "Where does the suspect enter through the back door in the CCTV footage?",
        "Play the highlight where the striker scores from outside the penalty box.",
        "When did the argument break out during the city council meeting?",
        "Show the part where the chef adds garlic and olive oil to the pan.",
        "Find the interval where the athlete falls and gets back up.",
        "Show the scene where the two characters reconcile in the park.",
        "When does the rocket booster detach from the main cabin?",
        "Find the gameplay clip where the player defeats the dungeon boss.",
        "Show me where the car spins out on the wet road.",
        "When does the surgeon perform the incision on the tissue?"
    ]
    for i, t in enumerate(clip_templates):
        queries.append({
            "query_id": f"syn_cl_{i}",
            "video_id": f"synthetic_vid_cl_{i % 5}",
            "query": t,
            "intent": "clip_level",
            "source": "synthetic_domain_template"
        })

    # 4. Ambiguous / needs_review challenge queries (underspecified or conflicting intents)
    ambiguous_templates = [
        "the dog",
        "show stuff",
        "what happened and what was on the slide and give me everything",
        "video",
        "database tutorial details",
        "tell me about it",
        "find things",
        "play it",
        "check the presentation overview and the slide text at the same time",
        "something cool"
    ]
    for i, t in enumerate(ambiguous_templates):
        queries.append({
            "query_id": f"syn_amb_{i}",
            "video_id": f"synthetic_vid_amb_{i % 3}",
            "query": t,
            "intent": "ambiguous",
            "source": "synthetic_ambiguous_challenge"
        })

    return queries


def build_intent_dataset(
    videomme_path: str = "test.jsonl",
    output_path: str = "data/intent_dataset.json",
    seed: int = 42
) -> Dict[str, Any]:
    """Construct auditable, GroupKFold train/val/test intent splits preventing video leakage."""
    rng = np.random.RandomState(seed)
    items = []

    # 1. Ingest VideoMME questions if available
    if os.path.exists(videomme_path):
        with open(videomme_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    intent = label_videomme_question(obj)
                    items.append({
                        "query_id": obj.get("question_id", f"vmme_{len(items)}"),
                        "video_id": obj.get("video_id", "unknown_video"),
                        "query": obj.get("question", ""),
                        "intent": intent,
                        "task_type": obj.get("task_type", ""),
                        "source": "videomme_sample10"
                    })
                except Exception as e:
                    logger.debug("Failed parsing VideoMME line: %s", e)

    # 2. Append domain and ambiguous query templates
    synthetic_items = generate_domain_queries()
    items.extend(synthetic_items)

    # Filter out empty queries
    items = [x for x in items if x["query"].strip()]

    # 3. Group-aware partitioning by video_id to prevent data leakage
    # Separate unambiguous intent samples for model training from ambiguous challenge samples
    clean_items = [x for x in items if x["intent"] in INTENT_CLASSES]
    ambiguous_items = [x for x in items if x["intent"] not in INTENT_CLASSES]

    unique_videos = sorted(list(set(x["video_id"] for x in clean_items)))
    rng.shuffle(unique_videos)

    n_total_vids = len(unique_videos)
    n_train = int(0.60 * n_total_vids)
    n_val = int(0.20 * n_total_vids)

    train_vids = set(unique_videos[:n_train])
    val_vids = set(unique_videos[n_train:n_train + n_val])
    test_vids = set(unique_videos[n_train + n_val:])

    train_data = [x for x in clean_items if x["video_id"] in train_vids]
    val_data = [x for x in clean_items if x["video_id"] in val_vids]
    test_data = [x for x in clean_items if x["video_id"] in test_vids]

    # Ambiguous challenge queries are appended to test split for evaluation
    test_data_with_ambiguous = list(test_data) + list(ambiguous_items)

    dataset_manifest = {
        "metadata": {
            "version": "1.2.0",
            "seed": seed,
            "total_samples": len(items),
            "clean_samples": len(clean_items),
            "ambiguous_samples": len(ambiguous_items),
            "num_videos": n_total_vids,
            "splits": {
                "train_count": len(train_data),
                "val_count": len(val_data),
                "test_count": len(test_data),
                "test_challenge_count": len(test_data_with_ambiguous)
            },
            "class_distribution_train": {c: sum(1 for x in train_data if x["intent"] == c) for c in INTENT_CLASSES},
            "class_distribution_val": {c: sum(1 for x in val_data if x["intent"] == c) for c in INTENT_CLASSES},
            "class_distribution_test": {c: sum(1 for x in test_data if x["intent"] == c) for c in INTENT_CLASSES}
        },
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "test_with_ambiguous": test_data_with_ambiguous
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    logger.info("Saved intent dataset manifest to %s (Train=%d, Val=%d, Test=%d)",
                output_path, len(train_data), len(val_data), len(test_data))
    return dataset_manifest
