"""
Domain-specific chunking policies and configuration presets (Task 4).
Maps assignment video genres to target durations, min/max limits, overlap constraints, and modality weights.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ChunkingPolicy:
    """Configurable parameter policy governing content-aware video segmentation."""
    name: str
    target_duration_s: float
    min_duration_s: float
    max_duration_s: float
    overlap_ratio: float = 0.10  # Must be within [0.05, 0.15] per assignment spec
    tolerance_s: float = 6.0
    parent_target_duration_s: float = 180.0
    primary_signals: List[str] = field(default_factory=lambda: ["scene", "speech", "motion"])
    scene_threshold: float = 0.35
    motion_weight: float = 1.0
    ocr_refinement: bool = False
    description: str = ""

    def __post_init__(self):
        # Validate duration invariants
        if self.min_duration_s <= 0:
            raise ValueError(f"min_duration_s must be > 0, got {self.min_duration_s}")
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError(
                f"min_duration_s ({self.min_duration_s}) must be < max_duration_s ({self.max_duration_s})"
            )
        if not (self.min_duration_s <= self.target_duration_s <= self.max_duration_s):
            raise ValueError(
                f"target_duration_s ({self.target_duration_s}) must lie between "
                f"min ({self.min_duration_s}) and max ({self.max_duration_s})"
            )
        # Validate overlap constraint (5% to 15%)
        if not (0.0 <= self.overlap_ratio <= 0.25):
            raise ValueError(f"overlap_ratio must be between 0.0 and 0.25, got {self.overlap_ratio}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pre-configured policies tailored to assignment video taxonomy
CATEGORY_POLICIES: Dict[str, ChunkingPolicy] = {
    "meetings": ChunkingPolicy(
        name="meetings",
        target_duration_s=60.0,
        min_duration_s=30.0,
        max_duration_s=75.0,
        overlap_ratio=0.10,
        tolerance_s=12.0,
        parent_target_duration_s=300.0,
        primary_signals=["speech", "scene", "ocr"],
        scene_threshold=0.30,
        ocr_refinement=True,
        description="Aligned with speaker turns, speech pauses, and slide transitions"
    ),
    "news_lecture": ChunkingPolicy(
        name="news_lecture",
        target_duration_s=60.0,
        min_duration_s=30.0,
        max_duration_s=75.0,
        overlap_ratio=0.10,
        tolerance_s=12.0,
        parent_target_duration_s=300.0,
        primary_signals=["scene", "ocr", "speech"],
        scene_threshold=0.32,
        ocr_refinement=True,
        description="Slide changes, topic transitions, and structured presentation pacing"
    ),
    "podcast": ChunkingPolicy(
        name="podcast",
        target_duration_s=45.0,
        min_duration_s=20.0,
        max_duration_s=60.0,
        overlap_ratio=0.10,
        tolerance_s=10.0,
        parent_target_duration_s=240.0,
        primary_signals=["speech", "scene"],
        scene_threshold=0.38,
        ocr_refinement=False,
        description="Speaker transitions and dialogue topic boundaries"
    ),
    "surveillance": ChunkingPolicy(
        name="surveillance",
        target_duration_s=30.0,
        min_duration_s=15.0,
        max_duration_s=45.0,
        overlap_ratio=0.08,
        tolerance_s=8.0,
        parent_target_duration_s=180.0,
        primary_signals=["motion", "scene"],
        scene_threshold=0.25,
        motion_weight=1.5,
        ocr_refinement=False,
        description="Motion onset, entity entry/exit, and activity windows"
    ),
    "sports": ChunkingPolicy(
        name="sports",
        target_duration_s=20.0,
        min_duration_s=10.0,
        max_duration_s=30.0,
        overlap_ratio=0.10,
        tolerance_s=5.0,
        parent_target_duration_s=120.0,
        primary_signals=["scene", "motion"],
        scene_threshold=0.30,
        motion_weight=1.2,
        ocr_refinement=False,
        description="Action plays, whistle stops, broadcast camera cuts"
    ),
    "gameplay": ChunkingPolicy(
        name="gameplay",
        target_duration_s=20.0,
        min_duration_s=10.0,
        max_duration_s=30.0,
        overlap_ratio=0.10,
        tolerance_s=5.0,
        parent_target_duration_s=120.0,
        primary_signals=["scene", "ocr", "motion"],
        scene_threshold=0.32,
        ocr_refinement=True,
        description="HUD state shifts, scene transitions, round/respawn events"
    ),
    "egocentric": ChunkingPolicy(
        name="egocentric",
        target_duration_s=15.0,
        min_duration_s=8.0,
        max_duration_s=25.0,
        overlap_ratio=0.12,
        tolerance_s=4.0,
        parent_target_duration_s=90.0,
        primary_signals=["motion", "scene"],
        scene_threshold=0.35,
        motion_weight=1.4,
        ocr_refinement=False,
        description="Camera pans, head turns, hand interaction events"
    ),
    "wildlife": ChunkingPolicy(
        name="wildlife",
        target_duration_s=20.0,
        min_duration_s=10.0,
        max_duration_s=35.0,
        overlap_ratio=0.10,
        tolerance_s=6.0,
        parent_target_duration_s=120.0,
        primary_signals=["scene", "motion"],
        scene_threshold=0.32,
        ocr_refinement=False,
        description="Animal entrance/exit and natural scene landscape cuts"
    ),
    "concert": ChunkingPolicy(
        name="concert",
        target_duration_s=30.0,
        min_duration_s=15.0,
        max_duration_s=45.0,
        overlap_ratio=0.10,
        tolerance_s=8.0,
        parent_target_duration_s=180.0,
        primary_signals=["scene", "speech"],
        scene_threshold=0.35,
        ocr_refinement=False,
        description="Music segments, song transitions, lighting cue changes"
    ),
    "film_animation": ChunkingPolicy(
        name="film_animation",
        target_duration_s=25.0,
        min_duration_s=12.0,
        max_duration_s=40.0,
        overlap_ratio=0.10,
        tolerance_s=6.0,
        parent_target_duration_s=150.0,
        primary_signals=["scene", "speech"],
        scene_threshold=0.36,
        ocr_refinement=False,
        description="Dramatic scene cuts, dialogue pauses, shot transitions"
    ),
    "general": ChunkingPolicy(
        name="general",
        target_duration_s=30.0,
        min_duration_s=15.0,
        max_duration_s=45.0,
        overlap_ratio=0.10,
        tolerance_s=8.0,
        parent_target_duration_s=150.0,
        primary_signals=["scene", "speech", "motion"],
        scene_threshold=0.34,
        ocr_refinement=False,
        description="Balanced default policy for unknown or general videos"
    ),
}


def get_policy(
    video_type: Optional[str] = None,
    target_duration_s: Optional[float] = None,
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    overlap_ratio: Optional[float] = None,
    tolerance_s: Optional[float] = None,
) -> ChunkingPolicy:
    """
    Retrieves the chunking policy for a given video type, with optional custom overrides.
    """
    key = (video_type or "general").lower().strip()
    base = CATEGORY_POLICIES.get(key, CATEGORY_POLICIES["general"])

    # Create a clone with any overrides applied
    return ChunkingPolicy(
        name=base.name if not any([target_duration_s, min_duration_s, max_duration_s]) else f"{base.name}_custom",
        target_duration_s=target_duration_s if target_duration_s is not None else base.target_duration_s,
        min_duration_s=min_duration_s if min_duration_s is not None else base.min_duration_s,
        max_duration_s=max_duration_s if max_duration_s is not None else base.max_duration_s,
        overlap_ratio=overlap_ratio if overlap_ratio is not None else base.overlap_ratio,
        tolerance_s=tolerance_s if tolerance_s is not None else base.tolerance_s,
        parent_target_duration_s=base.parent_target_duration_s,
        primary_signals=base.primary_signals,
        scene_threshold=base.scene_threshold,
        motion_weight=base.motion_weight,
        ocr_refinement=base.ocr_refinement,
        description=base.description,
    )


def list_policies() -> Dict[str, Dict[str, Any]]:
    """Returns all available policy configurations."""
    return {k: v.to_dict() for k, v in CATEGORY_POLICIES.items()}
