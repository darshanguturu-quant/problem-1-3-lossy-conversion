# System Architecture Specification
## AIntropy Video Intelligence Platform

| Document Attribute | Details |
|---|---|
| **System** | AIntropy End-to-End Multimodal Video Intelligence Architecture |
| **Document Version** | 1.1.0-HYBRID-SPEC |
| **Author** | Senior ML Engineer & Software Architect |
| **Status** | Approved Specification Layer (Pre-Implementation) |
| **Architectural Focus** | Cascaded Hybrid Classification, Modular Decoupling, Lightweight CPU-First Inference |

---

## 1. System Overview & Architectural Vision

The AIntropy Video Intelligence Platform is designed to bridge the gap between large-scale raw video archives and downstream multimodal retrieval-augmented generation (Video-RAG). Rather than passing high-resolution, unsegmented video streams directly into expensive Vision-Language Models (VLMs), this platform acts as an intelligent, cost-optimizing perception and routing layer.

### Core Architectural Tenets
1. **Cascaded Hybrid Perception (Task 1)**: Ingestion classification is executed via a two-stage cascade:
   - *Stage 1*: An ultra-lightweight metadata and coarse statistical model acts as a high-speed gate, resolving high-confidence videos in under 15 ms.
   - *Stage 2*: High-capacity CLIP embeddings with adaptive frame sampling resolve ambiguous, high-entropy videos only when triggered by a confidence threshold ($\theta_{\text{escalate}}$).
2. **Separation of Planning and Execution**: Chunking plans, compression policies, and query routes are generated as lightweight metadata artifacts prior to expensive media operations.
3. **Cascaded Hybrid Routing (Task 1 & Task 2)**: Both video format classification (Task 1) and query intent classification (Task 2) operate via two-stage cascades. Stage 1 lightweight gates resolve 75%–85% of unambiguous requests in sub-milliseconds on CPU, escalating only uncertain cases to Stage 2 high-capacity semantic models, achieving deep-learning accuracy at linear-model speeds.
4. **Linear-Time Complexity $O(D)$**: Ingestion and chunking algorithms scale linearly with video duration, avoiding quadratic all-pairs frame comparisons.
5. **Idempotent Content-Addressing**: Media derivatives and chunks are indexed by `sha256(content) + policy_version`, ensuring safe distributed retries and deduplication.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion_Stage ["1. Ingestion & Pre-Processing"]
        RawVideo["Raw Video File / Stream"] --> Probe["Metadata Prober (ffprobe)"]
        Probe --> MetaFeatures["Metadata: Duration, Bitrate, Codec, FPS, Audio"]
        RawVideo --> FastSampler["Fast Keyframe Sampler (2-4 frames)"]
        FastSampler --> CoarseStats["Coarse Color Histograms & Pixel Diffs"]
    end

    subgraph Task1_Cascade ["2. Task 1: Cascaded Hybrid Video Type Classifier"]
        MetaFeatures & CoarseStats --> Stage1Classifier["Stage 1: Lightweight Classifier (HistGBDT / LR)"]
        Stage1Classifier --> ConfidenceCheck{"p1 >= θ_escalate (e.g. 0.85)?"}
        
        ConfidenceCheck -->|YES (High Conf ~75-80%)| EarlyExit["Direct Stage 1 Prediction"]
        ConfidenceCheck -->|NO (Uncertain ~20-25%)| Stage2Trigger["Escalate to Stage 2"]
        
        Stage2Trigger --> AdaptiveSampler["Adaptive Frame Sampler (4, 8, or 12 frames)"]
        RawVideo --> AdaptiveSampler
        AdaptiveSampler --> CLIPEncoder["CLIP Visual Encoder (ViT-B/32 or RN50)"]
        CLIPEncoder --> Stage2Head["Calibrated Linear Classification Head"]
        Stage2Head --> Stage2Result["Stage 2 Refined Prediction"]
        
        EarlyExit & Stage2Result --> FinalVideoClass["Resolved Video Class + Chunking Strategy"]
    end

    subgraph Conversion_Stage ["3. Task 3: Lossy Conversion & Derivation Gating"]
        FinalVideoClass & MetaFeatures --> LossyGate["Lossy Conversion Break-Even Gate"]
        LossyGate -->|Passes Gate| Transcoder["FFmpeg Transcoder (H.264/H.265 Profile)"]
        LossyGate -->|Fails Gate / Ineligible| OriginalKeeper["Retain Immutable Original"]
        Transcoder --> DerivativeStore["Derived Video Asset (.mp4)"]
        OriginalKeeper --> MasterStore["Master Storage Pool"]
    end

    subgraph Chunking_Stage ["4. Task 4: Content-Aware Intelligent Chunking"]
        FinalVideoClass --> StrategySelector["Strategy Policy Selector"]
        DerivativeStore & MasterStore --> SignalExtractor["Signal Extractor (Scene Cuts, VAD, Motion)"]
        SignalExtractor & StrategySelector --> ChunkEngine["Intelligent Video Chunking Engine"]
        ChunkEngine --> ChunkManifest["Hierarchical Chunk Manifest (JSON)"]
    end

    subgraph Query_Routing_Stage ["5. Task 2: Cascaded Hybrid Intent Router"]
        UserQuery["Natural Language Query"] --> QueryFeatures["Feature Extractor: TF-IDF (1-2 word, 3-5 char) + 34 Cues"]
        QueryFeatures --> Stage1Intent["Stage 1: Calibrated Logistic Regression"]
        Stage1Intent --> IntentCheck{"p1 >= θ_escalate (0.82) & margin >= Δ (0.30)?"}
        
        IntentCheck -->|YES (Unambiguous ~80-85%)| EarlyIntentExit["Direct Stage 1 Posterior p1"]
        IntentCheck -->|NO (Uncertain ~15-20%)| Stage2IntentTrigger["Escalate to Stage 2"]
        
        Stage2IntentTrigger --> MiniLMEncoder["Dense Sentence Embeddings (all-MiniLM-L6-v2)"]
        MiniLMEncoder --> Stage2IntentHead["Lightweight Calibrated Head (LR / MLP)"]
        Stage2IntentHead --> Stage2IntentPosterior["Stage 2 Posterior p2 & Blending"]
        
        EarlyIntentExit & Stage2IntentPosterior --> FusedIntentProb["Fused Posterior p(t|q)"]
        FusedIntentProb --> AmbiguityGuard{"max(p) < θ_ambiguity (0.45)?"}
        AmbiguityGuard -->|YES (Ambiguous Query)| AmbiguityFallback["Safe Fallback: clip_level"]
        AmbiguityGuard -->|NO| BayesDecision["Bayes-Optimal Utility Decision Layer"]
        
        AmbiguityFallback --> ClipIndex
        BayesDecision -->|frame_level| FrameIndex["Keyframe & OCR Search"]
        BayesDecision -->|clip_level| ClipIndex["Temporal Segment Search"]
        BayesDecision -->|whole_video| SummaryIndex["Whole-Video Document Synopsis Search"]
    end

    subgraph Benchmarking_Stage ["6. Task 5: Cross-Dataset Evaluation"]
        ChunkManifest & UserQuery --> BenchmarkCLI["Retrieval Benchmark Builder"]
        DatasetA["Dataset A: Meetings/Lectures"] & DatasetB["Dataset B: Sports/Gameplay"] & DatasetC["Dataset C: Surveillance/Ambient"] --> BenchmarkCLI
        VideoMME["VideoMME Reference Sample"] --> BenchmarkCLI
        BenchmarkCLI --> Scorecard["Evaluation Scorecard (Recall@K, MRR, nDCG@K, Regret)"]
    end
```

---

## 3. Component Responsibilities & Module Decomposition

### 3.1 Module 1: Ingestion & Profiler (`src/profiler.py`)
- **Responsibilities**:
  - Run non-blocking `ffprobe` subprocesses to capture stream parameters (duration, audio channels, video codec, container, pixel format, frame count, display aspect ratio).
  - Extract 2–4 fast sparse keyframes using `-ss` fast seeking for coarse color histogram statistics and inter-frame differences.
  - Return a strongly typed `VideoProfile` dataclass.
- **Interface**:
  ```python
  class VideoProfiler:
      def probe_metadata(self, video_path: str) -> VideoMetadata: ...
      def extract_coarse_signals(self, video_path: str) -> CoarseVideoSignals: ...
  ```

### 3.2 Module 2: Cascaded Hybrid Video Type Classifier (`src/classifier/`) — Task 1
- **Architecture & Responsibilities**:
  - **Stage 1 (`src/classifier/stage1.py`)**:
    - Input: Normalized metadata vector (log duration, bitrate, FPS, aspect ratio, audio presence, audio sample rate) + coarse 3-channel color histogram mean/std and frame difference metrics.
    - Model: Calibrated Histogram Gradient Boosting Classifier (`HistGradientBoostingClassifier`) or class-balanced Logistic Regression.
    - Output: Class probabilities $\mathbf{p}_1$ and top confidence $p_{\max} = \max(\mathbf{p}_1)$.
  - **Confidence Gate (`src/classifier/cascade.py`)**:
    - If $p_{\max} \ge \theta_{\text{escalate}}$ (configurable threshold, default 0.85): Assign prediction directly without invoking neural networks.
    - If $p_{\max} < \theta_{\text{escalate}}$: Escalate to Stage 2.
    - Tracks and reports the **escalation rate**:
      $$\text{EscalationRate} = \frac{N_{\text{escalated}}}{N_{\text{total}}}$$
  - **Stage 2 (`src/classifier/stage2.py`)**:
    - **Adaptive Frame Sampler**: Samples frames dynamically:
      - Duration $\le 300$ s: 4 frames uniformly sampled.
      - $300\text{ s} < \text{Duration} \le 1800$ s: 8 frames uniformly sampled.
      - Duration $> 1800$ s: 12 frames paced by scene-change candidate points.
    - Visual Encoder: Pretrained CLIP visual backbone (`ViT-B/32` or compact `RN50`).
    - Aggregation: Mean-pooled frame embeddings.
    - Head: Calibrated linear classification head (or zero-shot text-prompt similarity + linear residual).
    - Fallback: If Stage 2 confidence is below $\theta_{\text{fallback}}$ (e.g. 0.40), assign `general` category.
- **Interface**:
  ```python
  class CascadedVideoClassifier:
      def __init__(self, stage1_model, stage2_model, threshold: float = 0.85): ...
      def predict(self, video_path: str) -> ClassificationResult:
          # Returns:
          # {
          #   "video_id": str,
          #   "predicted_class": str,
          #   "confidence": float,
          #   "resolved_at_stage": 1 or 2,
          #   "chunking_strategy": str
          # }
  ```

### 3.3 Module 3: Lossy Conversion & ROI Engine (`src/transcoder/`) — Task 3
- **Responsibilities**:
  - Evaluate incoming video metadata against the quantitative break-even formula.
  - Bypass re-encoding for high-efficiency streams (< 80 MB/min H.264/H.265/AV1) or short clips where $T_{\text{conv}} > \Delta T_{\text{IO}}$.
  - Construct parameterized FFmpeg commands using vetted profiles (H.264 CRF 23 `veryfast` baseline; H.265 CRF 26 storage profile).
  - Execute conversion into secondary storage derivatives without modifying source assets.
- **Interface**:
  ```python
  class LossyConversionEngine:
      def calculate_roi(self, meta: VideoMetadata, params: ROIParameters) -> ROIEstimate: ...
      def generate_ffmpeg_cmd(self, input_path: str, output_path: str, profile: str) -> List[str]: ...
      def execute_transcode(self, cmd: List[str]) -> TranscodeResult: ...
  ```

### 3.4 Module 4: Intelligent Video Chunker (`src/chunker/`) — Task 4
- **Responsibilities**:
  - Accept discrete signal candidate arrays: scene cuts (visual frame diff peaks), speech activity intervals (VAD/ASR), and slide/motion transitions.
  - Apply category-specific policies defining base target window $W$, overlap $\Delta$, tolerance $\tau$, and minimum/maximum window clamping $[T_{\min}, T_{\max}]$.
  - Assemble contiguous temporal intervals adhering to parent-child hierarchies.
  - Output standard JSON manifests ready for downstream vector indexing.
- **Interface**:
  ```python
  class IntelligentChunker:
      def load_policy(self, video_type: str) -> ChunkingPolicy: ...
      def segment(self, video_id: str, duration_s: float, cues: SignalCues, policy: ChunkingPolicy) -> ChunkManifest: ...
  ```

### 3.5 Module 5: Cascaded Hybrid User Intent Classifier & Router (`src/intent/`) — Task 2
- **Architecture & Responsibilities**:
  - **Intent Taxonomy (Retrieval Granularities)**:
    - `whole_video`: Video-level synopsis, topic synthesis, overarching themes, global narrative. Routes to whole-video document index.
    - `frame_level`: Single-moment visual, OCR, timestamp-specific, or fine object search. Routes to keyframe image/OCR index.
    - `clip_level`: Temporal event, multi-second action sequence, speech dialogue segment. Routes to temporal chunk index.
  - **Stage 1 (`src/intent/stage1.py`)**:
    - *Preprocessing*: NFKC normalization, lowercase standardizing, punctuation handling, whitespace collapse.
    - *Feature Vector*: Sparse word (1-2) + char (3-5) n-gram TF-IDF concatenated with 34 dense grammatical cue extractors (temporal timestamps `\b\d{1,2}:\d{2}\b`, WH-interrogatives, action verbs, frame OCR cues, synopsis tokens, query length).
    - *Classifier*: Calibrated Multinomial Logistic Regression (`LogisticRegression(C=1.0, max_iter=500)`) with temperature scaling $T_1$.
    - *Confidence Gating*: Evaluates top probability $p_{\max}^{(1)} = \max_c p_1(c)$ and probability margin $m^{(1)} = p_{(1)}^{(1)} - p_{(2)}^{(1)}$. If $p_{\max}^{(1)} \ge \theta_{\text{escalate\_intent}}$ (provisional 0.82) AND $m^{(1)} \ge \Delta_{\text{margin}}$ (provisional 0.30), return prediction immediately (< 0.2 ms CPU).
  - **Stage 2 (`src/intent/stage2.py`)**:
    - *Triggered conditionally* on uncertain queries ($p_{\max}^{(1)} < \theta_{\text{escalate\_intent}}$ or $m^{(1)} < \Delta_{\text{margin}}$).
    - *Semantic Encoder*: Compact sentence embedding model (`all-MiniLM-L6-v2`, 22.7M parameters, 384-dimensional dense semantic embedding vector).
    - *Head*: Calibrated lightweight classifier (class-balanced Logistic Regression or 2-layer MLP head with temperature scaling $T_2$).
    - *Feature Concat & Fusion*: Concatenates 384-d semantic embedding with normalized Stage 1 probability vector $\mathbf{p}_1$ and 12 structural cues (399-dim vector). Final posterior blended via: $\mathbf{p}_{\text{final}} = \alpha \cdot \mathbf{p}_2 + (1 - \alpha) \cdot \mathbf{p}_1$ ($\alpha = 0.85$).
  - **Cascade Router & Ambiguity Layer (`src/intent/cascade.py`)**:
    - Tracks and reports **escalation rate**: $\rho_{\text{intent}} = N_{\text{escalated}} / N_{\text{total}}$.
    - *Bayes-Optimal Decision Layer*: Maximizes expected utility:
      $$\hat{r} = \arg\max_r \sum_{t \in \{\text{whole\_video}, \text{frame\_level}, \text{clip\_level}\}} p_{\text{final}}(t|q) \cdot U(t, r)$$
      where $U(t, r) = A(t, r) - \lambda \cdot C(r)$. Eliminates catastrophic under-routing ($t=\text{whole\_video} \to r=\text{frame\_level}$) where $A = 0.0$.
    - *Ambiguity Fallback*: If $\max(\mathbf{p}_{\text{final}}) < \theta_{\text{ambiguity}}$ (provisional 0.45), safely routes to `clip_level` (the median-granularity safe middle ground).
  - **Standalone Runtime (`src/intent/route.py`)**: Zero-dependency runtime module for low-overhead production query serving.
  - **Baseline Suite (`src/intent/baselines.py`)**: Benchmarking harness evaluating Rule-Based Regex, TF-IDF alone, Sentence-Embedding alone, Compact Transformer, Zero-Shot NLI, and Cloud LLM baselines.
- **Interface**:
  ```python
  from dataclasses import dataclass
  from typing import Dict, List, Literal, Optional

  IntentClass = Literal["whole_video", "frame_level", "clip_level"]

  @dataclass(frozen=True)
  class IntentRoutingDecision:
      query: str
      predicted_intent: IntentClass
      confidence: float
      stage_resolved: int  # 1 or 2
      escalated_to_stage2: bool
      probabilities: Dict[IntentClass, float]
      stage1_confidence: float
      stage1_margin: float
      utility_score: float
      routing_target: Literal["summary_index", "frame_index", "clip_index"]

  class CascadedIntentRouter:
      def __init__(
          self,
          stage1_model,
          stage2_model=None,
          escalate_threshold: float = 0.82,
          margin_threshold: float = 0.30,
          ambiguity_threshold: float = 0.45,
      ): ...
      def extract_stage1_features(self, query: str) -> np.ndarray: ...
      def extract_sentence_embeddings(self, query: str) -> np.ndarray: ...
      def route_query(self, query: str) -> IntentRoutingDecision: ...
      def benchmark_baselines(self, test_queries: List[Dict]) -> Dict[str, Dict]: ...
  ```

### 3.6 Module 6: Retrieval Benchmark Builder & Evaluator (`src/benchmark/`) — Task 5
- **Responsibilities**:
  - Parse heterogeneous video manifests across three production datasets and VideoMME.
  - Generate stratified evaluation splits by genre, duration, difficulty, and query intent.
  - Formulate retrieval ground-truth pairs and hard-negative distractors.
  - Ingest retrieval prediction outputs and compute standardized ranking metrics: Recall@K ($K \in \{1, 5, 10, 20\}$), MRR, nDCG@K, Context Precision, and Evidence Temporal IoU.
  - Evaluate domain transfer across in-domain, cross-domain zero-shot, and mixed-corpus modes.
- **Interface**:
  ```python
  class RetrievalBenchmarkBuilder:
      def build_benchmark(self, manifest_path: str, config: BenchmarkConfig) -> BenchmarkManifest: ...
      def evaluate_predictions(self, benchmark: BenchmarkManifest, predictions: Dict[str, List[str]]) -> EvaluationScorecard: ...
  ```

---

## 4. Latency & Resource Budget Across Cascade Stages

### 4.1 Task 1: Video Type Classifier Cascade
```
========================================================================================
Stage                Execution Path     CPU Latency (P50)   Memory (RAM)    Model Size
========================================================================================
Stage 1 Gate         100% of videos     12 ms               ~80 MB          ~1.5 MB
Stage 2 Refinement   ~20% of videos     140 ms              ~550 MB         ~140 MB
Overall Cascaded     Weighted Average   ~38 ms              ~550 MB (peak)  ~141.5 MB
========================================================================================
Comparison (Task 1 Alternatives):
Metadata-Only        100% of videos     10 ms               ~70 MB          ~1.0 MB
CLIP-Only            100% of videos     165 ms              ~550 MB         ~140 MB
MobileNetV3-Only     100% of videos     45 ms               ~150 MB         ~11.5 MB
========================================================================================
```

### 4.2 Task 2: User Intent Classifier Cascade
```
========================================================================================
Stage                Execution Path     CPU Latency (P50)   Memory (RAM)    Model Size
========================================================================================
Stage 1 Gate (TF-IDF)100% of queries    0.15 ms             ~35 MB          ~180 KB
Stage 2 Refinement   ~15% of queries    18.5 ms             ~90 MB          ~88.0 MB
Overall Cascaded     Weighted Average   ~0.25 ms            ~90 MB (peak)   ~88.2 MB
========================================================================================
Comparison (Task 2 Alternatives):
Rule-Based Regex     100% of queries    0.05 ms             ~10 MB          < 1 KB
Sentence-Embedding   100% of queries    18.5 ms             ~90 MB          ~88.0 MB
Compact-Transformer  100% of queries    42.0 ms             ~180 MB         ~250.0 MB
Zero-Shot NLI        100% of queries    160.0 ms            ~450 MB         ~380.0 MB
Cloud LLM API        100% of queries    450.0 ms            External        External API
========================================================================================
```
*Note: Latency and memory figures represent provisional engineering estimates on 4-core standard CPU hardware and are pending empirical benchmarking.*

---

## 5. Data Flow & Data Contracts

### 5.1 Cascaded Classification Output Schema (`classification_result.json`)
```json
{
  "video_id": "vid_4812a0f8",
  "predicted_class": "sports",
  "confidence": 0.892,
  "stage_resolved": 1,
  "escalated_to_stage2": false,
  "stage1_probabilities": {
    "sports": 0.892,
    "gameplay": 0.054,
    "general": 0.054
  },
  "stage2_details": null,
  "chunking_strategy": "event_and_play",
  "sampling_policy": "adaptive_motion_and_action"
}
```

### 5.2 Escalated Classification Output Schema (`classification_escalated.json`)
```json
{
  "video_id": "vid_9941b7c2",
  "predicted_class": "podcast",
  "confidence": 0.835,
  "stage_resolved": 2,
  "escalated_to_stage2": true,
  "stage1_confidence": 0.621,
  "stage2_details": {
    "frames_sampled": 8,
    "clip_backbone": "ViT-B/32",
    "top_candidates": [
      {"class": "podcast", "score": 0.835},
      {"class": "news_lecture", "score": 0.120}
    ]
  },
  "chunking_strategy": "speech_turn_and_topic",
  "sampling_policy": "speech_dialogue_continuity"
}
```

### 5.3 Cascaded Intent Routing Output Schema (`intent_result.json` — Stage 1 Resolved)
```json
{
  "query": "when does the referee issue the yellow card?",
  "predicted_intent": "clip_level",
  "confidence": 0.895,
  "stage_resolved": 1,
  "escalated_to_stage2": false,
  "probabilities": {
    "whole_video": 0.032,
    "frame_level": 0.073,
    "clip_level": 0.895
  },
  "stage1_metrics": {
    "confidence": 0.895,
    "margin": 0.822,
    "escalation_threshold": 0.82,
    "margin_threshold": 0.30
  },
  "stage2_details": null,
  "utility_score": 0.945,
  "routing_target": "clip_index"
}
```

### 5.4 Escalated Intent Routing Output Schema (`intent_escalated.json` — Stage 2 Resolved)
```json
{
  "query": "summarize key speaker takeaways and check if revenue slide was presented",
  "predicted_intent": "clip_level",
  "confidence": 0.784,
  "stage_resolved": 2,
  "escalated_to_stage2": true,
  "probabilities": {
    "whole_video": 0.114,
    "frame_level": 0.102,
    "clip_level": 0.784
  },
  "stage1_metrics": {
    "confidence": 0.521,
    "margin": 0.082,
    "escalation_threshold": 0.82,
    "margin_threshold": 0.30
  },
  "stage2_details": {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "raw_stage2_confidence": 0.812,
    "top_candidates": [
      {"intent": "clip_level", "score": 0.784},
      {"intent": "whole_video", "score": 0.114}
    ]
  },
  "utility_score": 0.875,
  "routing_target": "clip_index"
}
```

---

## 6. Security, Isolation, and Data Integrity

1. **Subprocess Sanitization**: All FFmpeg and ffprobe calls pass strictly parameterized argument lists; no `shell=True` execution is permitted, eliminating command injection risks.
2. **File Path Sandboxing**: Input and output file paths are resolved against validated base directories; relative directory traversals (`../`) are explicitly blocked.
3. **Deterministic Idempotency**: Derived files and chunk manifests are named via:
   `artifact_name = sha256(input_media_header + config_hash)[:16]`
   This guarantees that interrupted jobs resume seamlessly without duplicating storage.
