# Product Requirements Document (PRD)
## AIntropy Video Intelligence Platform

| Document Attribute | Details |
|---|---|
| **Project Name** | AIntropy Video Intelligence Engine & Evaluation Suite |
| **Document Version** | 1.1.0-HYBRID-SPEC |
| **Author** | Senior ML Engineer & Software Architect |
| **Status** | Approved Specification Layer (Pre-Implementation) |
| **Target Execution** | Automated Evaluation via AIntropy Grader Bot & Local CLI |
| **Reference Date** | October 2026 |

---

## 1. Executive Summary & Vision

Modern multimodal video retrieval-augmented generation (Video-RAG) systems are bounded by severe operational trade-offs: processing 4K/60fps raw video through heavy Vision-Language Models (VLMs) like Qwen3-VL or Gemini is cost-prohibitive, while naive uniform chunking and blind full-video indexing waste millions of dollars in compute and storage.

The **AIntropy Video Intelligence Platform** is designed as a production-grade, cost-optimized, and latency-sensitive video understanding and routing foundation. Rather than deploying heavyweight neural networks indiscriminately, the system operates as a tiered, decision-theoretic intelligence layer:
1. **At Ingestion (Task 1: Cascaded Hybrid Video Type Classifier)**: Videos are categorized into functional genres using a **two-stage cascaded hybrid architecture**:
   - **Stage 1**: Ultra-fast, lightweight classifier evaluating container metadata and low-cost temporal/histogram cues (< 15 ms CPU).
   - **Stage 2**: High-capacity semantic classifier leveraging **adaptive frame sampling** and **CLIP visual embeddings** + a calibrated linear head, triggered conditionally *only* when Stage 1 confidence falls below a calibrated escalation threshold $\theta_{\text{escalate}}$.
   - Compared against metadata-only, CLIP-only, and MobileNetV3 baselines to prove superior Pareto efficiency across accuracy, latency, and compute cost.
2. **At Storage Optimization (Task 3: Lossy Conversion)**: Videos are evaluated for compression trade-offs using an empirical break-even equation to minimize I/O overhead before indexing.
3. **At Segmentation (Task 4: Intelligent Video Chunking)**: Videos are decomposed into coherent temporal spans using content-aware, category-tailored policies rather than arbitrary fixed windows.
4. **At Query Time (Task 2: Cascaded Hybrid User Intent Classifier)**: User queries are analyzed by a **two-stage cascaded hybrid router**:
   - **Stage 1**: Ultra-fast word/character n-gram TF-IDF vectorizer + dense grammatical cue features with a calibrated Logistic Regression classifier (< 0.2 ms CPU). Resolves unambiguous queries immediately when confidence $p_1 \ge \theta_{\text{escalate\_intent}}$.
   - **Stage 2**: Dense sentence embeddings (`all-MiniLM-L6-v2`) + a lightweight classification head, triggered conditionally *only* when Stage 1 confidence falls below $\theta_{\text{escalate\_intent}}$ or probability margin is uncertain.
   - Classifies queries into the three retrieval granularities (`whole_video`, `frame_level`, `clip_level`) with Bayes-optimal decision layers to eliminate catastrophic routing regret and minimize downstream compute.
5. **At Evaluation (Task 5: Retrieval Benchmark Builder)**: A standardized benchmarking suite creates auditable, cross-dataset evaluation manifests to test retrieval accuracy, temporal localization, latency, and cross-dataset generalization across three heterogeneous production datasets alongside the reference VideoMME benchmark.

---

## 2. Scope & Problem Statement

### 2.1 In Scope
- **Classification & Routing**:
  - Ingestion-time classification of video formats into distinct categories: `gameplay`, `egocentric`, `podcast`, `wildlife`, `sports`, `news_lecture`, `meetings`, `surveillance`, `concert`, `film_animation`, and `general` fallback.
  - Implementation of a **Cascaded Hybrid Classifier** for Task 1:
    - Stage 1: Fast metadata/statistical feature classifier.
    - Stage 2: Adaptive frame sampling + CLIP embedding linear classifier for uncertain predictions.
    - Routing optimization based on calibrated confidence thresholds and escalation rate monitoring.
  - Implementation of a **Cascaded Hybrid User Intent Classifier** for Task 2:
    - Stage 1: Fast TF-IDF (word/char n-grams) + 34 grammatical cues + calibrated Logistic Regression head.
    - Stage 2: Dense sentence embeddings (`all-MiniLM-L6-v2`) + lightweight classification head for uncertain queries.
    - Query-time classification into retrieval granularities: `whole_video`, `frame_level`, and `clip_level`.
    - Routing optimization with cost-sensitive Bayes-optimal decision layers, ambiguity handling, confidence gating, and escalation rate monitoring.
- **Cost & Storage Optimization**:
  - Metadata-driven lossy compression decisions calculating I/O time savings, compute costs, storage savings, and monetary break-even points.
  - Safe codec derivation profiles (H.264/AVC baseline, H.265/HEVC storage-bound) that preserve VLM captioning and OCR readability.
- **Content-Aware Chunking**:
  - Single-pass, linear-time temporal segmentation honoring domain-specific dynamics (e.g., slide changes for meetings, motion onsets for surveillance, play transitions for sports).
  - Parent-child hierarchical interval schemas preventing physical media duplication.
- **Evaluation & Benchmarking**:
  - CLI tooling to build stratified, auditable query benchmarks across three heterogeneous real-world datasets and evaluate retrieval ranking metrics (Recall@K, MRR, nDCG@K, Context Precision, Temporal IoU).
  - Validation protocol against the official VideoMME 10% sample (`aintropy-ai/video-mme-sample10`).

### 2.2 Out of Scope (Explicit Non-Goals)
- Training massive foundation VLMs from scratch (e.g., pre-training a 70B parameter model).
- Running end-to-end Whisper ASR transcription when native audio timestamps or transcripts already exist.
- Hardcoding proprietary cloud infrastructure; the entire system must run on standard multi-core CPU environments.
- Destructive modification of source media; original uploads remain immutable masters.

---

## 3. The Five Core Assignment Tasks

### 3.1 Task 1: Cascaded Hybrid Video Type Classifier (Problem 1.1)
- **Objective**: Automatically infer the dominant video genre/format upon ingestion to select the optimal downstream chunking and compression policies while minimizing inference compute.
- **Input**: Raw video file (`.mp4`, `.mov`, `.mkv`, etc.) or local file path.
- **Architecture**:
  - **Stage 1 (Lightweight Gate)**:
    - Features: File duration, bitrate, container, FPS, resolution class, aspect ratio, audio channel count/sample rate, plus fast statistical metrics (mean and standard deviation of coarse color histograms across 2–4 keyframes).
    - Classifier: Calibrated Histogram Gradient Boosting / Logistic Regression head.
    - Decision: If prediction confidence $p_1 \ge \theta_{\text{escalate}}$ (e.g., $\theta = 0.85$), emit prediction immediately.
  - **Stage 2 (High-Capacity Semantic Refinement)**:
    - Triggered *only* when $p_1 < \theta_{\text{escalate}}$ (uncertain predictions).
    - **Adaptive Frame Sampling**: Sample frame count dynamically paced by video duration and visual change:
      - Short videos ($\le 5$ min): 4 frames.
      - Medium videos (5–30 min): 8 frames.
      - Long/High-entropy videos ($> 30$ min): 12 frames.
    - Embeddings: Pretrained CLIP visual encoder (`ViT-B/32` or compact `RN50`).
    - Classification Head: Class-balanced linear classifier / calibrated softmax layer fitted on pooled CLIP embeddings.
    - Final Decision: If Stage 2 confidence $p_2 \ge \theta_{\text{fallback}}$, assign class; otherwise route to `general` fallback.
- **Comparative Baseline Suite**:
  1. *Metadata-Only Baseline*: GBDT on container metadata only (duration, FPS, resolution, bitrate, audio presence).
  2. *CLIP-Only Baseline*: Unconditional CLIP feature extraction on all videos (100% heavy inference).
  3. *MobileNetV3 Baseline*: Single-stage frozen MobileNetV3-Small over fixed 8 frames.
  4. *Cascaded Hybrid (Proposed)*: Stage 1 metadata gate + conditional Stage 2 CLIP.
- **Taxonomy & Downstream Routing**:
  1. `gameplay` $\to$ `state_change_and_event`
  2. `egocentric` $\to$ `activity_and_temporal`
  3. `podcast` $\to$ `speech_turn_and_topic`
  4. `wildlife` $\to$ `scene_and_activity`
  5. `sports` $\to$ `event_and_play`
  6. `news_lecture` $\to$ `topic_and_slide`
  7. `meetings` $\to$ `speaker_turn_and_slide`
  8. `surveillance` $\to$ `activity_and_time_window`
  9. `concert` $\to$ `music_segment_and_performance`
  10. `film_animation` $\to$ `scene_and_shot`
  11. `general` $\to$ `adaptive_scene_and_semantic` (fallback for low confidence)
- **Deliverables**:
  - `src/classifier/stage1.py`: Lightweight metadata & histogram classifier.
  - `src/classifier/stage2.py`: Adaptive sampler + CLIP linear head.
  - `src/classifier/cascade.py`: Cascaded hybrid router with confidence gating.
  - CLI entrypoints in `src/main.py`: `train-classifier`, `predict-type`, and `benchmark-classifier-cascade`.

### 3.2 Task 2: Cascaded Hybrid User Intent Classifier (Problem 1.2)
- **Objective**: Route incoming natural-language retrieval queries to the minimum sufficient temporal granularity (`whole_video`, `frame_level`, `clip_level`) using a **two-stage cascaded hybrid architecture**, preventing catastrophic answer loss ("routing regret") while operating within strict sub-millisecond CPU latency budgets.
- **Input**: User natural language query string $q$.
- **Intent Taxonomy (Retrieval Granularities)**:
  1. `whole_video`: High-level queries requiring holistic understanding across the entire video (e.g., summaries, overarching topics, global sentiment, document synopsis, key takeaways). Downstream target: whole-video summary index or long-context VLM synopsis.
  2. `frame_level`: Pinpoint visual/OCR queries seeking exact single-instant information, specific objects, text on screen, HUD counters, slide text, or specific scene attributes (e.g., "what is written on the whiteboard at 04:12?", "what color shirt is the speaker wearing in the opening slide?", "find the score in the top right corner"). Downstream target: keyframe/image embedding and OCR index.
  3. `clip_level`: Temporal event or action queries spanning continuous intervals (e.g., "show me the penalty kick", "find the discussion about Q3 revenue", "when does the lion chase the zebra?", "where does the speaker explain backpropagation?"). Downstream target: temporal segment / chunk embedding index.
- **Two-Stage Cascaded Hybrid Architecture**:
  - **Stage 1 (Lightweight Lexical & Syntactic Gate)**:
    - *Feature Engineering*:
      - Sparse features: Word unigrams & bigrams (1-2), Character n-grams (3-5) with sublinear TF-IDF scaling.
      - Dense features (34 handcrafted cues): Temporal anchors (e.g., `\b\d{1,2}:\d{2}\b`, "at the end", "in the beginning"), frame visual/OCR markers ("what text", "what does the sign say", "what color", "who is shown", "logo", "slide title", "screenshot"), clip action/event markers ("show the moment", "when does X happen", "highlight", "play", "how did X do Y", "during the scene"), whole-video topic/synopsis markers ("summarize", "main topic", "what is this video about", "overview", "key takeaways", "overall conclusion"), WH-interrogative types (`what`, `when`, `where`, `who`, `why`, `how`), and query length/structural metrics.
    - *Classifier*: Calibrated Multinomial Logistic Regression with temperature scaling ($T_1$).
    - *Decision Gating*:
      - Let $\mathbf{p}_1 = [p_1(\text{whole\_video}), p_1(\text{frame\_level}), p_1(\text{clip\_level})]$.
      - Compute top confidence $p_{\max}^{(1)} = \max_c p_1(c)$ and probability margin $m^{(1)} = p_{(1)}^{(1)} - p_{(2)}^{(1)}$.
      - If $p_{\max}^{(1)} \ge \theta_{\text{escalate\_intent}}$ (provisional threshold: 0.82) AND $m^{(1)} \ge \Delta_{\text{margin}}$ (provisional margin: 0.30):
        $\implies$ Immediately emit Stage 1 prediction (< 0.2 ms CPU latency; 0% embedding overhead).
      - Otherwise ($p_{\max}^{(1)} < \theta_{\text{escalate\_intent}}$ or $m^{(1)} < \Delta_{\text{margin}}$):
        $\implies$ Escalate query to Stage 2.
  - **Stage 2 (High-Capacity Semantic Refinement)**:
    - *Triggered conditionally* on uncertain queries (~15%–20% of traffic).
    - *Feature Engineering*: Dense sentence embeddings using a compact, CPU-efficient sentence embedding model (`all-MiniLM-L6-v2`, 22.7M parameters, 384-dimensional dense semantic vector) concatenated with normalized Stage 1 prior probabilities and dense grammatical cue vectors (399-dim combined feature vector).
    - *Classifier Head*: Calibrated lightweight classifier (class-balanced Logistic Regression or 2-layer MLP head with dropout and temperature scaling $T_2$).
    - *Posterior Fusion*: Blended posterior $\mathbf{p}_{\text{final}} = \alpha \cdot \mathbf{p}_2 + (1 - \alpha) \cdot \mathbf{p}_1$ ($\alpha = 0.85$).
- **Ambiguity Handling & Bayes-Optimal Routing Decision**:
  - *Ambiguity Challenge*: Queries may be multi-intent (e.g., "Find the slide where revenue is shown and summarize the whole presentation"), underspecified ("database tutorial", "the dog"), or borderline frame/clip.
  - *Bayes-Optimal Decision Layer*: Rather than naive argmax, routes via expected utility:
    $$\hat{r} = \arg\max_r \sum_t p_{\text{final}}(t|q) \cdot U(t, r)$$
    where utility matrix $U(t, r) = A(t, r) - \lambda \cdot C(r)$:
    - $A(t, r)$ preserves answer retrieval: Exact match $A(t, t) = 1.0$; over-routing $A(\text{frame\_level}, \text{clip\_level}) = 0.75$ (frame is preserved within clip); catastrophic under-routing $A(\text{whole\_video}, \text{frame\_level}) = 0.0$ (complete answer loss); under-routing $A(\text{clip\_level}, \text{frame\_level}) = 0.20$.
    - $C(r)$ penalizes downstream retrieval compute ($C = [0.1, 0.3, 1.0]$ for frame, clip, whole-video).
  - *Ambiguity Fallback Policy*: If Stage 2 prediction remains uncertain ($\max(\mathbf{p}_{\text{final}}) < \theta_{\text{ambiguity}}$, provisional threshold 0.45), safely fall back to `clip_level` (the median-granularity safe middle ground, bounding regret without expensive full-video scans).
- **Comparative Baseline Suite**:
  1. *Rule-Based Heuristic Baseline*: Keyword and regex matching (brittle, zero probability calibration).
  2. *TF-IDF + Logistic Regression Baseline*: Unconditional Stage 1 alone (fast, but misses subtle semantic paraphrases).
  3. *Sentence-Embedding Baseline*: Unconditional `all-MiniLM-L6-v2` + linear head (100% heavy inference, ~18 ms latency on every query).
  4. *Compact-Transformer Baseline*: Fine-tuned DistilBERT / MobileBERT sequence classifier (~66M params, ~40 ms CPU latency).
  5. *Zero-Shot NLI Baseline*: DeBERTa-v3-small / BART-large-MNLI zero-shot classification (~150 ms latency, prompt-sensitive).
  6. *Cloud LLM Baseline*: GPT-4o-mini / Claude 3.5 Haiku API (450+ ms latency, high financial cost, non-deterministic, violates offline grader requirement).
  7. *Cascaded Hybrid (Proposed)*: Stage 1 TF-IDF+LR Gate + conditional Stage 2 Sentence Embeddings + Head (Pareto-optimal: sub-millisecond effective latency with deep transformer accuracy).
- **Deliverables**:
  - `src/intent/stage1.py`: Fast TF-IDF + 34 grammatical cue extractor and calibrated LR gate.
  - `src/intent/stage2.py`: Compact sentence embedding extractor (`all-MiniLM-L6-v2`) and calibrated lightweight head.
  - `src/intent/cascade.py`: Cascaded hybrid router with confidence/margin gating, Bayes utility layer, and ambiguity fallback.
  - `src/intent/route.py`: Standalone zero-dependency runtime script.
  - `src/intent/baselines.py`: Comprehensive benchmark suite evaluating all 6 alternative architectures.
  - CLI entrypoints in `src/main.py`: `train-intent`, `route-query`, and `benchmark-intent-cascade`.

### 3.3 Task 3: Metadata-Driven Lossy Video Conversion (Problem 1.3)
- **Objective**: Determine whether an incoming video should be converted to an optimized secondary representation for downstream VLM captioning, calculate the break-even ROI, and specify exact codec parameters.
- **Decision Engine**:
  - Evaluates original size $S$, duration $D$, compression ratio $r$, encoding speed $v$, and expected VLM passes $p$.
  - Time break-even: $S \cdot (1 - r) \cdot t_{\text{io}} \cdot p > D / v$.
  - Financial break-even: $S \cdot (1 - r) \cdot (c_s \cdot m + p \cdot c_{\text{read}}) > \frac{D}{v} \cdot c_{\text{compute}}$.
  - Retains immutable originals; bypasses files with bitrate < 80 MB/min already in modern codecs.
- **Deliverables**:
  - Metadata analyzer and break-even calculator CLI (`src/main.py analyze-conversion`).
  - FFmpeg execution wrapper for H.264 (CRF 23 baseline) and H.265 (CRF 26 storage).

### 3.4 Task 4: Intelligent Video Chunking (Problem 1.4)
- **Objective**: Partition continuous video streams into semantically coherent, variable-duration chunks tuned to specific video genres.
- **Algorithm**: Single-pass linear $O(D \cdot f_s)$ forward-pointer scan over visual cut candidates, speech intervals, and motion onsets.
- **Policies**: 8+ category policies (Meetings 60s/75s, Surveillance 30s, Sports 20s, Concerts 30s, Gameplay 20s, Egocentric 15s, Wildlife 20s, General 30s) with 5%–15% overlap constraint.
- **Deliverables**:
  - Chunking engine (`src/chunker.py`) generating JSON chunk manifests with parent-child interval hierarchies.

### 3.5 Task 5: Retrieval Benchmark Builder (Problem 1.5)
- **Objective**: Create a CLI tool that synthesizes, standardizes, and evaluates cross-dataset video retrieval benchmarks across three heterogeneous datasets and the reference VideoMME benchmark.
- **Capabilities**:
  - Generates stratified query shells by genre, duration, difficulty, and query intent.
  - Evaluates retrieval metrics: Recall@K ($K \in \{1, 5, 10, 20\}$), MRR, nDCG@K, Context Precision, and Evidence Temporal IoU.
  - Cross-dataset testing: In-domain, Zero-shot transfer (A+B $\to$ C), and Mixed-corpus retrieval.
- **Deliverables**:
  - Benchmark CLI generator and evaluator (`src/main.py build-benchmark`, `src/main.py evaluate-benchmark`).

---

## 4. Real-World Datasets & VideoMME Reference Reality Check

### 4.1 Critical Correction: VideoMME Dataset Labels
> [!IMPORTANT]
> **VideoMME Does NOT Contain Out-of-the-Box Video-Type Routing Labels.**
> While the VideoMME benchmark contains high-level `domain` categories (e.g. *Knowledge*, *Film & Television*, *Sports Competition*) and `sub_category` tags, it **does not** provide ground-truth labels for our required chunking-oriented taxonomy (`gameplay`, `egocentric`, `podcasts`, `wildlife`, `sports`, `news_lecture`, `meetings`, `surveillance`, `concert`, `film_animation`).
> Assuming VideoMME can be directly evaluated for video classification without an explicit label mapping or verified manual ground-truth manifest is an invalid assumption.

### 4.2 Dataset Roles & Verification Status
1. **VideoMME 10% Sample (`aintropy-ai/video-mme-sample10`)**:
   - Status: Verified available on Hugging Face (90 videos, 270 questions, 78 subtitles).
   - Role: Seed data for Task 2 intent queries and Task 5 smoke-testing. For Task 1, VideoMME's `domain` and `sub_category` tags serve only as coarse priors; a verified mapping manifest (`data/video_type_mapping.json`) must be created and verified before Task 1 training on VideoMME.
2. **The Three Real-World Heterogeneous Production Datasets (Datasets A, B, C)**:
   - **Dataset A (Structured Dialogue / Corporate)**: Meetings, earnings calls, slides, lectures.
   - **Dataset B (High-Motion Dynamic Visuals)**: Sports broadcasts, competitive gameplay, wildlife.
   - **Dataset C (Ambient / Surveillance / Continuous)**: CCTV streams, egocentric field cameras.
   - Status: Identities and concrete file paths are external inputs supplied via `examples/videos.json` and `run.yaml`.

---

## 5. Functional & Non-Functional Requirements

### 5.1 Functional Requirements
- **FR-1**: Single CLI interface supporting `train-classifier`, `predict-type`, `benchmark-classifier-cascade`, `train-intent`, `route-query`, `benchmark-intent-cascade`, `analyze-conversion`, `chunk-video`, `build-benchmark`, and `evaluate-benchmark`.
- **FR-2**: Autonomous execution via `run.yaml` without human prompts, external API key dependencies for core operations, or non-deterministic internet calls during grading.
- **FR-3**: Both Task 1 (video type) and Task 2 (query intent) must log whether each prediction was resolved at Stage 1 or escalated to Stage 2, outputting the respective escalation rate across test batches.
- **FR-4**: Chunking output must include exact millisecond timestamps and unique deterministic UUIDs based on video SHA-256 and span coordinates.
- **FR-5**: Benchmark builder must enforce split isolation: queries derived from Video X can never appear in train if Video X is in test.

### 5.2 Non-Functional Requirements (Provisional Targets)
> [!NOTE]
> All quantitative performance targets listed below are **provisional engineering targets** based on architecture calculations. They must be validated through empirical measurement during Phase 1–5 execution. Never fabricate numbers.

- **NFR-1 (Latency)**:
  - Task 1 Stage 1: P50 latency < 15 ms, P95 latency < 35 ms per video (CPU).
  - Task 1 Cascaded Overall: P50 latency < 45 ms, P95 latency < 180 ms per video (assuming $\le 25\%$ escalation).
  - Task 2 Stage 1 (TF-IDF + LR): P50 latency < 0.2 ms, P95 latency < 0.5 ms per query (CPU).
  - Task 2 Stage 2 (Sentence Embeddings + Head): P50 latency < 18 ms, P95 latency < 35 ms per query (CPU).
  - Task 2 Cascaded Overall: P50 latency < 0.3 ms, P95 latency < 19 ms per query (assuming $\le 20\%$ escalation).
  - Task 4 Chunking: Throughput $> 15\times$ real-time on 4-core CPU.
- **NFR-2 (Memory & Footprint)**:
  - Task 1 Stage 1 Model: < 2 MB weights, < 100 MB RAM.
  - Task 1 Stage 2 Model: < 150 MB weights (CLIP ViT-B/32 or compact RN50), < 600 MB RAM.
  - Task 2 Stage 1 Model: < 200 KB weights, < 40 MB RAM.
  - Task 2 Stage 2 Model: < 90 MB weights (`all-MiniLM-L6-v2` + linear head), < 100 MB RAM.
  - Total process memory: < 1.5 GB RAM.
- **NFR-3 (Cost & Efficiency)**:
  - Task 1 Escalation Rate Target: 15% – 25% of total videos escalated to Stage 2.
  - Task 2 Escalation Rate Target: 10% – 20% of total queries escalated to Stage 2.
  - 100% CPU executable; no GPU required for default evaluation.

---

## 6. Deliverables & Provisional Acceptance Criteria

| Task | Primary Deliverable | Provisional Acceptance Criteria (Pending Validation) |
|---|---|---|
| **Task 1: Cascaded Classifier** | `src/main.py predict-type --video <path>` | Macro-F1 $\ge 0.80$ on held-out test split; Escalation rate $\le 25\%$; Cascaded P50 latency $< 50$ ms; Outperforms metadata-only baseline by $\ge 8\%$ F1 and matches CLIP-only baseline within $2\%$ F1 at $4\times$ lower compute. |
| **Task 2: Cascaded Intent Classifier** | `src/main.py route-query --query <text>` | In-domain Macro-F1 $\ge 0.75$, OOD Macro-F1 $\ge 0.90$; Escalation rate $\le 20\%$; Cascaded P50 latency $< 0.3$ ms, P95 $< 19$ ms; Expected Calibration Error (ECE) $\le 0.06$; Catastrophic misroute rate ($whole\_video \to frame\_level$) $\le 1.0\%$; Outperforms TF-IDF-only baseline by $\ge 4\%$ F1 and matches Sentence-Embedding-only baseline within $1\%$ F1 at $5\times$ lower compute. |
| **Task 3: Lossy Conversion** | `src/main.py analyze-conversion <video>` | Exact break-even calculation matching analytical formula; verified H.264/H.265 profiles; zero data corruption on derived files. |
| **Task 4: Intelligent Chunking**| `src/main.py chunk-video --video <path>` | Boundary F1 $\ge 0.80$ ($\pm 2$s tolerance); overlap ratio constrained to 5–15%; throughput $> 15\times$ real-time on CPU. |
| **Task 5: Benchmark Builder** | `src/main.py build-benchmark` & `evaluate-benchmark` | End-to-end benchmark manifest creation across 3 datasets; metrics calculation for Recall@1/5/10, MRR, nDCG@10; verified 0% split leakage. |
| **Grader Integration** | `run.yaml` in repo root | Successfully validated by `aintropy-grader-bot` GitHub App without timeout, unhandled exceptions, or missing files. |

---

## 7. Risk Analysis & Mitigation

| Identified Risk | Impact | Likelihood | Mitigation Strategy |
|---|---|---|---|
| **VideoMME Label Mismatch** | High | High | Create an explicit, auditable domain-to-genre mapping manifest (`data/video_type_mapping.json`) and verified seed manifest; never claim unverified accuracy. |
| **Stage 2 Over-Escalation** | Medium | Medium | Calibrate threshold $\theta_{\text{escalate}}$ on validation split to target an escalation rate of 15%–25%; monitor latency impact. |
| **CLIP CPU Latency on Long Videos** | Medium | Low | Use adaptive frame sampling (cap at 12 frames maximum for long videos); leverage quantized OpenCLIP if needed. |
| **Downstream Caption Loss (Task 3)**| High | Medium | Enforce conservative CRF 23 baseline and validate against Qwen3-VL caption fidelity before transcoding production archives. |
| **Data Leakage in Benchmarks** | Critical | Low | Group-split by `video_id` strictly; queries from the same video are never partitioned across train and test. |
