# Technology Stack & Model Selection Matrix
## AIntropy Video Intelligence Platform

| Document Attribute | Details |
|---|---|
| **System** | Technology Stack, Lightweight ML Models & Dependency Governance |
| **Document Version** | 1.1.0-HYBRID-SPEC |
| **Author** | Senior ML Engineer & Software Architect |
| **Status** | Approved Specification Layer (Pre-Implementation) |
| **Core Focus** | Cascaded Hybrid Classification, Adaptive Sampling, Pareto-Optimal Compute |

---

## 1. Architectural Philosophy & Decision Drivers

Modern enterprise video retrieval pipelines often fail in production because they over-index on heavy neural networks for every ingestion decision. Running a 300 MB vision model across thousands of hours of predictable corporate meetings or surveillance streams wastes massive compute budgets.

## 1. Architectural Philosophy & Decision Drivers

Modern enterprise video retrieval pipelines often fail in production because they over-index on heavy neural networks for every ingestion and routing decision. Running heavy vision or language models across thousands of hours of predictable corporate meetings, surveillance streams, or simple keyword search queries wastes massive compute budgets.

Our platform architecture is uniformly governed by the **Cascaded Hybrid Principle** across both ingestion and retrieval:
- **Task 1 Video Format Classifier**:
  - *Stage 1 (Fast Gate)*: High-speed classification using container metadata, audio presence, and coarse visual color histograms (< 15 ms CPU latency, < 1.5 MB model).
  - *Stage 2 (Semantic Refinement)*: High-capacity visual representation using **adaptive frame sampling** and **CLIP embeddings** (`ViT-B/32` or compact `RN50`), executed *only* on the uncertain slice of videos ($\le 25\%$ escalation rate).
- **Task 2 Query Intent Classifier**:
  - *Stage 1 (Fast Gate)*: High-speed lexical and syntactic routing using word/character n-gram TF-IDF vectorization and 34 dense grammatical cue extractors with a calibrated Logistic Regression classifier (< 0.2 ms CPU latency, < 200 KB model).
  - *Stage 2 (Semantic Refinement)*: Dense semantic sentence embeddings (`all-MiniLM-L6-v2`) paired with a lightweight classifier head (LR / 2-layer MLP), executed *only* on uncertain or ambiguous queries ($\le 20\%$ escalation rate).

---

## 2. Model Selection & Comparative Trade-Offs (Task 1)

We evaluated four explicit configurations for Task 1 to establish quantitative trade-offs across accuracy, latency, memory, model size, and compute cost:

| Architecture / Configuration | Parameters | Model Size | CPU Latency (P50) | Memory Footprint (RAM) | Expected Macro-F1 (Provisional) | Escalation Rate | Compute Cost per 1M Videos (CPU) | Architectural Verdict |
|---|---|---|---|---|---|---|---|---|
| **Baseline 1: Metadata-Only** | ~50K | 0.8 MB | **10 ms** | **~70 MB** | 0.65 – 0.70 | 0% (Single-stage) | **$0.03** | Trivial baseline; fast but lacks visual discrimination. |
| **Baseline 2: MobileNetV3-Only** | 2.5M | 9.8 MB | 42 ms | ~120 MB | 0.76 – 0.81 | 0% (Single-stage) | $0.14 | Previous baseline; reasonable but misses fine semantics. |
| **Baseline 3: CLIP-Only (ViT-B/32)** | 87.8M | 140.0 MB | 165 ms | ~550 MB | 0.84 – 0.88 | 100% (All videos) | $0.55 | Maximum accuracy, but 16x slower than metadata; wasteful on obvious videos. |
| **PROPOSED: Cascaded Hybrid (Stage 1 + Stage 2 CLIP)** | **~50K + 87.8M** | **~141.5 MB** | **~38 ms (effective)** | **~550 MB (peak)** | **0.82 – 0.86** | **~20% – 25%** | **$0.16** | **SELECTED: Pareto-optimal; matches CLIP accuracy at MobileNetV3 speed/cost.** |

*Note: All latency, memory, and Macro-F1 numbers in this table are provisional engineering estimates based on 4-core standard CPU benchmarks and will be verified empirically on the held-out test split in Milestone M-1.*

### 2.1 Why the Cascaded Hybrid Outperforms Single-Stage Backbones (Task 1)
1. **Bimodal Problem Distribution**: Many video types exhibit unmistakable metadata signatures (e.g. podcasts have distinct stereo speech audio with low visual motion; surveillance has 24/7 durations and low FPS; lectures have 16:9 high-res slide aspect ratios). Stage 1 resolves ~75%–80% of typical enterprise traffic at near-zero compute cost.
2. **Adaptive Frame Sampling in Stage 2**:
   - Short/Simple clips ($\le 5$ min): Sample 4 frames.
   - Standard clips (5–30 min): Sample 8 frames.
   - Long/Dynamic clips ($> 30$ min): Sample 12 frames at motion peaks.
   This avoids the latency penalty of extracting 16+ dense frames on long videos.
3. **Escalation Gating**: By tuning the confidence threshold $\theta_{\text{escalate}}$ on the validation split, system operators can dynamically adjust the trade-off between inference speed and accuracy without retraining the underlying models.

---

## 3. Task 2 Model Selection: Cascaded Hybrid Intent Classifier & Query Router

To route natural-language retrieval queries to the minimum sufficient temporal granularity (`whole_video`, `frame_level`, `clip_level`), we benchmarked seven architectural options across speed, size, accuracy, and operational cost:

### 3.1 Comparative Architectural Trade-Off Matrix (Task 2)

| Router Architecture / Alternative | Parameters | Artifact Size | CPU Latency (P50) | Memory Footprint | In-Domain Macro-F1 (Prov.) | OOD Macro-F1 (Prov.) | Escalation Rate | Cost per 1M Queries | Architectural Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **1. Rule-Based Heuristic (Regex & Keywords)** | 0 | 0.2 KB | **0.05 ms** | **~10 MB** | 0.65 – 0.68 | 0.70 – 0.74 | 0% | **$0.00** | **Rejected**: Extremely brittle; zero probability calibration; fails on paraphrases. |
| **2. TF-IDF + Grammatical Cues + Logistic Regression (Stage 1 Alone)** | ~25K | **180 KB** | **0.15 ms** | **~35 MB** | 0.71 – 0.74 | 0.88 – 0.92 | 0% | **$0.00** | **Baseline**: Sub-millisecond speed, but misses complex semantic paraphrases and compound queries. |
| **3. Sentence-Embedding Only (`all-MiniLM-L6-v2` + Head)** | 22.7M | 88.0 MB | 18.5 ms | ~90 MB | 0.75 – 0.77 | 0.92 – 0.94 | 100% | $0.06 | **Baseline**: Strong semantic accuracy, but 120x slower than TF-IDF on simple queries. |
| **4. Compact Transformer Classifier (DistilBERT-base)** | 66.4M | 255.0 MB | 42.0 ms | ~180 MB | 0.76 – 0.78 | 0.92 – 0.94 | 100% | $0.15 | **Rejected**: Heavy footprint (> 250 MB); 280x slower; adds PyTorch overhead for clear queries. |
| **5. Zero-Shot NLI (DeBERTa-v3-small NLI)** | 141.0M | 380.0 MB | 160.0 ms | ~450 MB | 0.74 – 0.77 | 0.89 – 0.92 | 100% | $0.50 | **Rejected**: Prompt hypothesis sensitive; slow latency violates interactive query UX. |
| **6. Cloud LLM API (e.g., GPT-4o-mini)** | ~8B (est.) | Remote | 450.0 ms | Remote | 0.78 – 0.80 | 0.93 – 0.95 | 100% | $300.00 – $600.00 | **Rejected**: Violates offline grading (FR-2); non-deterministic; unacceptable latency & cost. |
| **7. PROPOSED: Cascaded Hybrid (TF-IDF Gate + Conditional Sentence Embeddings)** | **~25K + 22.7M** | **~88.2 MB** | **~0.25 ms (effective)** | **~90 MB (peak)** | **0.76 – 0.79** | **0.93 – 0.95** | **~15% – 20%** | **$0.01** | **SELECTED**: Pareto-optimal. Sub-millisecond P50 latency with full transformer generalization. |

*Note: All latency, memory, and Macro-F1 numbers in this table are provisional engineering estimates based on 4-core standard CPU benchmarks and will be verified empirically on the held-out test split in Milestone M-2.*

### 3.2 Why the Cascaded Hybrid Outperforms Alternatives (Task 2)
1. **Asymmetric Query Distribution**: In production Video-RAG, ~80%–85% of incoming queries contain explicit syntactic markers or keyword cues:
   - "What did the slide say at 04:15?" $\to$ explicit timestamp and OCR token (`frame_level`).
   - "Show the moment the goal was scored" $\to$ action verb and event trigger (`clip_level`).
   - "Give me a summary of the whole meeting" $\to$ synopsis trigger (`whole_video`).
   Stage 1 resolves these queries in **0.15 ms** on CPU using pure numpy/scikit-learn linear weights, saving 99% of compute cycles.
2. **Conditional Semantic Escalation**: Only queries lacking explicit lexical markers or exhibiting high ambiguity (e.g., "explain backpropagation with visual diagrams", "check whether the coach was upset") fall below the escalation threshold ($\theta_{\text{escalate\_intent}} = 0.82$) or margin ($\Delta_{\text{margin}} = 0.30$) and trigger Stage 2 `all-MiniLM-L6-v2` dense embedding inference.
3. **Calibrated Decision Layer & Routing Regret**:
   Under-routing ($whole\_video \to frame\_level$) causes complete search failure (100% regret), while over-routing ($frame\_level \to clip\_level$) causes only minor compute overhead (25% regret). The calibrated Bayes utility matrix ensures safe, asymmetric error trade-offs that naive neural classifiers ignore.
4. **Ambiguity Fallback**:
   Queries that remain ambiguous even after Stage 2 ($\max(\mathbf{p}) < \theta_{\text{ambiguity}} = 0.45$) safely fall back to `clip_level`, the median granularity that preserves answer recall without triggering expensive full-video scans.

---

## 4. Recommended Library & Framework Stack

### 4.1 Primary Runtime Dependencies (`requirements.txt`)

```ini
# Core Computation & Machine Learning
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
scikit-learn>=1.3.0,<1.6.0
joblib>=1.3.0

# Computer Vision & Lightweight Deep Learning
torch>=2.0.0,<2.5.0; sys_platform != 'darwin'
torchvision>=0.15.0,<0.20.0; sys_platform != 'darwin'
open-clip-torch>=2.20.0,<3.0.0
opencv-python-headless>=4.8.0.76,<5.0.0.0

# Natural Language & Sentence Embeddings (Task 2 Stage 2)
sentence-transformers>=2.2.2,<3.0.0
transformers>=4.30.0,<5.0.0

# Video Probing & Media Utilities
ffmpeg-python>=0.2.0

# Data Manipulation & Formats
pandas>=2.0.0,<3.0.0
pyarrow>=14.0.0
polars>=0.20.0

# Configuration & Logging
pyyaml>=6.0.1
pydantic>=2.4.0

# Testing & Linting
pytest>=7.4.0
pytest-cov>=4.1.0
```

### 4.2 System & Tooling Prerequisites
- **FFmpeg & ffprobe**: Required version $\ge 4.4$ (accessible on system `$PATH`). Used for metadata extraction, packet timestamp decoding, and transcoding.
- **Python Runtime**: Standard CPython 3.10 or 3.11 (64-bit).

---

## 5. Component-by-Component Technology Mapping

| Platform Component | Primary Library | Primary Model / Algorithm | Alternative Evaluated | Rejection Rationale for Alternative |
|---|---|---|---|---|
| **Video Probing** | Native `ffprobe` / `subprocess` | JSON Stream Extraction | `pymediainfo` | Adds heavy C library dependency (`libmediainfo`). |
| **Fast Frame Sampling** | `opencv-python-headless` | Keyframe seeking & BGR decoding | `pyav` | PyAV build complications across certain Windows/Linux distributions. |
| **Task 1: Stage 1 Gate** | `scikit-learn` | `HistGradientBoostingClassifier` over metadata + color stats | Deep Neural Network | Overkill for metadata; adds unnecessary startup and inference latency. |
| **Task 1: Stage 2 Semantic**| `open-clip-torch` / `torch` | Pretrained CLIP `ViT-B/32` (or compact `RN50`) + Linear Head | VideoMAE / Timesformer | VideoMAE is 340 MB and 10x slower on CPU. |
| **Query Preprocessing** | Standard `re` + `unicodedata` | NFKC normalization & regex tokenization | SpaCy | SpaCy models require ~15-50 MB downloads and slow down startup. |
| **Task 2: Stage 1 Gate** | `scikit-learn` | Word (1-2) + Char (3-5) TF-IDF + 34 grammatical cues + Calibrated LR | Hardcoded Regex | Brittle; zero probability calibration; fails on paraphrases. |
| **Task 2: Stage 2 Semantic**| `sentence-transformers` / `torch` | Pretrained `all-MiniLM-L6-v2` dense embeddings (384-d) + Linear Head | DistilBERT / DeBERTa | DistilBERT is 250 MB; DeBERTa is 380 MB and 8x slower. |
| **Task 2: Decision Layer** | `numpy` | Bayes-Optimal Expected Utility Maximizer + Regret Penalty | Hardcoded if-else | Brittle; unable to dynamically adjust for compute cost parameter $\lambda$. |
| **Chunking Engine** | Standard Library (`dataclasses`) | Content-Aware Boundary Planning | PySceneDetect | Custom chunker operates in linear single-pass with multi-cue fusion. |
| **Benchmark Suite** | `polars` + `numpy` | Vectorized Recall@K, MRR, nDCG@K | `ranx` / `trec_eval` | Eliminates external C-extension dependencies while guaranteeing exact math. |

---

## 6. Memory & Deployment Budget Governance

```
========================================================================================
Artifact / Component            Target Disk Budget   Target RAM Budget   Latency Budget
========================================================================================
Task 1 Stage 1 Model            < 2.0 MB             < 100 MB            < 15 ms
Task 1 Stage 2 CLIP Model       < 150.0 MB           < 600 MB            < 160 ms
Task 2 Stage 1 Gate (TF-IDF+LR) < 200 KB             < 40 MB             < 0.2 ms
Task 2 Stage 2 Model (MiniLM)   < 90.0 MB            < 100 MB            < 18 ms
Task 3 Transcoder Engine        < 50 KB              < 50 MB             < 10 ms (decision)
Task 4 Intelligent Chunker      < 50 KB              < 100 MB            > 15x real-time
Task 5 Benchmark Builder        < 100 KB             < 200 MB            < 1.0 s / 1k queries
Total Platform Footprint        < 245 MB             < 1.1 GB            Well within 1.5 GB RAM
========================================================================================
```
