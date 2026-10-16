# Development Rules, Engineering Standards & Governance Policy
## AIntropy Video Intelligence Platform

| Document Attribute | Details |
|---|---|
| **System** | Engineering Governance, Code Quality, Security & Reproducibility Policy |
| **Document Version** | 1.1.0-HYBRID-SPEC |
| **Author** | Senior ML Engineer & Software Architect |
| **Status** | Approved Specification Layer (Pre-Implementation) |
| **Enforcement** | Mandatory Pre-Commit, Pre-PR, and Automated CI/Grader Evaluation |

---

## 1. Absolute Cardinal Rules

1. **No Code Without Verification**: Do not start implementing tasks before completing and approving the specification layer.
2. **Never Fabricate Numbers**: Never invent benchmark accuracies, dataset sizes, compute costs, or test results. Distinguish explicitly between **measured empirical results**, **analytical engineering estimates**, and **unvalidated assumptions**.
3. **No Phantom Datasets & No Label Assumptions**:
   - Never invent names or identities for the three real-world datasets mentioned in the assignment.
   - **Never assume VideoMME provides ready-made video-type routing labels**. VideoMME (`aintropy-ai/video-mme-sample10`) provides `domain`, `sub_category`, and `task_type`, but NOT chunking-oriented video type categories. Ground truth for Task 1 must be created via verified mapping manifests or annotated manifests (`src/labels.csv`).
4. **Mark Provisional Targets**: All performance goals (e.g. Macro-F1 $\ge 0.80$, effective P50 latency $< 50$ ms, escalation rate $\le 25\%$) must be marked as **provisional engineering targets pending empirical measurement** on the held-out test split.
5. **Idempotent, Non-Interactive Commands**: Every command defined in `run.yaml` must execute without interactive prompts, user input, or non-deterministic internet dependencies during grading.
6. **CPU-First Lightweight Discipline**: Prioritize lightweight models (MobileNetV3-Small, Stage 1 metadata GBDT, TF-IDF + linear models) with low RAM footprints (< 1.5 GB) and sub-millisecond CPU latency over heavy transformer models.

---

## 2. Specific Rules for Task 1 (Cascaded Hybrid Classifier)

1. **Strict Two-Stage Execution**:
   - In production inference (`predict-type`), Stage 1 (metadata + coarse stats) must **always** execute first.
   - Stage 2 (adaptive frame sampling + CLIP) must **only** execute conditionally when Stage 1 confidence is below the escalation threshold $\theta_{\text{escalate}}$ ($p_1 < \theta$).
   - Blind, unconditional invocation of CLIP across all videos is strictly forbidden in default prediction mode (reserved solely for the CLIP-only baseline evaluation).
2. **Adaptive Frame Sampling Bounds**:
   - Frame sampling in Stage 2 must scale dynamically with video duration and visual change:
     - Short ($\le 5$ min): 4 frames.
     - Standard (5–30 min): 8 frames.
     - Long/Dynamic ($> 30$ min): 12 frames maximum.
   - Dense frame sampling (e.g., 30+ frames) on CPU is strictly forbidden due to latency budgets.
3. **Escalation Rate Monitoring**:
   - The system must explicitly track and report the escalation rate ($\rho = N_{\text{escalated}} / N_{\text{total}}$). The target escalation rate on standard traffic is $15\% \le \rho \le 25\%$.
4. **Mandatory Baseline Reporting**:
   - Task 1 evaluation must always report comparative metrics against all three baselines: Metadata-only, MobileNetV3-only, and CLIP-only.

### 2.2 Specific Rules for Task 2 (Cascaded Hybrid Intent Classifier)

1. **Strict Two-Stage Execution**:
   - In production query routing (`route-query`), Stage 1 (word/char TF-IDF + 34 grammatical cues + Logistic Regression) must **always** execute first.
   - Stage 2 (`all-MiniLM-L6-v2` dense sentence embeddings + lightweight head) must **only** execute conditionally when Stage 1 confidence falls below the escalation threshold ($p_{\max}^{(1)} < \theta_{\text{escalate\_intent}} = 0.82$) or probability margin is narrow ($m^{(1)} < \Delta_{\text{margin}} = 0.30$).
   - Blind, unconditional invocation of sentence embeddings across 100% of queries is strictly forbidden in default serving mode (reserved solely for the sentence-embedding baseline evaluation).
2. **Taxonomy & Intent Governance**:
   - Strictly adhere to the canonical retrieval granularities: `whole_video`, `frame_level`, and `clip_level`. Do not use alternate ad-hoc label strings.
3. **Escalation Rate Monitoring**:
   - The system must explicitly track and report the escalation rate ($\rho_{\text{intent}} = N_{\text{escalated}} / N_{\text{total}}$). The target escalation rate on standard traffic is $10\% \le \rho_{\text{intent}} \le 20\%$.
4. **Bayes Regret Optimization & Catastrophic Loss Prevention**:
   - The decision layer must employ the cost-sensitive utility matrix $U(t, r)$ to penalize under-routing ($whole\_video \to frame\_level$). Catastrophic under-routing must be held $\le 1.0\%$.
5. **Ambiguity Fallback Policy**:
   - When query confidence remains uncertain even after Stage 2 ($\max(\mathbf{p}) < \theta_{\text{ambiguity}} = 0.45$), the router must safely fall back to `clip_level` (the median-granularity safe middle ground), never randomly guessing or throwing unhandled exceptions.
6. **Mandatory Baseline Reporting**:
   - Task 2 evaluation must always report comparative metrics against all six alternative baselines: Rule-Based Heuristics, TF-IDF alone, Sentence-Embedding alone, Compact Transformer, Zero-Shot NLI, and Cloud LLM reference.
7. **Zero-Heavy-Dependency Fast Path**:
   - Stage 1 standalone inference (`src/intent/route.py`) must execute with zero heavyweight dependencies (pure numpy/scikit-learn runtime) with sub-millisecond execution.

---

## 3. Coding & Architecture Standards

### 3.1 Python Code Standards
- **Version Compatibility**: Target Python 3.10 and 3.11.
- **Style Guide**: Strict adherence to PEP 8. Format using `black` (line length 100) and lint with `ruff` or `flake8`.
- **Type Annotations**: 100% of public function signatures and module interfaces must have explicit Python type annotations (`typing` module: `List`, `Dict`, `Optional`, `Tuple`, `Union`, `Literal`).
- **Docstring Format**: Google or NumPy style docstrings for every class, public method, and module, explaining arguments, return types, exceptions, and algorithmic complexity.
- **Preserve Existing Documentation**: Never blindly overwrite existing READMEs or configs; adapt existing templates thoughtfully.

### 3.2 Subprocess & Shell Safety
- **Forbidden**: `shell=True` in `subprocess.run` or `subprocess.Popen` is strictly forbidden.
- **Mandatory**: All FFmpeg, ffprobe, and system calls must pass argument lists:
  ```python
  # CORRECT:
  cmd = ["ffprobe", "-v", "error", "-show_format", "-of", "json", video_path]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)

  # FORBIDDEN:
  subprocess.run(f"ffprobe {video_path}", shell=True)  # CRITICAL SECURITY RISK
  ```
- **Timeout Enforcement**: Every external subprocess must specify an explicit timeout (`timeout=30.0`) to prevent hanging grader runs.

### 3.3 Path Sanitization & File I/O
- Always use `pathlib.Path` or `os.path.abspath` to resolve file paths.
- Guard against path traversal attacks (`../` sequences) when accepting user-supplied video paths or output directories.
- Non-destructive writes: Derived videos must be placed in a dedicated `derived/` directory; source media files are strictly read-only.

---

## 4. Dependency Management & Minimalist Stack Rules

1. **Pinned Dependencies**: All dependencies must be version-constrained in `requirements.txt`.
2. **Zero Bloat Policy**: Do not import heavy libraries for minor utility functions.
   - Use Python's built-in `re` and `unicodedata` instead of installing spaCy or NLTK for text normalization.
   - Use built-in `dataclasses` and `json` instead of complex third-party serialization libraries unless required.
3. **Headless Computer Vision**: Always specify `opencv-python-headless` instead of `opencv-python` to avoid missing X11/GUI system library crashes in headless Linux server environments.
4. **Environment Isolation**: All development and testing must take place inside a dedicated virtual environment (`.venv`).

---

## 5. Experiment Reproducibility & Leakage Control

### 5.1 Strict Global Determinism
Every script performing training, feature selection, or dataset splitting must seed all RNG sources deterministically:
```python
RANDOM_SEED = 42

import random
import numpy as np

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
except ImportError:
    pass
```

### 5.2 Data Partitioning Anti-Leakage Protocol
- **Group Isolation by Video**: Data splits for training, validation, and testing MUST be grouped by `video_id`.
- **Reason**: Multiple queries or sampled frames from the same video share entity names, actors, objects, and acoustic profiles. Randomly splitting queries across train and test results in massive data leakage and misleading evaluation metrics.
- **Metric Verification**: Confidence intervals must be computed using cluster bootstrapping over video entities.

---

## 6. External Access Requirements & Manual Setup Instructions

### 6.1 AIntropy Grader Bot Installation (Mandatory Manual Step)
The grading pipeline requires the **AIntropy Grader Bot** GitHub App:
1. Navigate to: `https://github.com/apps/aintropy-grader-bot`
2. Click **Install** or **Configure**.
3. Select your GitHub user account or target organization.
4. Grant the app access to your fork of the problem set repository (`aintropy-video-intelligence` or problem-specific fork).
5. Verify that repository checks and commit webhooks are enabled.
*Note: Without this step, the grader cannot dispatch automated evaluation runs or grade submissions upon git push.*

### 6.2 System Prerequisites
Ensure the target host environment provides:
- Python 3.10 or 3.11
- FFmpeg and ffprobe binaries accessible on the system `$PATH`:
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Ensure `ffmpeg.exe` and `ffprobe.exe` are in system environment PATH.

---

## 7. Mandatory Validation Rules for `run.yaml`

The `run.yaml` file is executed literally by the AIntropy Grader Bot. It must strictly conform to these rules:

1. **Idempotence**: Every command can be run multiple times on a fresh checkout without failing or leaving dirty locks.
2. **Non-Interactive**: Commands must not prompt for confirmation (e.g. use `pip install -r requirements.txt`, not interactive prompts).
3. **Exact Command Verification**:
   ```yaml
   setup_commands:
     - "pip install -r requirements.txt"

   run_commands:
     - "python src/main.py --help"
     - "python src/main.py test-all"

   test_commands:
     - "python -m pytest tests/ -v"

   notes: "Executes on lightweight CPU. Pretrained CLIP weights cached locally during setup."
   ```
4. **Pre-Push Validation**: Before pushing any commit to the default branch, the developer must run all commands listed in `run.yaml` locally in a fresh virtual environment to ensure zero breaks.
