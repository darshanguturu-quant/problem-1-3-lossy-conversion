# Problem 1.3: Lossy Conversion

## Submission instructions

1. Install the **AIntropy Grader Bot** GitHub App (https://github.com/apps/aintropy-grader-bot) and grant it access to your fork, if you haven't already — we can't evaluate your submission without this.
2. Put your code under `src/`.
3. Put your write-up under `docs/` — markdown, slides, a notebook, or another format of your choosing are all fine, as long as it's committed in this repo.
4. Fill in `run.yaml` with the **exact** commands needed to set up and run your code. We run these commands exactly as written to evaluate your submission — this is the one part of the submission we read literally, not loosely: if `run.yaml` doesn't work, we can't see your code run.
5. Push to your own fork's default branch any time during your test's time limit. No pull request is needed or expected.

---

## Solution Overview: Metadata-Driven Lossy Video Conversion & Break-Even ROI Engine

An analytical optimization engine that models storage cost, compute expenditure, and I/O query latency to determine when video transcoding is economically justified.

### Architecture Highlights
- **Analytical Break-Even Engine**: Computes time break-even ($S(1-r)t_{\text{io}}p > D/v$) and financial break-even ($S(1-r)(c_s m + p c_{\text{read}}) > \frac{D}{v} c_{\text{compute}}$).
- **Smart Bypass**: Skips conversion when bitrate < 80 MB/min and codec is already AVC/HEVC.
- **Safe Codec Profiles**: H.264 Baseline (CRF 23, `preset=fast`, AAC audio) and H.265 Storage Profile (CRF 26, `preset=medium`).
- **Non-Destructive Storage**: Original files are immutable masters. All derived representations are isolated in `derived/`.

### Repository Layout
- `src/transcoder/`: Analytical ROI calculator, FFmpeg parameter generator, and codec benchmark suite.
- `docs/TASK3_LOSSY_CONVERSION.md`: Mathematical derivations and cost break-even analysis.
- `specs/`: System requirements, codec configurations, and rules.
- `tests/`: Automated unit and integration test suite (20 passing tests).
- `run.yaml`: Evaluator entrypoint executed non-interactively by AIntropy Grader Bot.

### Commands & Execution
```bash
# 1. Setup
pip install -r requirements.txt

# 2. Run Solution / Break-Even Analysis
python src/main.py --help
python src/main.py analyze-conversion --dry-run

# 3. Run Automated Tests
pytest tests/ -v
```
