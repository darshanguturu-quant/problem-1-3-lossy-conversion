# Task 3 — Lossy Video Conversion

Metadata-driven video transcoding decision engine and financial break-even ROI analyzer ensuring compute costs do not exceed storage and transfer savings.

## Directory Structure
- `src/`: Core implementation and CLI entrypoint.
- `tests/`: Automated unit and integration test suite.
- `specs/`: Engineering specifications (`PRD.md`, `ARCHITECTURE.md`, `PROGRESS.md`, `RULES.md`).
- `docs/`: Technical writeups and task documentation.
- `run.yaml`: Evaluator configuration and non-interactive commands.
- `requirements.txt`: Environment dependencies.

## Execution & Verification Commands
```bash
# Setup dependencies
pip install -r requirements.txt

# Run Task Execution Commands
python src/main.py --help
python src/main.py analyze-conversion --dry-run

# Run Test Suite
pytest tests/ -v
```

## Grader Bot Integration (`run.yaml`)
This folder is fully compatible with the [AIntropy Grader Bot](https://github.com/apps/aintropy-grader-bot).
