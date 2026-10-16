"""
AIntropy Video Intelligence Engine & Evaluation Suite
Main CLI entrypoint for Task 1 (Video Classification) and Task 2 (Query Intent Classification).
"""

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
import argparse
from typing import Dict, Any, Optional

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("aintropy")

from src.intent.data import build_intent_dataset, INTENT_CLASSES
from src.intent.stage1 import Stage1IntentClassifier
from src.intent.stage2 import Stage2IntentClassifier, SentenceEmbeddingExtractor
from src.intent.cascade import CascadedIntentClassifier
from src.intent.baselines import IntentBenchmarkRunner
from src.intent.route import load_router, route_query_cli

import numpy as np
from src.profiler import VideoProfiler, VideoMetadata
from src.classifier.stage1 import Stage1Classifier, Stage1Features, DEFAULT_CLASSES
from src.classifier.stage2 import Stage2Classifier
from src.classifier.cascade import CascadedClassifier, CascadeResult
from src.classifier.baselines import MobileNetV3Classifier, BenchmarkRunner
from src.dataset import (
    inspect_videomme_dataset,
    create_synthetic_benchmark_dataset,
    load_manifest_csv,
    generate_synthetic_video,
)
from src.transcoder import (
    VideoProfilerFFprobe,
    ConversionDecisionEngine,
    CostModelParameters,
    DecisionRecommendation,
    get_profile,
    adapt_profile_for_content,
    TranscodeEngine,
    measure_video_quality,
    CodecBenchmarkRunner,
)
from src.chunker import (
    IntelligentChunker,
    FixedDurationChunker,
    BlindSceneCutChunker,
    ChunkingPolicy,
    get_policy,
    list_policies,
    chunk_video,
    ChunkerBenchmarkRunner,
)
from src.benchmark import (
    RetrievalBenchmarkBuilder,
    RetrievalEvaluator,
    CrossDatasetTransferEvaluator,
    BenchmarkManifest,
    QueryRetrievalPrediction,
)

def handle_train_intent(args) -> None:
    """Train Task 2 Stage 1 (TF-IDF + LR) and Stage 2 (Sentence Embeddings + Head) models."""
    logger.info("Building intent dataset and partitions from %s...", args.data)
    manifest = build_intent_dataset(videomme_path=args.data, output_path=args.dataset_out, seed=args.seed)

    train_samples = manifest["train"]
    val_samples = manifest["val"]

    train_queries = [x["query"] for x in train_samples]
    train_labels = [x["intent"] for x in train_samples]
    val_queries = [x["query"] for x in val_samples]
    val_labels = [x["intent"] for x in val_samples]

    logger.info("Training Stage 1 (TF-IDF + 34 Grammatical Cues + Logistic Regression)...")
    s1 = Stage1IntentClassifier(c_param=args.c1, random_state=args.seed)
    s1.fit(train_queries, train_labels, val_queries=val_queries, val_labels=val_labels)

    logger.info("Training Stage 2 (Sentence Embeddings + Priors + Lightweight Head)...")
    train_s1_probs = [s1.predict_proba(q) for q in train_queries]
    val_s1_probs = [s1.predict_proba(q) for q in val_queries]

    s2 = Stage2IntentClassifier(c_param=args.c2, blend_alpha=args.alpha, random_state=args.seed)
    s2.fit(
        train_queries,
        train_labels,
        stage1_probs=train_s1_probs,
        val_queries=val_queries,
        val_labels=val_labels,
        val_stage1_probs=val_s1_probs
    )

    logger.info("Initializing and tuning Cascaded Intent Classifier on validation data...")
    cascade = CascadedIntentClassifier(
        s1,
        s2,
        escalation_threshold=args.escalation_th,
        margin_threshold=args.margin_th,
        ambiguity_threshold=args.ambiguity_th,
        ambiguity_mode=args.ambiguity_mode
    )
    tuned = cascade.tune_thresholds(val_queries, val_labels)

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    s1_path = os.path.join(args.output_dir, "intent_stage1.joblib")
    s2_path = os.path.join(args.output_dir, "intent_stage2.joblib")
    s1.save(s1_path)
    s2.save(s2_path)

    config_path = os.path.join(args.output_dir, "intent_cascade_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "escalation_threshold": cascade.escalation_threshold,
            "margin_threshold": cascade.margin_threshold,
            "ambiguity_threshold": cascade.ambiguity_threshold,
            "ambiguity_mode": cascade.ambiguity_mode,
            "classes": INTENT_CLASSES,
            "stage1_path": s1_path,
            "stage2_path": s2_path
        }, f, indent=2)

    logger.info("Training complete. Artifacts saved to '%s'.", args.output_dir)


def handle_route_query(args) -> None:
    """Route a single search query to temporal granularity."""
    s1_path = os.path.join(args.model_dir, "intent_stage1.joblib")
    s2_path = os.path.join(args.model_dir, "intent_stage2.joblib")

    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        logger.info("Models not found in '%s'. Training quick baseline model first...", args.model_dir)
        args_train = argparse.Namespace(
            data="test.jsonl",
            dataset_out="data/intent_dataset.json",
            output_dir=args.model_dir,
            seed=42,
            c1=1.0,
            c2=1.0,
            alpha=0.85,
            escalation_th=0.82,
            margin_th=0.30,
            ambiguity_th=0.45,
            ambiguity_mode=args.ambiguity_mode
        )
        handle_train_intent(args_train)

    router = load_router(
        s1_path,
        s2_path,
        ambiguity_mode=args.ambiguity_mode
    )
    res = router.predict(args.query)

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print("=" * 60)
        print(f"Query:              {res.query}")
        print(f"Predicted Intent:   {res.predicted_intent.upper()} (conf: {res.confidence:.4f})")
        print(f"Stage Resolved:     {res.stage_resolved} (Escalated: {res.escalated_to_stage2})")
        print(f"Routing Target:     {res.routing_target}")
        print(f"Latency:            {res.latency_ms:.3f} ms")
        print(f"Utility Score:      {res.utility_score:.4f}")
        print("Probabilities:")
        for k, v in res.probabilities.items():
            print(f"  - {k:15s}: {v:.4f}")
        print("=" * 60)


def handle_benchmark_intent(args) -> None:
    """Run full empirical benchmark comparing TF-IDF alone, Sentence Embeddings alone, Rule-based, and Cascade."""
    s1_path = os.path.join(args.model_dir, "intent_stage1.joblib")
    s2_path = os.path.join(args.model_dir, "intent_stage2.joblib")

    dataset_path = args.data
    if not os.path.exists(dataset_path):
        logger.info("Dataset manifest '%s' not found. Generating from test.jsonl...", dataset_path)
        build_intent_dataset(output_path=dataset_path, seed=args.seed)

    with open(dataset_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Ensure models exist
    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        logger.info("Models not found in '%s'. Training intent models first...", args.model_dir)
        args_train = argparse.Namespace(
            data="test.jsonl",
            dataset_out=dataset_path,
            output_dir=args.model_dir,
            seed=args.seed,
            c1=1.0,
            c2=1.0,
            alpha=0.85,
            escalation_th=0.82,
            margin_th=0.30,
            ambiguity_th=0.45,
            ambiguity_mode=args.ambiguity_mode
        )
        handle_train_intent(args_train)

    s1 = Stage1IntentClassifier.load(s1_path)
    s2 = Stage2IntentClassifier.load(s2_path)
    cascade = CascadedIntentClassifier(
        s1,
        s2,
        escalation_threshold=0.82,
        margin_threshold=0.30,
        ambiguity_threshold=0.45,
        ambiguity_mode=args.ambiguity_mode
    )

    test_samples = manifest["test"]
    test_queries = [x["query"] for x in test_samples]
    test_labels = [x["intent"] for x in test_samples]

    runner = IntentBenchmarkRunner(INTENT_CLASSES)
    logger.info("Running Task 2 Intent Benchmark over %d test queries...", len(test_queries))
    reports = runner.run_benchmark(s1, s2, cascade, test_queries, test_labels, save_dir=args.output_dir)

    # Print markdown table
    print("\n" + "=" * 90)
    print("TASK 2: USER INTENT CLASSIFIER BENCHMARK RESULTS")
    print("=" * 90)
    print(runner.format_markdown_table(reports))
    print("=" * 90 + "\n")

    # Save summary report
    out_file = os.path.join(args.output_dir, "intent_benchmark_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in reports.items()}, f, indent=2)
    logger.info("Saved Task 2 benchmark scorecard to %s", out_file)


def handle_inspect_dataset(args) -> None:
    """Audit VideoMME dataset labels and print report."""
    print("=" * 70)
    print("DATASET INSPECTION & LABEL AUDIT")
    print("=" * 70)
    jsonl_path = getattr(args, "jsonl", None) or "test.jsonl"
    report = inspect_videomme_dataset(jsonl_path)

    if not report.get("available"):
        print(f"Error: {report.get('error')}")
        return

    print(f"Dataset path: {jsonl_path}")
    print(f"Total QA records: {report['total_qa_records']}")
    print(f"Unique video IDs: {report['unique_videos']}")
    print("\nDomain Distribution:")
    for d, cnt in sorted(report["domains"].items(), key=lambda x: -x[1]):
        print(f"  - {d}: {cnt} videos")

    print("\nTop Sub-Categories:")
    for s, cnt in sorted(report["sub_categories"].items(), key=lambda x: -x[1])[:10]:
        print(f"  - {s}: {cnt} videos")

    print(f"\nNative Video-Type Routing Labels Present: {report['native_routing_labels_present']}")
    print(f"\nLimitation Note:\n{report['limitation_note']}")
    print("=" * 70)


def handle_predict_type(args) -> None:
    """Predict video type and routing strategy for an incoming video file."""
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: video not found at {video_path}", file=sys.stderr)
        sys.exit(1)

    model_dir = getattr(args, "model_dir", None) or "artifacts/cascade_model"
    if os.path.exists(os.path.join(model_dir, "stage1.joblib")):
        cascade = CascadedClassifier.load(model_dir)
    else:
        cascade = CascadedClassifier(
            escalation_threshold=getattr(args, "escalation_threshold", 0.85)
        )

    res = cascade.predict_video(video_path, use_cache=not getattr(args, "no_cache", False))
    output = res.to_dict()

    if getattr(args, "output_json", None):
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Prediction saved to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))


def handle_train_classifier(args) -> None:
    """Train Task 1 cascade model on dataset manifest or synthetic suite."""
    save_dir = getattr(args, "save_dir", None) or "artifacts/cascade_model"
    logger.info("Training cascaded classifier to %s...", save_dir)

    manifest_path = getattr(args, "manifest", None)
    if manifest_path and os.path.exists(manifest_path):
        paths, labels = load_manifest_csv(manifest_path)
    else:
        logger.info("No manifest provided; generating synthetic training dataset...")
        paths, labels, _, _ = create_synthetic_benchmark_dataset(
            base_dir="artifacts/synthetic_dataset",
            samples_per_class=getattr(args, "samples_per_class", 3)
        )

    profiler = VideoProfiler()
    s1_features, s2_features = [], []
    stage2 = Stage2Classifier(classes=DEFAULT_CLASSES)

    for p in paths:
        meta = profiler.probe(p)
        frames = profiler.extract_frames(p, num_frames=4)
        signals = profiler.compute_coarse_signals(frames)
        vec1 = Stage1Features.transform(meta, signals)
        s1_features.append(vec1)

        emb_s2 = stage2.feature_extractor.extract_frame_embedding(frames)
        s2_features.append(emb_s2)

    X_s1 = np.array(s1_features, dtype=np.float32)
    X_s2 = np.array(s2_features, dtype=np.float32)

    stage1 = Stage1Classifier(classes=DEFAULT_CLASSES).fit(X_s1, labels)
    stage2.fit(X_s2, labels)

    cascade = CascadedClassifier(
        stage1=stage1,
        stage2=stage2,
        escalation_threshold=getattr(args, "escalation_threshold", 0.85)
    )
    cascade.save(save_dir)
    print(f"Successfully trained and saved cascade model to {save_dir}!")


def handle_benchmark_classifier_cascade(args) -> None:
    """Execute Task 1 benchmark classifier cascade comparing all 4 architectures."""
    print("=" * 75)
    print("AINTROPY TASK 1: CASCADED HYBRID CLASSIFIER BENCHMARK")
    print("=" * 75)

    # 1. Dataset verification audit
    print("\n[Step 1/5] Verifying dataset labels and ground truth...")
    audit = inspect_videomme_dataset("test.jsonl")
    if audit.get("available"):
        print(f"  * VideoMME sample detected: {audit['unique_videos']} unique videos, {audit['total_qa_records']} QA records.")
        print(f"  * Label limitation confirmed: {audit['limitation_note']}")
    else:
        print("  * test.jsonl not found; using verified synthetic reproducible test dataset.")

    # 2. Prepare training & test data
    print("\n[Step 2/5] Preparing reproducible evaluation dataset...")
    manifest_path = getattr(args, "manifest", None)
    if manifest_path and os.path.exists(manifest_path):
        print(f"  * Loading manifest from {manifest_path}")
        all_paths, all_labels = load_manifest_csv(manifest_path)
        split_idx = int(0.7 * len(all_paths))
        train_paths, train_labels = all_paths[:split_idx], all_labels[:split_idx]
        test_paths, test_labels = all_paths[split_idx:], all_labels[split_idx:]
    else:
        print("  * Generating multi-class synthetic video suite (11 assignment classes)...")
        train_paths, train_labels, test_paths, test_labels = create_synthetic_benchmark_dataset(
            base_dir="artifacts/synthetic_dataset",
            samples_per_class=3
        )

    print(f"  * Train set: {len(train_paths)} videos across {len(set(train_labels))} classes.")
    print(f"  * Test set:  {len(test_paths)} videos across {len(set(test_labels))} classes.")

    profiler = VideoProfiler()

    # 3. Extract features for training
    print("\n[Step 3/5] Extracting training representations...")
    s1_features_train = []
    mb_features_train = []
    s2_features_train = []

    mobilenet = MobileNetV3Classifier(classes=DEFAULT_CLASSES)
    stage2 = Stage2Classifier(classes=DEFAULT_CLASSES)

    for p in train_paths:
        meta = profiler.probe(p)
        frames = profiler.extract_frames(p, num_frames=4)
        signals = profiler.compute_coarse_signals(frames)
        vec1 = Stage1Features.transform(meta, signals)
        s1_features_train.append(vec1)

        emb_mb = mobilenet.feature_extractor.extract_frame_embedding(frames)
        mb_features_train.append(emb_mb)

        emb_s2 = stage2.feature_extractor.extract_frame_embedding(frames)
        s2_features_train.append(emb_s2)

    X_s1 = np.array(s1_features_train, dtype=np.float32)
    X_mb = np.array(mb_features_train, dtype=np.float32)
    X_s2 = np.array(s2_features_train, dtype=np.float32)

    # 4. Fit models
    print("\n[Step 4/5] Training baseline models and cascade head...")
    stage1 = Stage1Classifier(classes=DEFAULT_CLASSES).fit(X_s1, train_labels)
    mobilenet.fit(X_mb, train_labels)
    stage2.fit(X_s2, train_labels)

    cascade = CascadedClassifier(
        stage1=stage1,
        stage2=stage2,
        escalation_threshold=getattr(args, "escalation_threshold", 0.85),
        fallback_threshold=0.35,
        profiler=profiler,
    )

    # 5. Execute benchmark on test split
    print("\n[Step 5/5] Evaluating test split across all 4 architectures...")
    runner = BenchmarkRunner(profiler=profiler)
    save_dir = getattr(args, "output_dir", None) or "artifacts/task1_benchmark"
    reports = runner.run_all_baselines(
        test_paths,
        test_labels,
        stage1_model=stage1,
        mobilenet_model=mobilenet,
        stage2_model=stage2,
        cascade_model=cascade,
        save_dir=save_dir,
    )

    # Display comparison table
    print("\n" + "=" * 75)
    print("BENCHMARK SCORECARD: ARCHITECTURAL COMPARISON")
    print("=" * 75)
    print(runner.format_markdown_table(reports))
    print("=" * 75)

    casc_report = reports.get("cascaded_hybrid")
    if casc_report:
        print(f"\nCascaded Hybrid Highlights:")
        print(f"  * Macro-F1:            {casc_report.macro_f1:.4f}")
        print(f"  * P50 / P95 Latency:   {casc_report.p50_latency_ms:.1f} ms / {casc_report.p95_latency_ms:.1f} ms")
        print(f"  * Escalation Rate:     {casc_report.escalation_rate_pct:.1f}%")
        print(f"  * Peak Memory:         {casc_report.peak_memory_mb:.1f} MB")
        print(f"  * Artifacts saved to:  {save_dir}/benchmark_results.json")
def handle_analyze_conversion(args) -> None:
    """Task 3: Analyze video conversion ROI, break-even trade-offs, and benchmark candidate codecs."""
    video_path = getattr(args, "video", None)
    if not video_path or not os.path.exists(video_path):
        logger.info("No video path provided or video not found; generating synthetic sample video for analysis...")
        temp_dir = os.path.join(getattr(args, "output_dir", "artifacts"), "synthetic_sample")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "sample_benchmark.mp4")
        if not os.path.exists(video_path):
            generate_synthetic_video(video_path, video_type="sports", duration_s=10.0, fps=15)

    cost_params = CostModelParameters(
        expected_vlm_passes=getattr(args, "expected_passes", 5),
        retention_months=getattr(args, "retention_months", 3.0)
    )
    profiler = VideoProfilerFFprobe()
    decision_engine = ConversionDecisionEngine(default_params=cost_params)

    # 1. Multi-codec benchmarking mode
    if getattr(args, "benchmark_codecs", False):
        runner = CodecBenchmarkRunner(output_dir=getattr(args, "output_dir", "derived"), cost_params=cost_params)
        scorecard = runner.benchmark(video_path, content_genre=getattr(args, "genre", None))
        if getattr(args, "json", False):
            print(json.dumps(scorecard.to_dict(), indent=2))
        else:
            print("=" * 78)
            print("AINTROPY TASK 3: MULTI-CODEC BENCHMARK SCORECARD")
            print("=" * 78)
            print(f"Video: {video_path} ({scorecard.duration_s:.1f}s, {scorecard.original_size_mb:.2f} MB, codec: {scorecard.original_codec})")
            print(scorecard.format_markdown_table())
            print("=" * 78)
            if scorecard.recommended_profile:
                print(f"Recommended Profile: {scorecard.recommended_profile}")
                print(f"Selection Rationale: {scorecard.selection_rationale}")
            print("=" * 78)
        return

    # 2. Single-video profiling and ROI break-even analysis
    meta = profiler.probe(video_path)
    base_profile = get_profile(getattr(args, "profile", "h264_baseline"))
    adapted_profile = adapt_profile_for_content(base_profile, meta, content_genre=getattr(args, "genre", None))

    roi_est = decision_engine.evaluate_estimate(meta, profile_name=adapted_profile.name, params=cost_params)

    # 3. Transcoding execution if requested or forced
    transcode_result = None
    quality_result = None
    roi_measured = None

    should_transcode = getattr(args, "transcode", False) and (
        roi_est.recommendation == DecisionRecommendation.TRANSCODE or getattr(args, "force", False)
    )

    if should_transcode:
        engine = TranscodeEngine(default_output_dir=getattr(args, "output_dir", "derived"))
        transcode_result = engine.transcode(
            video_path, adapted_profile, duration_s=meta.duration_s
        )
        if transcode_result.success:
            out_meta = profiler.probe(transcode_result.output_path)
            quality_result = measure_video_quality(video_path, transcode_result.output_path)
            roi_measured = decision_engine.evaluate_measured(
                meta, out_meta, transcode_result.elapsed_time_s, params=cost_params
            )

    output = {
        "video": meta.to_dict(),
        "profile": adapted_profile.to_dict(),
        "roi_estimate": roi_est.to_dict(),
        "transcode": transcode_result.to_dict() if transcode_result else None,
        "quality": quality_result.to_dict() if quality_result else None,
        "roi_measured": roi_measured.to_dict() if roi_measured else None,
    }

    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
        return

    print("=" * 78)
    print("AINTROPY TASK 3: VIDEO CONVERSION & BREAK-EVEN ANALYSIS")
    print("=" * 78)
    print(f"Video ID:           {meta.video_id}")
    print(f"Original File:      {video_path} ({meta.file_size_mb:.2f} MB, {meta.duration_s:.1f}s)")
    print(f"Dimensions & FPS:   {meta.width}x{meta.height} @ {meta.fps} FPS (Codec: {meta.codec})")
    print(f"Bitrate:            {meta.bitrate_kbps:.1f} kbps")
    print("-" * 78)
    print(f"Target Profile:     {adapted_profile.name} (Codec: {adapted_profile.video_codec}, CRF: {adapted_profile.crf}, Preset: {adapted_profile.preset})")
    print(f"Recommendation:     {roi_est.recommendation.value}")
    print(f"Rationale:          {roi_est.rationale}")
    print("-" * 78)
    print("PRE-TRANSCODE ANALYTICAL ESTIMATE (PRIORS):")
    print(f"  * Expected Size:         {roi_est.output_size_mb:.2f} MB (Compression Ratio: {roi_est.compression_ratio:.2f})")
    print(f"  * Estimated Transcode:   {roi_est.time_to_encode_s:.1f}s (Compute Cost: ${roi_est.encoding_cost_usd:.4f})")
    print(f"  * Storage Savings:       ${roi_est.storage_savings_usd:.4f} (over {cost_params.retention_months:.1f} months)")
    print(f"  * I/O Time Savings:      {roi_est.io_time_savings_s:+.2f}s ({cost_params.expected_vlm_passes} passes @ {cost_params.io_transfer_speed_mb_per_s:.0f} MB/s)")
    print(f"  * Net Financial Savings: ${roi_est.net_financial_savings_usd:.4f} (Break-even passes: {roi_est.break_even_passes})")

    if transcode_result:
        print("-" * 78)
        print("POST-TRANSCODE EMPIRICAL MEASUREMENT:")
        print(f"  * Transcode Success:     {transcode_result.success} (Backend: {transcode_result.backend})")
        print(f"  * Derived Media File:    {transcode_result.output_path}")
        print(f"  * Actual Output Size:    {transcode_result.output_size_bytes / (1024*1024):.2f} MB (Actual Ratio: {transcode_result.compression_ratio:.2f})")
        print(f"  * Wall Encode Time:      {transcode_result.elapsed_time_s:.2f}s (Speed Factor: {transcode_result.speed_factor:.1f}x)")
        if quality_result:
            print(f"  * Measured PSNR / SSIM:  {quality_result.psnr_db:.1f} dB / {quality_result.ssim:.3f} (Meets quality: {quality_result.meets_constraints})")
            print(f"  * Edge/OCR Preservation: {quality_result.edge_preservation_ratio:.2f}")
        if roi_measured:
            print(f"  * Measured Net Savings:  ${roi_measured.net_financial_savings_usd:.4f}")
    print("=" * 78)


def handle_chunk_video(args) -> None:
    """Execute content-aware adaptive video chunking and emit hierarchical JSON manifest."""
    video_path = getattr(args, "video", None)
    if not video_path or not os.path.exists(video_path):
        logger.info("No video path provided or video not found; generating synthetic sample video for chunking...")
        temp_dir = os.path.join("artifacts", "synthetic_sample")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "sample_chunk_input.mp4")
        if not os.path.exists(video_path):
            generate_synthetic_video(video_path, video_type=getattr(args, "type", "sports"), duration_s=45.0, fps=15)

    policy = get_policy(
        video_type=getattr(args, "type", None) or getattr(args, "policy", "general"),
        target_duration_s=getattr(args, "target_dur", None),
        min_duration_s=getattr(args, "min_dur", None),
        max_duration_s=getattr(args, "max_dur", None),
        overlap_ratio=getattr(args, "overlap", None),
    )

    manifest = chunk_video(
        video_path=video_path,
        video_type=policy.name,
        output_path=getattr(args, "output", None),
        target_duration_s=policy.target_duration_s,
        min_duration_s=policy.min_duration_s,
        max_duration_s=policy.max_duration_s,
        overlap_ratio=policy.overlap_ratio,
    )

    if getattr(args, "json", False):
        print(json.dumps(manifest.to_dict(), indent=2))
        return

    print("=" * 80)
    print("AINTROPY TASK 4: CONTENT-AWARE ADAPTIVE VIDEO CHUNKING")
    print("=" * 80)
    print(f"Video ID:                {manifest.video_id}")
    print(f"Video File:              {manifest.video_path}")
    print(f"Total Duration:          {manifest.total_duration_s:.2f}s")
    print(f"Policy / Strategy:       {manifest.policy_name} ({manifest.strategy})")
    print(f"Target Window:           {policy.target_duration_s:.1f}s (Min: {policy.min_duration_s:.1f}s, Max: {policy.max_duration_s:.1f}s)")
    print(f"Configured Overlap:      {policy.overlap_ratio * 100:.1f}%")
    print("-" * 80)
    print(f"Generated Child Chunks:  {manifest.num_chunks}")
    print(f"Generated Parent Groups: {manifest.num_parents}")
    print(f"Average Chunk Duration:  {manifest.average_chunk_duration_s:.2f}s")
    print(f"Measured Overlap Ratio:  {manifest.overlap_ratio * 100:.1f}%")
    print(f"Processing Wall Time:    {manifest.processing_time_s:.3f}s")
    print(f"Throughput Factor:       {manifest.throughput_factor:.1f}x real-time")
    print("-" * 80)
    print("SAMPLE CHUNKS (First 5):")
    for c in manifest.chunks[:5]:
        print(f"  [{c.chunk_id}] {c.start_s:6.2f}s -> {c.end_s:6.2f}s ({c.duration_s:5.2f}s) | Boundary: {c.boundary_type:15s} (conf: {c.boundary_confidence:.2f}) | Parent: {c.parent_id}")
    if len(manifest.chunks) > 5:
        print(f"  ... ({len(manifest.chunks) - 5} more chunks)")
    print("-" * 80)
    print("PARENT HIERARCHICAL SEGMENTS:")
    for p in manifest.parent_segments:
        print(f"  [{p.segment_id}] {p.start_s:6.2f}s -> {p.end_s:6.2f}s ({p.duration_s:5.2f}s) | Children: {len(p.child_chunk_ids)} chunks ({', '.join(p.child_chunk_ids[:3])}...)")
    print("=" * 80)


def handle_benchmark_chunker(args) -> None:
    """Benchmark intelligent chunker against fixed-duration and blind scene-cut baselines."""
    video_path = getattr(args, "video", None)
    temp_video_created = False

    gt_cuts = None
    if not video_path or not os.path.exists(video_path):
        logger.info("No video provided or video missing; creating multi-scene benchmark video with ground truth transitions...")
        temp_dir = os.path.join(getattr(args, "output_dir", "artifacts"), "synthetic_chunk_test")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, "benchmark_multiscene.mp4")
        from src.chunker.benchmarks import generate_multiscene_benchmark_video
        video_path, gt_cuts = generate_multiscene_benchmark_video(video_path, duration_s=120.0, scene_interval_s=20.0)
        temp_video_created = True

    video_type = getattr(args, "type", "news_lecture")
    output_dir = getattr(args, "output_dir", "artifacts/chunker_benchmark")

    runner = ChunkerBenchmarkRunner(tolerance_s=2.0)
    logger.info("Running Task 4 Video Chunking Benchmark on %s (%s)...", video_path, video_type)
    scorecards = runner.run_benchmark(
        video_path=video_path,
        video_type=video_type,
        ground_truth_cuts=gt_cuts,
        output_dir=output_dir,
    )

    if getattr(args, "json", False):
        print(json.dumps({k: v.to_dict() for k, v in scorecards.items()}, indent=2))
        return

    print("\n" + "=" * 100)
    print("AINTROPY TASK 4: VIDEO CHUNKING COMPARATIVE BENCHMARK SCORECARD")
    print("=" * 100)
    print(runner.format_markdown_table(scorecards))
    print("=" * 100 + "\n")
    logger.info("Chunking benchmark complete. Results saved to '%s'.", output_dir)


def handle_build_benchmark(args) -> None:
    """Build multi-dataset retrieval benchmark manifest with anti-leakage group splits."""
    builder = RetrievalBenchmarkBuilder(random_seed=args.seed)
    logger.info("Building multi-dataset retrieval benchmark (Seed: %d)...", args.seed)

    manifest = builder.build_benchmark(
        videomme_path=getattr(args, "videomme", "test.jsonl"),
        dataset_a_path=getattr(args, "dataset_a", None),
        dataset_b_path=getattr(args, "dataset_b", None),
        dataset_c_path=getattr(args, "dataset_c", None),
        train_ratio=getattr(args, "train_ratio", 0.60),
        val_ratio=getattr(args, "val_ratio", 0.20),
        test_ratio=getattr(args, "test_ratio", 0.20),
    )

    output_path = getattr(args, "output", None) or "artifacts/retrieval_benchmark.json"
    manifest.save(output_path)
    logger.info("Saved benchmark manifest to '%s'", output_path)

    if getattr(args, "json", False):
        print(json.dumps(manifest.to_dict(), indent=2))
        return

    print("=" * 80)
    print("AINTROPY TASK 5: RETRIEVAL BENCHMARK BUILDER MANIFEST SUMMARY")
    print("=" * 80)
    print(f"Benchmark ID:            {manifest.benchmark_id}")
    print(f"Total Unique Queries:    {manifest.total_queries}")
    print(f"Total Unique Videos:     {manifest.total_videos}")
    print(f"Datasets Included:       {', '.join(manifest.datasets_included)}")
    print("-" * 80)
    print("STRATIFICATION BREAKDOWN:")
    strat = manifest.stratification_summary
    print(f"  * Datasets:            {json.dumps(strat.get('dataset_counts', {}))}")
    print(f"  * Temporal Intents:    {json.dumps(strat.get('intent_counts', {}))}")
    print(f"  * Difficulties:        {json.dumps(strat.get('difficulty_counts', {}))}")
    print(f"  * Modalities:          {json.dumps(strat.get('modality_counts', {}))}")
    print("-" * 80)
    print("ANTI-LEAKAGE SPLIT PARTITIONS (GroupKFold by video_id):")
    for sname, sdata in manifest.splits.items():
        print(f"  * Split '{sname:5s}': {len(sdata.queries):4d} queries across {len(sdata.video_ids):3d} unique videos (0% leakage)")
    print(f"Output File:             {output_path}")
    print("=" * 80)


def handle_evaluate_benchmark(args) -> None:
    """Evaluate retrieval ranking and end-to-end QA alignment on benchmark splits."""
    bench_path = getattr(args, "benchmark", None) or "artifacts/retrieval_benchmark.json"
    if not os.path.exists(bench_path):
        logger.info("Benchmark '%s' not found. Generating default benchmark first...", bench_path)
        builder = RetrievalBenchmarkBuilder(random_seed=42)
        manifest = builder.build_benchmark(videomme_path="test.jsonl")
        manifest.save(bench_path)
    else:
        manifest = BenchmarkManifest.load(bench_path)

    evaluator = RetrievalEvaluator()
    top_k = getattr(args, "top_k", 10)
    output_path = getattr(args, "output", None) or "artifacts/retrieval_metrics.json"

    preds_path = getattr(args, "predictions", None)
    if preds_path and os.path.exists(preds_path):
        logger.info("Evaluating supplied predictions from '%s'...", preds_path)
        with open(preds_path, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        preds = [QueryRetrievalPrediction(**p) for p in (p_data if isinstance(p_data, list) else p_data.get("predictions", []))]
        test_queries = manifest.splits["test"].queries
        scorecard = evaluator.evaluate_predictions(
            queries=test_queries,
            predictions=preds,
            evaluator_name=os.path.basename(preds_path),
            evaluation_mode="external_predictions",
        )
    else:
        logger.info("Running baseline Hybrid RRF Retrieval & Transfer Evaluation on test split...")
        transfer_evaluator = CrossDatasetTransferEvaluator(evaluator=evaluator)
        scorecard = transfer_evaluator.evaluate_mixed_corpus(
            manifest=manifest,
            split_name="test",
            top_k=top_k,
        )

    scorecard.save(output_path)
    logger.info("Saved evaluation scorecard to '%s'", output_path)

    if getattr(args, "json", False):
        print(json.dumps(scorecard.to_dict(), indent=2))
        return

    print("\n" + "=" * 90)
    print(evaluator.format_markdown_scorecard(scorecard))
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="aintropy",
        description="AIntropy Video Intelligence Engine & Evaluation Suite"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- Task 2: Intent Classification Verbs ---
    p_route = subparsers.add_parser("route-query", help="Route natural language query to temporal granularity")
    p_route.add_argument("--query", "-q", type=str, required=True, help="User query text")
    p_route.add_argument("--model-dir", type=str, default="artifacts", help="Model directory")
    p_route.add_argument("--ambiguity-mode", choices=["ambiguous", "needs_review"], default="ambiguous", help="Label to return when confidence is low")
    p_route.add_argument("--json", action="store_true", help="Print JSON output")

    p_train_intent = subparsers.add_parser("train-intent", help="Train Task 2 intent classifier models")
    p_train_intent.add_argument("--data", type=str, default="test.jsonl", help="Input VideoMME test.jsonl path")
    p_train_intent.add_argument("--dataset-out", type=str, default="data/intent_dataset.json", help="Output path for split dataset")
    p_train_intent.add_argument("--output-dir", type=str, default="artifacts", help="Model artifact output directory")
    p_train_intent.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_train_intent.add_argument("--c1", type=float, default=1.0, help="Stage 1 regularization C")
    p_train_intent.add_argument("--c2", type=float, default=1.0, help="Stage 2 regularization C")
    p_train_intent.add_argument("--alpha", type=float, default=0.85, help="Posterior blending weight")
    p_train_intent.add_argument("--escalation-th", type=float, default=0.82, help="Escalation confidence threshold")
    p_train_intent.add_argument("--margin-th", type=float, default=0.30, help="Escalation margin threshold")
    p_train_intent.add_argument("--ambiguity-th", type=float, default=0.45, help="Ambiguity threshold")
    p_train_intent.add_argument("--ambiguity-mode", choices=["ambiguous", "needs_review"], default="ambiguous")

    p_bench_intent = subparsers.add_parser("benchmark-intent-cascade", help="Benchmark Task 2 intent cascade against baselines")
    p_bench_intent.add_argument("--data", type=str, default="data/intent_dataset.json", help="Dataset manifest path")
    p_bench_intent.add_argument("--model-dir", type=str, default="artifacts", help="Model directory")
    p_bench_intent.add_argument("--output-dir", type=str, default="artifacts", help="Output directory for reports")
    p_bench_intent.add_argument("--seed", type=int, default=42, help="Random seed")
    p_bench_intent.add_argument("--ambiguity-mode", choices=["ambiguous", "needs_review"], default="ambiguous")

    # --- Task 1: Video Classification Verbs ---
    p_bench_t1 = subparsers.add_parser("benchmark-classifier-cascade", help="Benchmark Task 1 video type classifier cascade")
    p_bench_t1.add_argument("--manifest", type=str, default=None, help="Manifest for test videos")
    p_bench_t1.add_argument("--escalation-threshold", type=float, default=0.85, help="Escalation threshold")
    p_bench_t1.add_argument("--output-dir", type=str, default="artifacts/task1_benchmark", help="Output directory")

    p_pred_t1 = subparsers.add_parser("predict-type", aliases=["predict"], help="Predict video type for an incoming video file")
    p_pred_t1.add_argument("--video", type=str, required=True, help="Path to video file")
    p_pred_t1.add_argument("--model-dir", type=str, default="artifacts/cascade_model", help="Path to saved cascade model")
    p_pred_t1.add_argument("--escalation-threshold", type=float, default=0.85, help="Escalation threshold")
    p_pred_t1.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    p_pred_t1.add_argument("--no-cache", action="store_true", help="Bypass feature cache")

    p_train_t1 = subparsers.add_parser("train-classifier", aliases=["train"], help="Train Task 1 video type classifier")
    p_train_t1.add_argument("--manifest", type=str, default=None, help="Path to labels CSV")
    p_train_t1.add_argument("--save-dir", type=str, default="artifacts/cascade_model", help="Directory to save model checkpoints")
    p_train_t1.add_argument("--samples-per-class", type=int, default=3, help="Samples per class if generating synthetic dataset")
    p_train_t1.add_argument("--escalation-threshold", type=float, default=0.85, help="Escalation threshold")

    p_inspect = subparsers.add_parser("inspect-dataset", help="Audit VideoMME dataset labels and report limitations")
    p_inspect.add_argument("--jsonl", type=str, default="test.jsonl", help="Path to test.jsonl")

    # --- Task 3: Lossy Conversion & Break-Even Verbs ---
    p_conv = subparsers.add_parser("analyze-conversion", aliases=["transcode"], help="Analyze video conversion ROI, break-even trade-offs, and benchmark codecs")
    p_conv.add_argument("--video", "-v", type=str, default=None, help="Path to video file (optional: generates benchmark sample if omitted)")
    p_conv.add_argument("--dry-run", action="store_true", help="Perform pre-transcode break-even analysis without re-encoding")
    p_conv.add_argument("--transcode", action="store_true", help="Execute transcoding if economically beneficial")
    p_conv.add_argument("--force", action="store_true", help="Force transcoding even if recommendation is BYPASS or KEEP_ORIGINAL")
    p_conv.add_argument("--benchmark-codecs", action="store_true", help="Benchmark candidate codecs on video")
    p_conv.add_argument("--profile", type=str, default="h264_baseline", choices=["h264_baseline", "h265_storage", "av1_efficient", "opencv_fallback"], help="Target codec profile")
    p_conv.add_argument("--genre", type=str, default=None, help="Video genre/domain for content-aware adaptation (e.g. sports, meetings)")
    p_conv.add_argument("--expected-passes", type=int, default=5, help="Expected downstream VLM passes (default: 5)")
    p_conv.add_argument("--retention-months", type=float, default=3.0, help="Planned storage retention in months (default: 3.0)")
    p_conv.add_argument("--output-dir", type=str, default="derived", help="Destination directory for derived media")
    p_conv.add_argument("--json", action="store_true", help="Output results in JSON format")

    # --- Task 4: Intelligent Video Chunking Verbs ---
    p_chunk = subparsers.add_parser("chunk-video", aliases=["chunk"], help="Partition video into content-aware adaptive chunks")
    p_chunk.add_argument("--video", "-v", type=str, default=None, help="Path to input video file (optional: generates benchmark sample if omitted)")
    p_chunk.add_argument("--type", "-t", type=str, default="general", help="Video domain category (meetings, sports, news_lecture, etc.)")
    p_chunk.add_argument("--policy", type=str, default=None, help="Policy name or preset")
    p_chunk.add_argument("--output", "-o", type=str, default=None, help="Destination JSON path for chunk manifest")
    p_chunk.add_argument("--target-dur", type=float, default=None, help="Target chunk duration in seconds")
    p_chunk.add_argument("--min-dur", type=float, default=None, help="Minimum chunk duration in seconds")
    p_chunk.add_argument("--max-dur", type=float, default=None, help="Maximum chunk duration in seconds")
    p_chunk.add_argument("--overlap", type=float, default=None, help="Sliding overlap ratio (e.g. 0.10 for 10%)")
    p_chunk.add_argument("--json", action="store_true", help="Output full JSON manifest to stdout")

    p_bench_chunk = subparsers.add_parser("benchmark-chunker", help="Benchmark intelligent chunker against fixed and blind baselines")
    p_bench_chunk.add_argument("--video", "-v", type=str, default=None, help="Path to input video file")
    p_bench_chunk.add_argument("--type", "-t", type=str, default="news_lecture", help="Video domain category")
    p_bench_chunk.add_argument("--output-dir", type=str, default="artifacts/chunker_benchmark", help="Directory to save benchmark reports")
    p_bench_chunk.add_argument("--json", action="store_true", help="Output benchmark scorecard in JSON format")

    # --- Task 5: Retrieval Benchmark Builder & Evaluator Verbs ---
    p_build_bench = subparsers.add_parser("build-benchmark", help="Build multi-dataset retrieval benchmark manifest")
    p_build_bench.add_argument("--videomme", type=str, default="test.jsonl", help="Path to VideoMME test.jsonl")
    p_build_bench.add_argument("--dataset-a", type=str, default=None, help="Path to Dataset A (Corporate) manifest")
    p_build_bench.add_argument("--dataset-b", type=str, default=None, help="Path to Dataset B (Dynamic) manifest")
    p_build_bench.add_argument("--dataset-c", type=str, default=None, help="Path to Dataset C (Surveillance) manifest")
    p_build_bench.add_argument("--output", "-o", type=str, default="artifacts/retrieval_benchmark.json", help="Destination path for benchmark manifest")
    p_build_bench.add_argument("--seed", type=int, default=42, help="Random seed for GroupKFold split partitioning")
    p_build_bench.add_argument("--train-ratio", type=float, default=0.60, help="Train ratio (default: 0.60)")
    p_build_bench.add_argument("--val-ratio", type=float, default=0.20, help="Validation ratio (default: 0.20)")
    p_build_bench.add_argument("--test-ratio", type=float, default=0.20, help="Test ratio (default: 0.20)")
    p_build_bench.add_argument("--json", action="store_true", help="Output JSON manifest to stdout")

    p_eval_bench = subparsers.add_parser("evaluate-benchmark", help="Evaluate retrieval ranking and QA predictions on benchmark")
    p_eval_bench.add_argument("--benchmark", "-b", type=str, default="artifacts/retrieval_benchmark.json", help="Path to benchmark manifest JSON")
    p_eval_bench.add_argument("--predictions", "-p", type=str, default=None, help="Path to retrieval predictions JSON")
    p_eval_bench.add_argument("--output", "-o", type=str, default="artifacts/retrieval_metrics.json", help="Destination path for scorecard JSON")
    p_eval_bench.add_argument("--top-k", type=int, default=10, help="Top-K cut-off for ranking evaluation (default: 10)")
    p_eval_bench.add_argument("--json", action="store_true", help="Output scorecard JSON to stdout")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train-intent":
        handle_train_intent(args)
    elif args.command == "route-query":
        handle_route_query(args)
    elif args.command == "benchmark-intent-cascade":
        handle_benchmark_intent(args)
    elif args.command == "benchmark-classifier-cascade":
        handle_benchmark_classifier_cascade(args)
    elif args.command in ["predict-type", "predict"]:
        handle_predict_type(args)
    elif args.command in ["train-classifier", "train"]:
        handle_train_classifier(args)
    elif args.command == "inspect-dataset":
        handle_inspect_dataset(args)
    elif args.command in ["analyze-conversion", "transcode"]:
        handle_analyze_conversion(args)
    elif args.command in ["chunk-video", "chunk"]:
        handle_chunk_video(args)
    elif args.command == "benchmark-chunker":
        handle_benchmark_chunker(args)
    elif args.command == "build-benchmark":
        handle_build_benchmark(args)
    elif args.command == "evaluate-benchmark":
        handle_evaluate_benchmark(args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
