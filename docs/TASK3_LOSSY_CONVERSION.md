# Task 3 Technical Write-up: Metadata-Driven Lossy Video Conversion & Break-Even ROI Engine

## 1. Executive Summary & Problem Formulation
Video transcoding is computationally expensive and introduces generational quality loss. The **Lossy Conversion Engine** analyzes whether an incoming video should be transcoded to a secondary representation for downstream VLM captioning, calculates the break-even ROI across time and financial cost, and enforces non-destructive storage.

## 2. Analytical Decision Engine & Mathematical Criteria
Let:
- $S$: Original video size (MB)
- $D$: Video duration (seconds)
- $r$: Compression ratio (compressed size / original size)
- $v$: Transcoding speed factor (encoded seconds / real second)
- $p$: Expected downstream query/VLM read passes
- $t_{\text{io}}$: I/O transfer time per MB
- $c_s$: Storage cost ($/MB/month)
- $c_{\text{read}}$: Network egress/read cost ($/MB)
- $c_{\text{compute}}$: Compute cost ($/second)

### 2.1 Break-Even Equations
1. **Time Break-Even**:
   $$\Delta T = S \cdot (1 - r) \cdot t_{\text{io}} \cdot p - \frac{D}{v} > 0$$
2. **Financial Break-Even**:
   $$\Delta C = S \cdot (1 - r) \cdot (c_s \cdot m + p \cdot c_{\text{read}}) - \frac{D}{v} \cdot c_{\text{compute}} > 0$$
3. **Smart Bypass**:
   If video bitrate < 80 MB/min and codec is already AVC/HEVC, transcoding is automatically bypassed (`BYPASS_ALREADY_OPTIMAL`).

## 3. Safe Codec Profiles & Non-Destructive Storage
- **Profiles**:
  - `h264_baseline`: CRF 23, `preset=fast`, AAC audio (universal compatibility).
  - `h265_storage`: CRF 26, `preset=medium` (maximum storage efficiency for archival).
- **Non-Destructive Guarantee**: Original uploaded media files are immutable masters. All converted outputs are isolated in `derived/`.
