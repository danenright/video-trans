# Video RAG Pipeline

A robust video processing pipeline that converts MP4 files into LLM-ready JSONL chunks for Retrieval-Augmented Generation (RAG) applications.

## Features

- **Speech-to-Text**: Transcription using faster-whisper with batched inference (3-4x speedup)
- **Scene Detection**: Automatic scene boundary detection using PySceneDetect
- **Keyframe Extraction**: Extract representative frames (start, mid, end) from each scene
- **OCR**: Extract on-screen text using Tesseract with preprocessing for improved accuracy
- **Visual Captioning**: Generate image descriptions using BLIP
- **Chunk Assembly**: Align transcript + OCR + captions into RAG-ready chunks

## Prerequisites

> **You MUST install these system dependencies BEFORE installing the Python package.**

### macOS (Homebrew)

```bash
brew install ffmpeg tesseract
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install ffmpeg tesseract-ocr
```

### Windows (Chocolatey)

```bash
choco install ffmpeg tesseract
```

### Verify Installation

```bash
# Both commands should return version info
ffmpeg -version
tesseract --version
```

## Installation

### Python Package

> **Note**: On macOS, use `python3` instead of `python`. The `python` command may not exist by default.

> **Important**: Use a dedicated virtual environment to avoid dependency conflicts.
> This package requires numpy 2.x and pydantic 2.x which may conflict with older packages like spacy, pandas<2, or numba.

```bash
# Create a FRESH virtual environment (recommended)
python3 -m venv vrag-env
source vrag-env/bin/activate  # Linux/macOS
# or: vrag-env\Scripts\activate  # Windows

# Navigate to package directory
cd video_rag_pipeline

# Install package
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

#### Verifying Installation

```bash
# Check imports work
python3 -c "from vrag.schema import Chunk; print('OK')"

# Check CLI works
python3 run_pipeline.py --help
```

## Quick Start

1. **Place videos** in `./videos/` directory (or configure input path)

2. **Run the pipeline**:
```bash
python3 run_pipeline.py --input ./videos --output ./output
```

3. **Find outputs** in `./output/<video_id>/`:
   - `chunks.jsonl` - Primary RAG payload
   - `transcript.json` - Raw transcription
   - `scenes.json` - Detected scene boundaries
   - `frames/` - Extracted keyframes
   - `ocr.json` - OCR results
   - `captions.json` - Visual captions

## Configuration

Edit `config.yaml` to customize pipeline behavior:

```yaml
# Key settings
transcription:
  model: "medium"  # tiny, base, small, medium, large-v3
  batch_enabled: true
  batch_size: 16

scenes:
  threshold: 27.0  # Lower = more sensitive
  min_scene_len_seconds: 6

crop:
  enabled: true
  slide_region: [0.05, 0.12, 0.78, 0.92]  # [left, top, right, bottom]

ocr:
  enabled: true
  preprocessing:
    upscale: 3.0
    clahe: true

caption:
  enabled: true
  model: "Salesforce/blip-image-captioning-base"
```

### Tuning the Crop Region

The crop region isolates slide content from player UI elements:

1. Run pipeline on one video
2. Check frames in `output/<video>/frames/`
3. Adjust `crop.slide_region` in `config.yaml`
4. Re-run with `--force` to regenerate

## Output Format

Each line in `chunks.jsonl`:

```json
{
  "chunk_id": "UNIT_01_abc123",
  "video_id": "UNIT_01_The_Footprint_Tool",
  "start": 60.5,
  "end": 120.8,
  "text": "...transcript...\n\n[ON-SCREEN TEXT]\n...ocr...\n\n[VISUAL SUMMARY]\n...caption...",
  "transcript": "raw transcript text",
  "ocr_text": "extracted on-screen text",
  "visual_caption": "image description",
  "metadata": {"scene_idx": 5, "source": "video_course"}
}
```

## CLI Options

```bash
python3 run_pipeline.py --help

Options:
  -i, --input PATH    Input directory with MP4 files
  -o, --output PATH   Output directory for artifacts
  -c, --config PATH   Path to config.yaml
  -f, --force         Force re-processing (ignore cache)
  --filter PATTERN    Video filename filter (glob)
  -v, --verbose       Enable debug logging
```

## Programmatic Usage

```python
from run_pipeline import run_pipeline, load_config

# Load custom config
config = load_config("my_config.yaml")

# Process videos
results = run_pipeline(
    input_dir="./videos",
    output_dir="./output",
    config=config,
    force=False,
)

# Check results
for result in results:
    print(f"{result['video_id']}: {result['chunks']} chunks")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=vrag --cov-report=html

# Type checking
mypy vrag/

# Linting
ruff check vrag/
```

## Project Structure

```
video_rag_pipeline/
├── config.yaml          # Configuration
├── run_pipeline.py      # Main orchestrator
├── vrag/
│   ├── schema.py        # Pydantic models
│   ├── io.py            # I/O utilities
│   ├── ffmpeg_utils.py  # Audio extraction
│   ├── transcribe.py    # Speech-to-text
│   ├── scene_detect.py  # Scene detection
│   ├── frames.py        # Frame extraction
│   ├── crop.py          # Image cropping
│   ├── ocr.py           # OCR processing
│   ├── caption.py       # Visual captioning
│   └── chunk.py         # Chunk assembly
└── tests/               # Test suite
```

## Requirements

- Python 3.10+
- ffmpeg (system)
- tesseract (system)
- GPU recommended for faster processing

## Troubleshooting

### Dependency Conflicts During Installation

If you see errors like:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
spacy 3.5.2 requires pydantic<1.11.0, but you have pydantic 2.11.5 which is incompatible.
pandas requires numpy<2, but you have numpy 2.2.6 which is incompatible.
```

**Solution**: Always use a fresh virtual environment:
```bash
# Create isolated environment
python3 -m venv vrag-env
source vrag-env/bin/activate
pip install -e .
```

This package requires modern versions of numpy (2.x) and pydantic (2.x) which conflict with older ML packages. A dedicated environment avoids these issues.

### ffmpeg/tesseract Not Found

Ensure system dependencies are installed and in your PATH:
```bash
# Verify ffmpeg
ffmpeg -version

# Verify tesseract
tesseract --version
```

### CUDA/GPU Issues

If transcription or captioning is slow, check GPU availability:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True for GPU
```

For CPU-only systems, the pipeline will work but be slower. Consider using smaller models:
```yaml
# config.yaml
transcription:
  model: "small"  # Instead of "medium" or "large"
caption:
  enabled: false  # Disable if too slow
```

## License

MIT
