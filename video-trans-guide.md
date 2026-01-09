Below is a practical “suite” design you can drop into a repo and run over a folder of MP4s to produce **LLM-digestible JSONL** (plus the intermediate artifacts you’ll want for debugging).

It’s built around this idea:

* **Transcript** (timestamps)
* **Scene/slide segments** (from the video)
* **Keyframes per scene** → **crop** → **preprocess** → **OCR**
* **Optional visual caption** (for diagrams/handwriting where OCR is weak)
* **Chunk builder** that aligns transcript + OCR + caption into RAG-ready records

---

## Repo layout

```
video_rag_pipeline/
  pyproject.toml (or requirements.txt)
  config.yaml
  run_pipeline.py

  vrag/
    io.py
    ffmpeg_utils.py
    transcribe.py
    scene_detect.py
    frames.py
    crop.py
    ocr.py
    caption.py
    chunk.py
    summarize.py   (optional)
    schema.py
```

Output:

```
output/
  UNIT_01_The_Footprint_Tool/
    meta.json
    audio.wav
    transcript.json
    scenes.json
    frames/
      scene_0001_start.jpg
      scene_0001_mid.jpg
      ...
    ocr.json
    captions.json
    chunks.jsonl        <-- primary RAG payload
    video_summary.json  (optional)
    units.jsonl         (optional “principles/how-tos/gotchas”)
```

---

## 1) `config.yaml` (tune once, run everywhere)

Because all your MP4s look like the same player UI, you can use a **fixed crop** rectangle (percent-based) that isolates the slide/canvas and removes webcam/sidebar.

```yaml
input_dir: "./videos"
output_dir: "./output"

ffmpeg_path: "ffmpeg"

transcription:
  backend: "faster-whisper"
  model: "medium"   # start with "small" for speed, upgrade if needed
  language: "en"
  vad_filter: true

scenes:
  method: "pyscenedetect"
  threshold: 27.0         # tune once (lower = more scenes)
  min_scene_len_seconds: 6

frames:
  per_scene: ["start", "mid", "end"]
  image_format: "jpg"
  jpeg_quality: 92

crop:
  # Percent-of-frame crop: left, top, right, bottom
  # Tune by exporting a couple frames and adjusting.
  slide_region: [0.05, 0.12, 0.78, 0.92]

ocr:
  enabled: true
  engine: "tesseract"
  tesseract_cmd: "tesseract"
  psm: 6
  upscale: 3.0
  preprocess: ["grayscale", "clahe", "sharpen"]

caption:
  enabled: true
  backend: "blip"     # local caption baseline
  max_new_tokens: 50

chunking:
  target_words: 350
  max_words: 650
  overlap_words: 60
  attach_ocr: true
  attach_caption: true
```

---

## 2) Install requirements

You need system deps:

* `ffmpeg`
* `tesseract` (if using OCR)

Python deps (example `requirements.txt`):

```txt
faster-whisper
torch
transformers
pyscenedetect[opencv]
opencv-python
pytesseract
Pillow
numpy
pydantic
tqdm
PyYAML
```

---

## 3) Core schemas (keeps everything consistent)

`vrag/schema.py`

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class TranscriptSeg(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None

class Scene(BaseModel):
    idx: int
    start: float
    end: float

class FrameRef(BaseModel):
    scene_idx: int
    kind: str  # start|mid|end
    path: str
    time: float

class OCRResult(BaseModel):
    frame_path: str
    text: str
    confidence: Optional[float] = None

class CaptionResult(BaseModel):
    frame_path: str
    caption: str

class Chunk(BaseModel):
    chunk_id: str
    video_id: str
    start: float
    end: float
    text: str                 # the “LLM-ready” content
    transcript: str
    ocr_text: Optional[str] = None
    visual_caption: Optional[str] = None
    metadata: Dict[str, Any]
```

---

## 4) Transcription (faster-whisper)

`vrag/transcribe.py`

```python
from faster_whisper import WhisperModel
from vrag.schema import TranscriptSeg
from typing import List
import os, json

def transcribe_audio(audio_path: str, model_name: str="small", language="en", vad_filter=True) -> List[TranscriptSeg]:
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter
    )
    out = []
    for s in segments:
        out.append(TranscriptSeg(start=float(s.start), end=float(s.end), text=s.text.strip()))
    return out

def save_transcript(path: str, segs: List[TranscriptSeg]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in segs], f, ensure_ascii=False, indent=2)
```

---

## 5) Scene/slide detection (PySceneDetect)

`vrag/scene_detect.py`

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from vrag.schema import Scene
from typing import List
import json

def detect_scenes(video_path: str, threshold: float=27.0, min_scene_len_seconds: int=6) -> List[Scene]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_seconds))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes: List[Scene] = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append(Scene(idx=i, start=start.get_seconds(), end=end.get_seconds()))
    return scenes

def save_scenes(path: str, scenes: List[Scene]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in scenes], f, ensure_ascii=False, indent=2)
```

---

## 6) Extract keyframes per scene with ffmpeg

`vrag/frames.py`

```python
import os, subprocess
from typing import List
from vrag.schema import Scene, FrameRef

def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_scene_frames(ffmpeg: str, video_path: str, scenes: List[Scene], out_dir: str, kinds=("start","mid","end")) -> List[FrameRef]:
    os.makedirs(out_dir, exist_ok=True)
    refs: List[FrameRef] = []

    for sc in scenes:
        times = {}
        if "start" in kinds: times["start"] = sc.start + 0.2
        if "mid" in kinds:   times["mid"] = (sc.start + sc.end) / 2.0
        if "end" in kinds:   times["end"] = max(sc.start, sc.end - 0.2)

        for kind, t in times.items():
            out_path = os.path.join(out_dir, f"scene_{sc.idx:04d}_{kind}.jpg")
            cmd = [ffmpeg, "-y", "-ss", str(t), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path]
            _run(cmd)
            refs.append(FrameRef(scene_idx=sc.idx, kind=kind, path=out_path, time=t))
    return refs
```

---

## 7) Crop + preprocess + OCR (robust enough for your UI videos)

`vrag/crop.py`

```python
from PIL import Image

def crop_percent(img: Image.Image, region):
    # region = [left, top, right, bottom] in 0..1
    w, h = img.size
    l = int(region[0] * w); t = int(region[1] * h)
    r = int(region[2] * w); b = int(region[3] * h)
    return img.crop((l, t, r, b))
```

`vrag/ocr.py`

```python
import cv2, pytesseract
import numpy as np
from PIL import Image
from typing import List
from vrag.schema import FrameRef, OCRResult
from vrag.crop import crop_percent

def _preprocess(pil_img: Image.Image, upscale=3.0):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if upscale and upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE improves text on dark backgrounds
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # mild sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray

def ocr_frames(frames: List[FrameRef], slide_region, psm=6, upscale=3.0) -> List[OCRResult]:
    results: List[OCRResult] = []
    config = f"--psm {psm}"

    for fr in frames:
        pil = Image.open(fr.path)
        pil = crop_percent(pil, slide_region)
        proc = _preprocess(pil, upscale=upscale)

        text = pytesseract.image_to_string(proc, config=config)
        text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])

        results.append(OCRResult(frame_path=fr.path, text=text))
    return results
```

**Why this works for you:**
You’ll get decent OCR on headings/bullets after crop+upscale+contrast, while still being fast.

---

## 8) Visual captions (covers diagrams & handwriting)

This is your “diagram insurance.” Even a basic caption model helps the RAG find the right section when OCR fails.

`vrag/caption.py` (local BLIP baseline; can swap later)

```python
from transformers import pipeline
from PIL import Image
from typing import List
from vrag.schema import FrameRef, CaptionResult
from vrag.crop import crop_percent

def caption_frames(frames: List[FrameRef], slide_region, max_new_tokens=50) -> List[CaptionResult]:
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    out: List[CaptionResult] = []

    for fr in frames:
        img = Image.open(fr.path)
        img = crop_percent(img, slide_region)
        res = pipe(img, max_new_tokens=max_new_tokens)
        caption = res[0]["generated_text"].strip()
        out.append(CaptionResult(frame_path=fr.path, caption=caption))
    return out
```

> Later, if you want better captions, you can replace this with a stronger VLM. But this gets you started without needing external services.

---

## 9) Chunk builder (RAG-ready JSONL)

This aligns transcript text to scene windows, then attaches OCR/captions from frames in the same scene.

`vrag/chunk.py`

```python
from typing import List, Dict, Optional
from vrag.schema import TranscriptSeg, Scene, FrameRef, OCRResult, CaptionResult, Chunk
import hashlib, json

def _join_transcript(segs: List[TranscriptSeg]) -> str:
    return " ".join(s.text for s in segs).strip()

def build_chunks(
    video_id: str,
    scenes: List[Scene],
    transcript: List[TranscriptSeg],
    frames: List[FrameRef],
    ocr: Optional[List[OCRResult]] = None,
    captions: Optional[List[CaptionResult]] = None,
    meta: Optional[Dict] = None,
) -> List[Chunk]:
    meta = meta or {}

    # index OCR/captions by frame_path
    ocr_map = {r.frame_path: r.text for r in (ocr or [])}
    cap_map = {r.frame_path: r.caption for r in (captions or [])}

    # group frames by scene
    frames_by_scene: Dict[int, List[FrameRef]] = {}
    for fr in frames:
        frames_by_scene.setdefault(fr.scene_idx, []).append(fr)

    chunks: List[Chunk] = []
    for sc in scenes:
        segs = [s for s in transcript if not (s.end < sc.start or s.start > sc.end)]
        transcript_text = _join_transcript(segs)

        # collect OCR/captions for this scene’s frames
        scene_frames = frames_by_scene.get(sc.idx, [])
        ocr_texts = [ocr_map.get(fr.path, "") for fr in scene_frames if ocr_map.get(fr.path)]
        cap_texts = [cap_map.get(fr.path, "") for fr in scene_frames if cap_map.get(fr.path)]

        # de-dup repeated OCR lines
        ocr_text = "\n".join(dict.fromkeys("\n".join(ocr_texts).splitlines())).strip() if ocr_texts else None
        cap_text = " | ".join(dict.fromkeys(cap_texts)).strip() if cap_texts else None

        # Construct the LLM-friendly text payload:
        parts = []
        if transcript_text:
            parts.append(transcript_text)
        if ocr_text:
            parts.append(f"[ON-SCREEN TEXT]\n{ocr_text}")
        if cap_text:
            parts.append(f"[VISUAL SUMMARY]\n{cap_text}")

        full_text = "\n\n".join(parts).strip()

        # stable id
        hid = hashlib.md5(f"{video_id}:{sc.idx}:{sc.start:.2f}:{sc.end:.2f}".encode()).hexdigest()[:12]
        chunks.append(Chunk(
            chunk_id=f"{video_id}_{hid}",
            video_id=video_id,
            start=sc.start,
            end=sc.end,
            text=full_text,
            transcript=transcript_text,
            ocr_text=ocr_text,
            visual_caption=cap_text,
            metadata={**meta, "scene_idx": sc.idx}
        ))

    return chunks

def save_chunks_jsonl(path: str, chunks: List[Chunk]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.model_dump(), ensure_ascii=False) + "\n")
```

---

## 10) Orchestrator: run the whole pipeline on a folder

`run_pipeline.py`

```python
import os, json, yaml
from tqdm import tqdm

from vrag.ffmpeg_utils import extract_audio
from vrag.transcribe import transcribe_audio, save_transcript
from vrag.scene_detect import detect_scenes, save_scenes
from vrag.frames import extract_scene_frames
from vrag.ocr import ocr_frames
from vrag.caption import caption_frames
from vrag.chunk import build_chunks, save_chunks_jsonl

def safe_id(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    in_dir = cfg["input_dir"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    mp4s = [f for f in os.listdir(in_dir) if f.lower().endswith(".mp4")]

    for fn in tqdm(mp4s, desc="Videos"):
        video_path = os.path.join(in_dir, fn)
        video_id = safe_id(os.path.splitext(fn)[0])
        vout = os.path.join(out_dir, video_id)
        os.makedirs(vout, exist_ok=True)

        meta_path = os.path.join(vout, "meta.json")
        if not os.path.exists(meta_path):
            json.dump({"video_id": video_id, "filename": fn}, open(meta_path, "w", encoding="utf-8"), indent=2)

        # 1) audio
        audio_path = os.path.join(vout, "audio.wav")
        extract_audio(cfg["ffmpeg_path"], video_path, audio_path)

        # 2) transcript
        tcfg = cfg["transcription"]
        transcript = transcribe_audio(audio_path, model_name=tcfg["model"], language=tcfg["language"], vad_filter=tcfg["vad_filter"])
        save_transcript(os.path.join(vout, "transcript.json"), transcript)

        # 3) scenes
        scfg = cfg["scenes"]
        scenes = detect_scenes(video_path, threshold=scfg["threshold"], min_scene_len_seconds=scfg["min_scene_len_seconds"])
        save_scenes(os.path.join(vout, "scenes.json"), scenes)

        # 4) frames
        fdir = os.path.join(vout, "frames")
        frames = extract_scene_frames(cfg["ffmpeg_path"], video_path, scenes, fdir, kinds=tuple(cfg["frames"]["per_scene"]))

        # 5) OCR
        ocr_res = None
        if cfg["ocr"]["enabled"]:
            ocr_res = ocr_frames(frames, cfg["crop"]["slide_region"], psm=cfg["ocr"]["psm"], upscale=cfg["ocr"]["upscale"])
            json.dump([r.model_dump() for r in ocr_res], open(os.path.join(vout, "ocr.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # 6) captions
        cap_res = None
        if cfg["caption"]["enabled"]:
            cap_res = caption_frames(frames, cfg["crop"]["slide_region"], max_new_tokens=cfg["caption"]["max_new_tokens"])
            json.dump([r.model_dump() for r in cap_res], open(os.path.join(vout, "captions.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # 7) chunks
        chunks = build_chunks(
            video_id=video_id,
            scenes=scenes,
            transcript=transcript,
            frames=frames,
            ocr=ocr_res,
            captions=cap_res,
            meta={"source": "video_course"}
        )
        save_chunks_jsonl(os.path.join(vout, "chunks.jsonl"), chunks)

if __name__ == "__main__":
    main()
```

And `vrag/ffmpeg_utils.py`:

```python
import subprocess

def extract_audio(ffmpeg: str, video_path: str, audio_out: str) -> None:
    cmd = [ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

---

## How this becomes “LLM/agent friendly”

Your main ingestion artifact is `chunks.jsonl`, where each line looks like:

```json
{
  "chunk_id": "UNIT_01_..._a1b2c3d4e5f6",
  "video_id": "UNIT_01_The_Footprint_Tool_1-17",
  "start": 1140.2,
  "end": 1282.8,
  "text": "...transcript...\n\n[ON-SCREEN TEXT]\n...ocr...\n\n[VISUAL SUMMARY]\n...caption...",
  "metadata": {"source":"video_course","scene_idx":42}
}
```

That is exactly what you feed into:

* embedding pipeline
* vector DB (plus optional keyword index)
* retrieval+rerank step for your agent

Even if you never show timestamps to users, the timestamps let you:

* debug hallucinations (“where did it get that?”)
* re-chunk
* later add citations/clips if you change your mind

---

## The one thing you’ll need to tune: the crop region

Do this once:

1. Run pipeline on **one MP4**
2. Inspect a few `output/.../frames/*.jpg`
3. Adjust `crop.slide_region` until the OCR text is mostly slide content, no UI

If you want, I can also give you a tiny helper script that exports a 3×3 grid of “crop previews” so you can pick the right rectangle quickly.

---

## Next upgrades (worth doing after MVP)

1. **Rerank** top retrieval results (big quality jump)
2. **OCR gating**: only include OCR in `text` if OCR quality passes a threshold (prevents noisy context)
3. **Derived units** (`units.jsonl`): automatically extract “principles / rules / gotchas / steps” per chunk (makes “design/build” tasks dramatically easier for agents)

If you tell me whether these videos are mostly **trading education** (your filenames suggest that) or **product/engineering workshops**, I’ll tailor the “derived units” extractor schema so the agent gets exactly the kind of reusable knowledge you want (patterns, setups, failure modes, checklists, etc.).

