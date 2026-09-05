# Speak Local

A local-only English practice tool. Record yourself speaking, get a transcript (Whisper), then a prioritized list of grammar/vocabulary errors plus a CEFR level estimate (local Llama via Ollama).

Phase 1: single user, web only, everything runs on your machine.

## Stack

- **Backend**: Rust + Axum (Cargo workspace under `backend/`)
- **Frontend**: React + Vite + TypeScript (pnpm workspace under `frontend/`)
- **Speech-to-text**: whisper-rs (whisper.cpp), model size configurable via env
- **Error analysis / CEFR**: local LLM served by Ollama (OpenAI-compatible API)
- **Design**: `TranscriptionProvider` and `AnalysisProvider` traits so cloud
  backends (OpenAI Whisper API, Claude, etc.) can be added later without
  rewriting the pipeline.

## Prerequisites

- Rust toolchain (`cargo`)
- Node.js + pnpm
- **cmake** and the **libclang** dev package — needed to compile whisper-rs
  (bindgen + whisper.cpp build):

  ```bash
  sudo apt install cmake libclang-18-dev
  ```

- **CUDA toolkit (optional)** — enables GPU inference (faster Whisper). Install
  `nvcc` + cuBLAS (CUDA 12):

  ```bash
  sudo apt install nvidia-cuda-toolkit
  ```

  Then build and run the backend with `cargo run --features cuda`. Without this
  build flag Whisper runs on CPU and `WHISPER_USE_GPU` is ignored.

## Setup

1. **Get a Whisper model** (whisper.cpp ggml format). `backend/.env` points at
   `ggml-small.en.bin` — the English-optimized "small" model, a good accuracy
   / CPU-latency balance for practice clips. Download it once:

   ```bash
   mkdir -p backend/models
   curl -L -o backend/models/ggml-small.en.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
   ```

   The repo also ships `backend/models/ggml-tiny.bin`. To run on that fast
   fallback instead (lower accuracy, sub-second latency), set `WHISPER_MODEL` to
   `models/ggml-tiny.bin` in `backend/.env`.

2. **Install Ollama and pull a model**:

   ```bash
   # https://ollama.com (installs a local daemon on :11434)
   ollama pull llama3.1:8b    # recommended (fine on 16 GB RAM, see Model tuning)
   # ollama pull llama3.2      # lighter/faster fallback
   ```

3. **Backend env** (optional — defaults already line up with the above):

   ```bash
   cp backend/.env.example backend/.env
   ```

   Note: `.env.example` points `WHISPER_MODEL` at `models/ggml-small.en.bin`. If
   you haven't downloaded that file (step 1), switch the line to
   `models/ggml-tiny.bin` so it matches the model already in `backend/models/`.

## Run

Start the backend first, then the frontend. Everything is run from the **project root**.

Terminal 1 — backend:

```bash
cargo run            # CPU (WHISPER_USE_GPU is ignored)
# or, for GPU inference (built with --features cuda; see Prerequisites):
cd backend && cargo run --features cuda
```

- Loads `backend/.env` automatically (regardless of the working directory you
  invoke it from), and relative model paths resolve against `backend/`.
- With the CUDA build, `WHISPER_USE_GPU=1` in `backend/.env` runs Whisper on the
  GPU. The startup log should show `use gpu = 1` and
  `whisper_model_load: CUDA0 total size = ...`.
- Wait for the `listening on http://127.0.0.1:8787` log line.
- Verify it's up: `curl http://127.0.0.1:8787/health` → `{"status":"ok"}`.

Terminal 2 — frontend (project root):

```bash
pnpm install   # first time only
pnpm dev
```

- `pnpm install` is only needed on a fresh checkout.
- Open the printed URL (default `http://localhost:5173`). The dev server proxies
  `/jobs` and `/health` to the backend on `127.0.0.1:8787`, so the backend must be
  running (Terminal 1) before the frontend is usable.

> Running the backend with `cd backend && cargo run` also works and is a fine
> alternative to running it from the project root.

### Troubleshooting

- **`failed to open 'models/ggml-*.bin'` / `Failed to create a new whisper
  context`** — the model file named by `WHISPER_MODEL` doesn't exist (or the path
  is wrong). Check what's present:
  `ls -lh backend/models/`. Either download that model (see Setup step 1) or point
  `WHISPER_MODEL` in `backend/.env` at a file that exists (the shipped one is
  `ggml-tiny.bin`).
- **`/health` connection refused** — the backend hasn't finished starting yet; look
  for the `listening on http://127.0.0.1:8787` line in Terminal 1.
- **`/health` returns ok but the UI can't reach it** — make sure the backend is
  still running and the frontend was started from the project root so the Vite
  proxy matches.

## How it works

1. The React UI records audio (WebM/Opus) via `MediaRecorder`.
2. The browser decodes the compressed audio to PCM with **WebAudio**
   (`AudioContext.decodeAudioData`) and re-encodes it as an uncompressed **WAV**.
3. `POST /jobs` uploads the WAV → creates an async job.
4. The backend decodes WAV/PCM with **Symphonia**, transcribes with **Whisper**,
   then analyzes with the Ollama-hosted LLM.
5. `GET /jobs/{id}` polls the status. On completion it returns:
   - `transcript`
   - `analysis.cefr_label` + `cefr_justification`
   - `analysis.errors[]` — each with `text`, `suggestion`, `category`,
     `criticality` (ordered most-critical first), `context`, `explanation`.

### Streaming (live transcript while recording)

The UI can also transcribe **while** you speak instead of only after stopping:

1. `POST /streams` creates a session → `{ "stream_id": "..." }`.
2. The frontend opens `GET /streams/{id}/ws` (WebSocket) and sends each ~2 s
   slice as an uncompressed WAV (16k mono) binary message; `"done"` closes the
   audio side.
3. Poll `GET /streams/{id}`: while recording it returns
   `{ "status": "active", "partial_text": "...", "audio_seconds": N }`. The
   partial is the **whole buffer re-transcribed as soon as ~2 s of new audio
   arrives** ("draft que refina"), never an append, so earlier words may change
   — that's expected. Chunks quieter than `STREAM_RMS_FLOOR` skip the partial to
   avoid Whisper hallucinating on pauses.
4. `POST /streams/{id}` finalizes: the full buffer runs the normal Whisper +
   LLM pipeline. Polling `GET /streams/{id}` then returns the standard
   `job`-shape response (`processing` → `completed`/`failed`).

Whisper runs a shared non-thread-safe context, so every model call — streaming
partials, finals, and whole-record `/jobs` — serializes through one lock. The LLM
analysis deliberately runs *outside* that lock, so one session's language-model
step never blocks another session's live draft. If the WebSocket fails to open,
the UI transparently falls back to the classic stop-and-upload `/jobs` flow.

## Configuration (env vars)

See `backend/.env.example`:

| Var | Default | Purpose |
| --- | --- | --- |
| `BIND_ADDR` | `127.0.0.1:8787` | Backend listen address |
| `WHISPER_MODEL` | `models/ggml-small.en.bin` | Whisper model path (relative paths resolve against `backend/`) |
| `WHISPER_USE_GPU` | `0` | Run Whisper on the GPU (requires a `--features cuda` build; the repo's `.env` sets `1`) |
| `WHISPER_INITIAL_PROMPT` | generic software-eng vocab seed | Seed vocabulary to bias the decoder (overrides the built-in base list entirely) |
| `WHISPER_VOCAB_HINTS` | *(none)* | Extra terms appended to the Whisper seed — the hook for your own product/company names without touching code |
| `WHISPER_LOW_CONF_THRESHOLD` | `0.6` | Min per-token confidence; lower tokens are flagged `«…»` in the analysis and treated as likely transcription artifacts |
| `OLLAMA_URL` | `http://localhost:11434` | LLM API base URL |
| `LLM_MODEL` | `llama3.1:8b` | Model served by Ollama (`llama3.1:8b` chosen by benchmark; see **Model tuning**) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature for extraction |
| `STREAM_RETENTION_SECS` | `600` | How long a stream session is kept before sweep |
| `STREAM_MAX_SECS` | `600` | Hard cap on recorded audio per session |
| `STREAM_PARTIAL_INTERVAL_SECS` | `2` | New audio (s) that triggers a refining partial transcribe |
| `STREAM_RMS_FLOOR` | `0.002` | Chunk RMS below which a partial is skipped (silence guard) |

Model trade-offs (all from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)):

| Model | Size | CPU time for ~30 s clip | Notes |
| --- | --- | --- | --- |
| `ggml-tiny.bin` | 75 MB | ~1–2 s | Fastest; garbles accents/pauses, can hallucinate on silence |
| `ggml-base.en.bin` | 142 MB | ~3–5 s | Mild accuracy bump, still snappy |
| `ggml-small.en.bin` | 466 MB | ~10–20 s | Recommended — much better on accented speech + punctuation |
| `ggml-medium.en.bin` | 1.5 GB | ~40–90 s | Very accurate; slow on CPU, ~2–3 GB RAM |

The `.en` variants are tuned for English-only (which is what the backend forces)
and are faster/more accurate than the multilingual files of the same size. After
swapping the model file you must **restart the backend** — it's loaded once at
startup.

## Model tuning

Backed by a measured tuning pass (this machine: Ryzen 5 1600, 16 GB RAM, GTX 1650
4 GB). Same ~150-word technical pitch (≈500 prompt tokens), analysis step only:

| LLM model (all local via Ollama) | Gen wall (s) | Errors flagged | False-positive style flags |
| --- | --- | --- | --- |
| llama3.2 | 84 | 6 | 5 (incl. invented `NestJS→Nest.js`, semicolon "grammar" @5) |
| qwen2.5:3b | 165 | 5 | 4 (verbose, 1 malformed item) |
| **llama3.1:8b** | 87 (GPU-auto) | **4** | **3** (all minor @2–3) + best CEFR (B2) |
| qwen2.5:7b | 136–173 | 6 | 5 |

- **llama3.1:8b is the default**: fewest false positives on acceptable stylistic
  variation, and it didn't invent a vocabulary error for the tech term `NestJS`.
  Ollama's automatic partial GPU offload keeps its latency on par with `llama3.2`
  (~87 s) instead of ~183 s on CPU, without OOM.
- Real end-to-end (`POST /jobs` on a ~62 s clip): **llama3.2+Whisper GPU 62.5 s**,
  **llama3.1:8b+Whisper GPU 64.5 s**, **llama3.1:8b+Whisper CPU 60.4 s**. All
  peaked ≤ ~3.1 GB VRAM with no added swap — so keeping Whisper on the GPU and
  8b analysis on GPU-auto is the recommended allocation (no inversion needed).
- Whisper `small.en` on this machine: **~5.9 s GPU** vs **~37 s CPU** per ~62 s of
  audio. The default `WHISPER_USE_GPU=1` is much faster for real recordings.
- `whisper.cpp` exposes vocab bias via `WHISPER_INITIAL_PROMPT`. It can fix
  mis-heard tech terms that are in the seed (e.g. "NestJS"), but it cannot recover
  a term transcriber mishears as a different known word. For real accuracy on
  proper nouns, re-record or correct the transcript.
- The base seed is **deliberately generic** (APIs, microservices, SQL/NoSQL,
  cloud, Docker, Kubernetes, common frameworks, …) so it stays useful for anyone.
  Add your own brand/product names with `WHISPER_VOCAB_HINTS` — it is always
  appended to the base, e.g.:
  ```
  WHISPER_VOCAB_HINTS=LuizaLabs, LIGA FACENS, LinkApi, BuildOne, Wellhub
  ```
- Since Whisper is often confidently wrong about domain terms, the backend also
  records per-token confidence and feeds the analysis a copy of the transcript
  where doubtful spans are wrapped in `«…»`. The model is instructed to treat
  those as probable transcription artifacts: it won't build a grammar rewrite
  around them or present garbled text as a speaker error. That's why a few `«…»`
  phrases can appear in the analysis prompt while the transcript you see stays
  clean — and why a flagged low-confidence span may be downgraded or omitted.
  Tune sensitivity with `WHISPER_LOW_CONF_THRESHOLD`.

The analysis prompt (`ollama.rs`) is tuned to report only real grammar/vocabulary
errors, to treat likely transcription artifacts (odd tech terms) cautiously, and
to prefer an empty `errors` list for correct English. A small post-filter also
strips measured false positives ("consisted", "a lot", "instead of", "every day",
"with Docker", … as vocabulary/awkward nits) while always passing grammar defects
through, and caps "awkward" criticality at 2.

## Notes / conventions

- **No persistence** in phase 1 (SQLx/Postgres conventions are documented in
  `docs/intent/english-practice-app.md` for phase 2).
- Backend runs natively on the host, **not containerized** (avoids the
  `network_mode: host` nftables issue noted for this machine).
- Trait boundaries keep STT and LLM swappable for cloud providers later.
