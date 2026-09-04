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
   ollama pull llama3.2        # or qwen2.5:3b if you want something lighter
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

## Configuration (env vars)

See `backend/.env.example`:

| Var | Default | Purpose |
| --- | --- | --- |
| `BIND_ADDR` | `127.0.0.1:8787` | Backend listen address |
| `WHISPER_MODEL` | `models/ggml-small.bin` | Whisper model path (relative paths resolve against `backend/`; the repo's `.env` uses `ggml-small.en.bin`) |
| `WHISPER_USE_GPU` | `0` | Run Whisper on the GPU (requires a `--features cuda` build; the repo's `.env` sets `1`) |
| `OLLAMA_URL` | `http://localhost:11434` | LLM API base URL |
| `LLM_MODEL` | `llama3` | Model served by Ollama (`backend/.env` sets `llama3.2`, which you should `ollama pull` first) |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature for extraction |

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

## Notes / conventions

- **No persistence** in phase 1 (SQLx/Postgres conventions are documented in
  `docs/intent/english-practice-app.md` for phase 2).
- Backend runs natively on the host, **not containerized** (avoids the
  `network_mode: host` nftables issue noted for this machine).
- Trait boundaries keep STT and LLM swappable for cloud providers later.
