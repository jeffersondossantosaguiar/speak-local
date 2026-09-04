# Intent: English Practice Voice App (Phase 1 — Local, Web Only)

- **Outcome:** A local, single-user web app: record spoken English, transcribe it (Whisper), analyze it (local Llama via Ollama) for prioritized grammar/vocab errors + a rough CEFR estimate.
- **User:** A backend engineer (Node/NestJS day job) building it as a personal study tool and Rust portfolio project.
- **Why now:** To practice spoken English with automated, local-only feedback without paying for cloud APIs.
- **Success:** Record → transcript → ordered error list (most critical first) + CEFR label with justification, all running on the local machine, with cloud-ready provider abstractions.
- **Constraint:** Everything local this phase. Single user, no auth, no persistence. GTX 1650 (4GB VRAM), 15GB RAM, Ryzen 1600.
- **Out of scope:** Mobile app, any cloud API, user accounts/history, SQLx/Postgres scaffolding.

## Grilled decisions (confirmed)

1. **Latency is not a constraint** — study tool, so 20–60s CPU analysis is acceptable.
2. **GPU allocation (Option A):** Whisper on GPU; LLM on CPU via Ollama. 8B-class model on CPU is the starting point (drop to 3B via env var if it grates).
3. **Audio path (Option C):** Browser records WebM/Opus (MediaRecorder); backend decodes to PCM in Rust with Symphonia + `opus` crate *inside* `TranscriptionProvider`.
4. **Request lifecycle (Option B):** Async job pattern — `POST /jobs` → poll `GET /jobs/{id}` → retrieve result.
5. **Structured output (Option B, rich):** each error has `context`, `suggestion`, `type` (grammar/vocab/etc), `criticality` (ordered most-critical first); CEFR = label + short justification (never a precise score).
6. **No persistence:** defer SQLx/Postgres entirely; document conventions only.
7. **Confirmed assumptions:** Whisper model = `small` (configurable via env var); package manager = **pnpm**; backend runs **natively, not containerized** (no Docker in phase 1).

## Key conventions
- Cargo workspace (Rust + Axum) + pnpm workspace (React), one repo.
- Trait interfaces: `TranscriptionProvider`, `AnalysisProvider` → cloud implementations swappable later.
- STT/LLM calls off async runtime via `tokio::task::spawn_blocking`.
- Env-var-configurable model size/path.
- When SQLx arrives (phase 2): compile-time query checking, single `db.rs`, migration naming `<timestamp>_<verbo>_<o_que>.sql`.
