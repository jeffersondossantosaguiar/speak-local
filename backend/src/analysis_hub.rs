use crate::providers::Analysis;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};
use tokio::sync::broadcast;
use uuid::Uuid;

/// An event on a job's analysis stream.
#[derive(Clone)]
pub enum HubEvent {
    /// A raw chunk of the LLM's JSON output as it is being generated.
    Delta(String),
    /// The full, validated analysis once the LLM finishes.
    Done(Analysis),
}

/// What an SSE subscriber receives up front: every chunk generated so far
/// (so a late joiner can replay the extractor) plus a settled result, if any.
pub struct HubSubscription {
    pub past_deltas: Vec<String>,
    pub rx: broadcast::Receiver<HubEvent>,
}

/// Collects the streaming LLM output for every job and broadcasts it to SSE
/// subscribers. `push_*` are called from the (blocking) analysis thread, so
/// they are deliberately synchronous.
pub struct AnalysisHub {
    entries: Mutex<HashMap<Uuid, Entry>>,
}

struct Entry {
    /// Raw JSON chunks, in order, so late subscribers can replay them.
    deltas: Vec<String>,
    created_at: SystemTime,
    tx: broadcast::Sender<HubEvent>,
}

impl AnalysisHub {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Subscribe to a job's analysis stream. Creates a slot eagerly so deltas
    /// produced before the LLM starts are not lost.
    pub fn subscribe(&self, id: Uuid) -> HubSubscription {
        let mut guard = self.entries.lock().unwrap();
        let entry = guard.entry(id).or_insert_with(|| Entry {
            deltas: Vec::new(),
            created_at: SystemTime::now(),
            tx: broadcast::channel(128).0,
        });
        HubSubscription {
            past_deltas: entry.deltas.clone(),
            rx: entry.tx.subscribe(),
        }
    }

    /// Record a raw JSON chunk from the LLM stream.
    pub fn push_delta(&self, id: Uuid, delta: &str) {
        let tx = {
            let mut guard = self.entries.lock().unwrap();
            let entry = guard.entry(id).or_insert_with(|| Entry {
                deltas: Vec::new(),
                created_at: SystemTime::now(),
                tx: broadcast::channel(128).0,
            });
            entry.deltas.push(delta.to_string());
            entry.tx.clone()
        };
        let _ = tx.send(HubEvent::Delta(delta.to_string()));
    }

    /// Settle the stream with the final validated analysis.
    pub fn push_done(&self, id: Uuid, analysis: Analysis) {
        let tx = {
            let mut guard = self.entries.lock().unwrap();
            let entry = guard.entry(id).or_insert_with(|| Entry {
                deltas: Vec::new(),
                created_at: SystemTime::now(),
                tx: broadcast::channel(128).0,
            });
            // The last subscriber that cares replays the final analysis only,
            // so drop the raw chunks to free memory once settled.
            entry.deltas.clear();
            entry.tx.clone()
        };
        let _ = tx.send(HubEvent::Done(analysis));
    }

    /// Drop entries older than the retention window (called alongside the job
    /// store sweep).
    pub fn sweep(&self, now: SystemTime, retention: Duration) {
        let mut guard = self.entries.lock().unwrap();
        guard.retain(|_, e| {
            now.duration_since(e.created_at)
                .map(|d| d <= retention)
                .unwrap_or(false)
        });
    }
}

impl Default for AnalysisHub {
    fn default() -> Self {
        Self::new()
    }
}