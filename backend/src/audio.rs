use crate::providers::AudioSamples;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decode a WebM/Opus (or other supported container) byte blob into mono
/// 16 kHz f32 PCM samples suitable for Whisper.
///
/// The input is probed by content rather than by extension, since the
/// browser blob arrives without a reliable extension in the multipart body.
pub fn decode_audio(data: &[u8]) -> Result<AudioSamples, String> {
    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    // WebM / MKV and Ogg are the two containers browsers produce for Opus.
    hint.with_extension("mkv");
    hint.mime_type("audio/webm");

    let format_opts: FormatOptions = Default::default();
    let meta_opts: MetadataOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &meta_opts)
        .map_err(|e| format!("probe failed: {e}"))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no audio track".to_string())?;

    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &Default::default())
        .map_err(|e| format!("unsupported codec: {e}"))?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut spec: Option<symphonia::core::audio::SignalSpec> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::ResetRequired) => break,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let s = *decoded.spec();
                if spec.is_none() {
                    spec = Some(s);
                }
                let mut buf = SampleBuffer::<f32>::new(decoded.frames() as u64, s);
                buf.copy_interleaved_ref(decoded);
                all_samples.extend_from_slice(buf.samples());
            }
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if all_samples.is_empty() {
        return Err("no audio samples decoded".to_string());
    }

    let s = spec.expect("audio spec captured during decode");
    let channels = s.channels.count();
    let sample_rate = s.rate as usize;

    Ok(resample_and_mix_to_mono_16k(all_samples, channels, sample_rate))
}

/// Mix multichannel interleaved samples to mono and resample to 16 kHz.
/// Linear-resamples internally; good enough for transcription input.
fn resample_and_mix_to_mono_16k(
    interleaved: Vec<f32>,
    channels: usize,
    sample_rate: usize,
) -> AudioSamples {
    let target_rate = 16000;
    let frames = interleaved.len() / channels.max(1);
    let mut mono: Vec<f32> = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * channels;
        let mut sum: f32 = 0.0;
        for c in 0..channels {
            sum += interleaved[start + c];
        }
        mono.push(sum / channels as f32);
    }

    if sample_rate == target_rate {
        return AudioSamples { samples: mono };
    }

    let ratio = sample_rate as f64 / target_rate as f64;
    let out_len = ((mono.len() as f64) / ratio) as usize;
    let mut resampled = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(mono.len() - 1);
        let frac = src_pos - i0 as f64;
        let val = mono[i0] as f64 * (1.0 - frac) + mono[i1] as f64 * frac;
        resampled.push(val as f32);
    }

    AudioSamples { samples: resampled }
}
