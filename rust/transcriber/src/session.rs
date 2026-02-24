use anyhow::Result;
use opus::{Channels, Decoder};
use shared_protocol::{
    CS_SAMPLES, FRAME_SIZE_SAMPLES, SAMPLE_RATE, ServerMessage,
};
use shared_vad::Vad;
use std::ffi::c_int;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

const MAX_PROMPT_TOKENS: usize = 224; // half of whisper's 448-token context
const MIN_FRAMES: u32 = 3; // do not transcribe if shorter than 3*60 = 180 ms
const MIN_SAMPLES: usize = (MIN_FRAMES * FRAME_SIZE_SAMPLES) as usize;

#[derive(Clone, Debug)]
pub struct TranscribeOpts {
    pub dynamic_audio_ctx: bool,
    pub temperature_inc: Option<f32>,
    pub entropy_thold: Option<f32>,
    pub reinit_state: bool,
}

pub struct Session {
    ctx: Arc<WhisperContext>,
    language: Option<String>, // None = auto-detect
    context: Option<String>,
    opus_decoder: Decoder,
    accumulated_audio: Vec<i16>,
    whisper_state: WhisperState, // reuse state for performance
    vad: Vad,
    prompt_tokens: Vec<c_int>, // token IDs from last transcription, for context
    advance_cs: i64,           // total centiseconds advanced from the beginning
    transcribed_up_to_cs: i64, // end timestamp of the last transcription
    advanced_since: bool,
    sampling_strategy: SamplingStrategy,
    opts: TranscribeOpts,
    max_len: i32,
    max_tokens: i32,
    single_segment: bool,
    max_initial_ts: f32,
}

impl Session {
    pub fn new(
        ctx: Arc<WhisperContext>,
        language: Option<String>,
        context: Option<String>,
        max_len: Option<i32>,
        max_tokens: Option<i32>,
        single_segment: Option<bool>,
        max_initial_ts: Option<f32>,
        sampling_strategy: SamplingStrategy,
        opts: TranscribeOpts,
    ) -> Result<Self> {
        let opus_decoder = Decoder::new(SAMPLE_RATE, Channels::Mono)?;
        let whisper_state = ctx.create_state()?;

        let language_opt = language.filter(|l| !l.is_empty() && l != "auto");
        match &language_opt {
            Some(lang) => info!("Session created with language {}", lang),
            None => info!("Session created with language auto-detection"),
        }

        Ok(Self {
            ctx,
            language: language_opt,
            context,
            opus_decoder,
            accumulated_audio: Vec::new(),
            whisper_state,
            vad: Vad::new(),
            prompt_tokens: Vec::new(),
            advance_cs: 0,
            transcribed_up_to_cs: 0,
            advanced_since: false,
            sampling_strategy,
            opts,
            max_len: max_len.unwrap_or(0),
            max_tokens: max_tokens.unwrap_or(0),
            single_segment: single_segment.unwrap_or(false),
            max_initial_ts: max_initial_ts.unwrap_or(0.),
        })
    }

    pub fn decode_and_append_opus(&mut self, packet: &[u8]) -> Result<()> {
        let mut output = vec![0i16; FRAME_SIZE_SAMPLES as usize];
        let samples_decoded =
            self.opus_decoder.decode(packet, &mut output, false)?;
        if samples_decoded != (FRAME_SIZE_SAMPLES as usize) {
            anyhow::bail!("decompressed to unexpected len {}", samples_decoded);
        }
        self.accumulated_audio.extend(&output); // see advance for draining
        self.vad.consume(&output);
        Ok(())
    }

    pub fn advance(
        &mut self,
        timestamp: i64,
        context: Option<shared_protocol::Segment>,
    ) -> Result<()> {
        if timestamp <= self.advance_cs {
            return Ok(()); // already advanced past this point
        }

        let drop_cs = timestamp - self.advance_cs;
        let drop_samples = (drop_cs as usize) * (CS_SAMPLES as usize);

        if drop_samples > self.accumulated_audio.len() {
            anyhow::bail!("cannot advance to {:.2}s", timestamp as f64 / 100.0);
        }

        // use client-provided context segment for prompt tokens (keep tail)
        self.prompt_tokens.clear();
        if let Some(segment) = context {
            let from = segment.tokens.len().saturating_sub(MAX_PROMPT_TOKENS);
            let token_ids = segment.tokens[from..].iter().map(|t| t.id);
            self.prompt_tokens.extend(token_ids);
        }

        self.accumulated_audio.drain(0..drop_samples);
        self.advance_cs = timestamp;
        self.advanced_since = true; // force retranscription
        self.vad.reset(); // and recalculate VAD from remaining audio
        self.vad.consume(&self.accumulated_audio);

        Ok(())
    }

    pub fn transcribe(
        &mut self,
        is_final: bool,
    ) -> Result<Option<ServerMessage>> {
        if self.accumulated_audio.len() < MIN_SAMPLES {
            return Ok(None);
        }

        let current_end_cs = self.advance_cs
            + (self.accumulated_audio.len() as i64 * 100) / SAMPLE_RATE as i64;

        if !is_final
            && !self.advanced_since
            && current_end_cs == self.transcribed_up_to_cs
        {
            return Ok(None); // do not re-transcribe if there's nothing new
        }

        let buffer_growth_cs = current_end_cs - self.transcribed_up_to_cs;
        if buffer_growth_cs > 0 {
            info!(
                "buffer grew {:.2}s since the last transcription",
                buffer_growth_cs as f64 / 100.
            );
        }

        let audio_f32: Vec<f32> = self
            .accumulated_audio
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        let mut params = FullParams::new(self.sampling_strategy.clone());
        params.set_language(self.language.as_deref()); // None = auto-detect
        params.set_suppress_nst(true);
        params.set_max_len(self.max_len);
        params.set_max_tokens(self.max_tokens);
        params.set_max_initial_ts(self.max_initial_ts);
        params.set_single_segment(self.single_segment);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_token_timestamps(true); // token-level timing
        params.set_tokens(&self.prompt_tokens);
        params.set_no_context(true);

        if let Some(v) = self.opts.temperature_inc {
            params.set_temperature_inc(v);
        }
        if let Some(v) = self.opts.entropy_thold {
            params.set_entropy_thold(v);
        }
        if self.opts.dynamic_audio_ctx {
            // scale audio_ctx to buffer length, multiple of 64, min 384
            let needed =
                (audio_f32.len() as i32 * 1500) / (SAMPLE_RATE as i32 * 30);
            let aligned = ((needed + 63) / 64) * 64;
            params.set_audio_ctx(aligned.max(384));
        }

        if let Some(ref prompt) = self.context {
            params.set_initial_prompt(prompt);
        }

        if self.opts.reinit_state {
            self.whisper_state = self.ctx.create_state()?;
        }

        let start = Instant::now();
        self.whisper_state.full(params, &audio_f32)?;
        let duration = start.elapsed().as_secs_f64();

        self.transcribed_up_to_cs = current_end_cs;
        self.advanced_since = false;

        let audio_duration =
            self.accumulated_audio.len() as f64 / SAMPLE_RATE as f64;
        let realtime_factor = audio_duration / duration;
        info!(
            "transcribing range={:.2}s-{:.2}s took {:.2}s at {:.2}x",
            self.advance_cs as f64 / 100.,
            current_end_cs as f64 / 100.,
            duration,
            realtime_factor
        );

        let n_segments = self.whisper_state.full_n_segments();
        let eot_id = self.ctx.token_eot();

        let mut complete = Vec::new();
        let mut incomplete = None;

        for i in 0..n_segments {
            let Some(segment) = self.whisper_state.get_segment(i) else {
                continue;
            };

            // add advance_cs for absolute "connection" time
            let start_time = segment.start_timestamp() + self.advance_cs;

            // extract token-level timing for precise merging
            let mut tokens = Vec::new();
            let n_tokens = segment.n_tokens();
            let buffer_len_cs = (self.accumulated_audio.len() as i64 * 100)
                / SAMPLE_RATE as i64;
            for j in 0..n_tokens {
                if let Some(token) = segment.get_token(j) {
                    let token_text = token.to_str_lossy()?.to_string();
                    let token_data = token.token_data();
                    // do not trust tokens beyond the actual audio
                    if token_data.t0 >= buffer_len_cs {
                        continue;
                    }
                    // token timestamps are relative to buffer start
                    tokens.push(shared_protocol::Token {
                        text: token_text,
                        id: token.token_id(),
                        special: token.token_id() >= eot_id,
                        start_cs: token_data.t0 + self.advance_cs,
                        end_cs: token_data.t1 + self.advance_cs,
                        probability: token.token_probability(),
                    });
                }
            }

            if tokens.is_empty() {
                continue; // skip segments with no meaningful tokens
            }

            let segment_text = tokens
                .iter()
                .filter(|t| !t.special)
                .map(|t| t.text.as_str())
                .collect::<String>()
                .trim()
                .to_string();
            let end_time =
                (segment.end_timestamp() + self.advance_cs).min(current_end_cs);

            let fallback_segmentation = (end_time - start_time) % 100 == 0;
            let end_vad_probability =
                self.vad.probability_at_cs(end_time - self.advance_cs);
            let no_speech_probability = segment.no_speech_probability();

            let segment = shared_protocol::Segment {
                text: segment_text,
                start_cs: start_time,
                end_cs: end_time,
                tokens,
                fallback_segmentation,
                end_vad_probability,
                no_speech_probability,
            };

            if i < n_segments - 1 {
                complete.push(segment); // not last - always complete
            } else {
                if is_final {
                    complete.push(segment); // last - complete if finalizing
                } else {
                    incomplete = Some(segment); // incomplete otherwise
                }
            }
        }

        // return all segments (client filters based on advance_cs)
        Ok(Some(ServerMessage::Transcription {
            complete,
            incomplete,
            fast_preview: None, // regular transcriber doesn't use fast_preview
            advance_cs: self.advance_cs,
        }))
    }

    /// Re-transcribe audio starting from `from_cs` (absolute) to the current
    /// buffer end. Reuses the existing whisper state (each full() call with
    /// no_context=true is independent). Returns all complete segments.
    pub fn transcribe_from(
        &mut self,
        from_cs: i64,
        is_final: bool,
    ) -> Result<Vec<shared_protocol::Segment>> {
        let offset_samples =
            ((from_cs - self.advance_cs) as usize) * (CS_SAMPLES as usize);
        if offset_samples >= self.accumulated_audio.len() {
            return Ok(Vec::new());
        }
        let audio_slice = &self.accumulated_audio[offset_samples..];
        if audio_slice.len() < MIN_SAMPLES {
            return Ok(Vec::new());
        }

        let audio_f32: Vec<f32> =
            audio_slice.iter().map(|&s| s as f32 / 32768.0).collect();

        let mut params = FullParams::new(self.sampling_strategy.clone());
        params.set_language(self.language.as_deref());
        params.set_suppress_nst(true);
        params.set_max_len(self.max_len);
        params.set_max_tokens(self.max_tokens);
        params.set_max_initial_ts(0.0);
        params.set_single_segment(false);
        params.set_print_progress(false);
        params.set_print_special(true);
        params.set_print_realtime(false);
        params.set_token_timestamps(true);
        params.set_no_context(true);

        if let Some(v) = self.opts.temperature_inc {
            params.set_temperature_inc(v);
        }
        if let Some(v) = self.opts.entropy_thold {
            params.set_entropy_thold(v);
        }
        if self.opts.dynamic_audio_ctx {
            let needed =
                (audio_f32.len() as i32 * 1500) / (SAMPLE_RATE as i32 * 30);
            let aligned = ((needed + 63) / 64) * 64;
            params.set_audio_ctx(aligned.max(384));
        }

        if let Some(ref prompt) = self.context {
            params.set_initial_prompt(prompt);
        }

        let start = Instant::now();
        self.whisper_state.full(params, &audio_f32)?;
        let duration = start.elapsed().as_secs_f64();

        let audio_duration = audio_slice.len() as f64 / SAMPLE_RATE as f64;
        info!(
            "two-stroke retranscription {:.2}s-end took {:.2}s at {:.2}x",
            from_cs as f64 / 100.,
            duration,
            audio_duration / duration,
        );

        let n_segments = self.whisper_state.full_n_segments();
        let eot_id = self.ctx.token_eot();
        let buffer_len_cs =
            (audio_slice.len() as i64 * 100) / SAMPLE_RATE as i64;
        let mut segments = Vec::new();

        for i in 0..n_segments {
            let Some(segment) = self.whisper_state.get_segment(i) else {
                continue;
            };

            let start_time = segment.start_timestamp() + from_cs;
            let mut tokens = Vec::new();
            let n_tokens = segment.n_tokens();
            for j in 0..n_tokens {
                if let Some(token) = segment.get_token(j) {
                    let token_text = token.to_str_lossy()?.to_string();
                    let token_data = token.token_data();
                    if token_data.t0 >= buffer_len_cs {
                        continue;
                    }
                    tokens.push(shared_protocol::Token {
                        text: token_text,
                        id: token.token_id(),
                        special: token.token_id() >= eot_id,
                        start_cs: token_data.t0 + from_cs,
                        end_cs: token_data.t1 + from_cs,
                        probability: token.token_probability(),
                    });
                }
            }

            let segment_text = tokens
                .iter()
                .filter(|t| !t.special)
                .map(|t| t.text.as_str())
                .collect::<String>()
                .trim()
                .to_string();
            let end_time = (segment.end_timestamp() + from_cs)
                .min(from_cs + buffer_len_cs);

            let fallback_segmentation = (end_time - start_time) % 100 == 0;
            // VAD relative to from_cs offset within our slice
            let vad_cs = end_time - from_cs;
            let end_vad_probability = if vad_cs >= 0 && vad_cs < buffer_len_cs {
                // Rebuild VAD for the slice
                0.0 // no VAD for retranscription
            } else {
                0.0
            };
            let no_speech_probability = segment.no_speech_probability();

            let seg = shared_protocol::Segment {
                text: segment_text,
                start_cs: start_time,
                end_cs: end_time,
                tokens,
                fallback_segmentation,
                end_vad_probability,
                no_speech_probability,
            };

            if i < n_segments - 1 || is_final {
                segments.push(seg);
            } else {
                segments.push(seg); // include incomplete too for comparison
            }
        }

        Ok(segments)
    }
}
