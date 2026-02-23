use anyhow::Result;
use opus::{Channels, Decoder};
use shared_protocol::{
    ServerMessage, CS_SAMPLES, FRAME_SIZE_SAMPLES, SAMPLE_RATE,
};
use std::time::Instant;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

const MIN_FRAMES: u32 = 3; // do not transcribe if shorter than 3*60 = 180 ms
const MIN_SAMPLES: usize = (MIN_FRAMES * FRAME_SIZE_SAMPLES) as usize;

#[derive(Clone, Debug)]
pub struct TranscribeOpts {
    pub dynamic_audio_ctx: bool,
    pub temperature_inc: Option<f32>,
    pub entropy_thold: Option<f32>,
}

pub struct Session {
    language: Option<String>, // None = auto-detect
    context: Option<String>,
    opus_decoder: Decoder,
    accumulated_audio: Vec<i16>,
    whisper_state: WhisperState, // reuse state for performance
    advance_cs: i64, // total centiseconds dropped from the beginning
    transcribed_up_to_cs: i64, // end timestamp of the last transcription
    advanced_since: bool,
    sampling_strategy: SamplingStrategy,
    opts: TranscribeOpts,
}

impl Session {
    pub fn new(
        ctx: &WhisperContext,
        language: Option<String>,
        context: Option<String>,
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
            language: language_opt,
            context,
            opus_decoder,
            accumulated_audio: Vec::new(),
            whisper_state,
            advance_cs: 0,
            transcribed_up_to_cs: 0,
            advanced_since: false,
            sampling_strategy,
            opts,
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
        Ok(())
    }

    pub fn advance(&mut self, timestamp: i64) -> Result<()> {
        if timestamp <= self.advance_cs {
            return Ok(()); // already dropped past this point
        }

        let drop_cs = timestamp - self.advance_cs;
        let drop_samples = (drop_cs as usize) * (CS_SAMPLES as usize);

        if drop_samples > self.accumulated_audio.len() {
            anyhow::bail!("cannot advance to {:.2}s", timestamp as f64 / 100.0);
        }

        if drop_samples > 0 {
            self.accumulated_audio.drain(0..drop_samples);
            self.advance_cs = timestamp;
            self.advanced_since = true; // force retranscription
        }

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
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_token_timestamps(true); // token-level timing

        if let Some(v) = self.opts.temperature_inc {
            params.set_temperature_inc(v);
        }
        if let Some(v) = self.opts.entropy_thold {
            params.set_entropy_thold(v);
        }
        if self.opts.dynamic_audio_ctx {
            // scale audio_ctx to buffer length, multiple of 64, min 384
            let needed = (audio_f32.len() as i32 * 1500)
                / (SAMPLE_RATE as i32 * 30);
            let aligned = ((needed + 63) / 64) * 64;
            params.set_audio_ctx(aligned.max(384));
        }

        if let Some(ref prompt) = self.context {
            params.set_initial_prompt(prompt);
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

        let mut complete = Vec::new();
        let mut incomplete = None;

        for i in 0..n_segments {
            let Some(segment) = self.whisper_state.get_segment(i) else {
                continue;
            };

            // add advance_cs for absolute "connection" time
            let start_time = segment.start_timestamp() + self.advance_cs;
            let end_time = (segment.end_timestamp() + self.advance_cs)
                .min(current_end_cs);

            // extract token-level timing for precise merging
            let mut tokens = Vec::new();
            let n_tokens = segment.n_tokens();
            let buffer_len_cs = (self.accumulated_audio.len() as i64 * 100) / SAMPLE_RATE as i64;
            for j in 0..n_tokens {
                if let Some(token) = segment.get_token(j) {
                    let token_text = token.to_str_lossy()?.to_string();
                    // skip special tokens ([_TT_NNN], [_BEG_], [_SOT_], etc.)
                    if token_text.starts_with("[_") {
                        continue;
                    }
                    let token_data = token.token_data();
                    // do not trust tokens beyond the actual audio
                    if token_data.t0 >= buffer_len_cs {
                        continue;
                    }
                    // token timestamps are relative to buffer start
                    tokens.push(shared_protocol::Token {
                        text: token_text,
                        start_cs: token_data.t0 + self.advance_cs,
                        end_cs: token_data.t1 + self.advance_cs,
                    });
                }
            }

            if tokens.is_empty() {
                continue; // skip segments with no meaningful tokens
            }

            let segment_text = tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<String>()
                .trim()
                .to_string();

            let fallback_segmentation = (end_time - start_time) % 100 == 0;
            let segment = shared_protocol::Segment {
                text: segment_text,
                tokens,
                start_cs: start_time,
                end_cs: end_time,
                fallback_segmentation,
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
}
