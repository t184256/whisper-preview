use serde::{Deserialize, Serialize};

pub const FRAME_SIZE_CS: u32 = 6; // 2*30ms = 60ms (common Opus frame size)
pub const SAMPLE_RATE: u32 = 16000; // Whisper requires 16kHz
pub const CS_SAMPLES: u32 = SAMPLE_RATE / 100; // 160 = 1 cs at 16kHz
pub const FRAME_SIZE_SAMPLES: u32 = FRAME_SIZE_CS * CS_SAMPLES; // 960

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    Configure {
        // sent once after connection,required
        token: Option<String>,    // optional auth token
        language: Option<String>, // defaults to "auto"
        context: Option<String>,  // extra context for transcription
    },
    // no explicit AudioChunk message - binary frames are implicitly audio
    Advance {
        timestamp_cs: i64, // forget audio before this, centiseconds from 0
        context: Option<Segment>, // last confirmed segment, for token IDs
    },
    EndOfStream, // trigger final transcription
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub id: i32, // whisper token ID, needed for prompt context
    pub start_cs: i64,
    pub end_cs: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    pub start_cs: i64,
    pub end_cs: i64,
    pub tokens: Vec<Token>,
    pub fallback_segmentation: bool,
    pub end_vad_probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    Transcription {
        complete: Vec<Segment>, // segments considered complete
        incomplete: Option<Segment>, // still-growing preview
        fast_preview: Option<Segment>, // preview from lower quality model
        advance_cs: i64, // beginning timestamp of the transcription result
    },
    Error {
        message: String,
    },
}
