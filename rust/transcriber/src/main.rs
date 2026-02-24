mod session;

use anyhow::Result;
use clap::Parser;
use futures_util::{FutureExt, SinkExt, StreamExt};
use session::{Session, TranscribeOpts};
use shared_protocol::{ClientMessage, ServerMessage};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info};
use whisper_rs::{SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Parser, Debug)]
#[command(name = "transcriber")]
struct Args {
    #[arg(short, long, default_value = "[::]", help = "address to listen on")]
    address: String,

    #[arg(short, long, default_value = "8001", help = "port to listen on")]
    port: u16,

    #[arg(short, long, help = "path to whisper model file")]
    model: String, // path to whisper model file

    #[arg(long, help = "path to optional API token")]
    token_file: Option<String>,

    #[arg(
        long,
        help = "Best-of (default: 1, mutually exclusive with --beam-size)",
        conflicts_with = "beam_size"
    )]
    best_of: Option<i32>,
    #[arg(
        long,
        help = "Beam search size (mutually exclusive with --best-of)",
        conflicts_with = "best_of"
    )]
    beam_size: Option<i32>,

    #[arg(
        long,
        help = "Scale audio_ctx to buffer length (faster for short chunks)"
    )]
    dynamic_audio_ctx: bool,

    #[arg(
        long,
        help = "Temp increment on decode retry (0 = no retry, default: 0.2)"
    )]
    temperature_inc: Option<f32>,

    #[arg(long, help = "Entropy threshold for decode retry (default: 2.4)")]
    entropy_thold: Option<f32>,

    #[arg(
        long,
        help = "Reinitialize whisper state before every transcription"
    )]
    reinit_state: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let addr: SocketAddr = format!("{}:{}", args.address, args.port).parse()?;
    info!("Loading whisper model: {}", args.model);

    let ctx = {
        let mut params = WhisperContextParameters::default();
        params.flash_attn(true);
        #[cfg(not(feature = "vulkan"))]
        info!("Running on CPU");
        #[cfg(feature = "vulkan")]
        {
            info!("Running with GPU acceleration (Vulkan)");
            params.use_gpu(true);
        }
        Arc::new(WhisperContext::new_with_params(&args.model, params)?)
    };

    let expected_token = match &args.token_file {
        Some(path) => {
            info!("API token authentication enabled");
            Some(
                std::fs::read_to_string(path)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|e| {
                        panic!("Failed to read {}: {}", path, e)
                    }),
            )
        }
        None => None,
    };

    // Configure sampling strategy
    let sampling_strategy = match (args.beam_size, args.best_of) {
        (Some(beam_size), None) => {
            info!("Using beam search with beam_size={}", beam_size);
            SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            }
        }
        (None, Some(best_of)) => {
            info!("Using greedy search with best_of={}", best_of);
            SamplingStrategy::Greedy { best_of }
        }
        (None, None) => {
            info!("Using greedy search with best_of=1 (default)");
            SamplingStrategy::Greedy { best_of: 1 }
        }
        (Some(_), Some(_)) => {
            unreachable!("beam_size and best_of are mutually exclusive")
        }
    };

    let transcribe_opts = TranscribeOpts {
        dynamic_audio_ctx: args.dynamic_audio_ctx,
        temperature_inc: args.temperature_inc,
        entropy_thold: args.entropy_thold,
        reinit_state: args.reinit_state,
    };

    info!("Listening on {}", addr);
    let listener = TcpListener::bind(addr).await?;
    while let Ok((stream, peer_addr)) = listener.accept().await {
        info!("Connection from {}", peer_addr);
        let ctx = ctx.clone();
        let exp_token = expected_token.clone();
        let strategy = sampling_strategy.clone();
        let opts = transcribe_opts.clone();
        tokio::spawn(async move {
            if let Err(e) =
                handle_connection(stream, ctx, exp_token, strategy, opts).await
            {
                error!("Connection error: {}", e);
            }
        });
    }
    Ok(())
}

macro_rules! bail {
    ($ws_sender:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        let m = ServerMessage::Error { message: msg.clone() };
        let m = serde_json::to_string(&m).unwrap();
        let _ = $ws_sender.send(Message::Text(m)).await;
        let _ = $ws_sender.send(Message::Close(None)).await;
        return Err(anyhow::anyhow!(msg));
    }};
}

fn normalize_for_comparison(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
}

fn compare_segments(
    original: &shared_protocol::Segment,
    retranscribed: &[shared_protocol::Segment],
) -> (bool, usize) {
    let first = match retranscribed.first() {
        Some(s) => s,
        None => return (false, 0),
    };
    let orig_norm = normalize_for_comparison(original.text.trim());
    let first_norm = normalize_for_comparison(first.text.trim());
    let exact_match = !orig_norm.is_empty() && orig_norm == first_norm;

    // Count matching leading tokens (case/punctuation insensitive)
    let orig_tokens: Vec<String> = original
        .tokens
        .iter()
        .map(|t| normalize_for_comparison(&t.text))
        .collect();
    let new_tokens: Vec<String> = retranscribed
        .iter()
        .flat_map(|s| s.tokens.iter())
        .map(|t| normalize_for_comparison(&t.text))
        .collect();
    let n_matching_tokens = orig_tokens
        .iter()
        .zip(new_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();

    (exact_match, n_matching_tokens)
}

async fn handle_connection(
    stream: TcpStream,
    ctx: Arc<WhisperContext>,
    expected_token: Option<String>,
    sampling_strategy: SamplingStrategy,
    opts: TranscribeOpts,
) -> Result<()> {
    let ws_stream = tokio_tungstenite::accept_async(stream).await?;
    let (mut ws_sender, ws_receiver) = ws_stream.split();
    let ws_receiver = ws_receiver.peekable();
    futures_util::pin_mut!(ws_receiver);

    // First wait for the mandatory Configure message:
    let (
        token,
        language,
        context,
        max_len,
        max_tokens,
        single_segment,
        max_initial_ts,
        no_preview,
        two_stroke,
    ) = match ws_receiver.as_mut().next().await {
        Some(Ok(Message::Text(text))) => {
            match serde_json::from_str::<ClientMessage>(&text) {
                Ok(ClientMessage::Configure {
                    token,
                    language,
                    context,
                    max_len,
                    max_tokens,
                    single_segment,
                    max_initial_ts,
                    no_preview,
                    two_stroke,
                }) => (
                    token,
                    language,
                    context,
                    max_len,
                    max_tokens,
                    single_segment,
                    max_initial_ts,
                    no_preview,
                    two_stroke,
                ),
                Ok(_) => bail!(ws_sender, "first message must be Configure"),
                Err(e) => bail!(ws_sender, "failed to parse Configure : {}", e),
            }
        }
        Some(Ok(_)) => bail!(ws_sender, "must send Configure first"),
        Some(Err(e)) => bail!(ws_sender, "pre-configure error {}", e),
        None => bail!(ws_sender, "connection closed before Configure"),
    };

    // Then check the token, if needed:
    if let Some(ref expected) = expected_token {
        match token {
            Some(ref t) if t == expected => (),
            Some(_) => bail!(ws_sender, "wrong API token"),
            None => bail!(ws_sender, "missing API token"),
        }
    }
    // Then configure the transcription session:
    info!("Configured: language={:?}, context={:?}", language, context);
    let mut session = match Session::new(
        ctx,
        language,
        context,
        max_len,
        max_tokens,
        single_segment,
        max_initial_ts,
        sampling_strategy,
        opts,
    ) {
        Ok(s) => s,
        Err(e) => bail!(ws_sender, "error creating session: {}", e),
    };

    let two_stroke = two_stroke.unwrap_or(false);

    // Drain all pending WebSocket messages (audio, advance, EOS)
    macro_rules! drain {
        ($ws_receiver:expr, $ws_sender:expr, $session:expr, $finalized:expr) => {
            loop {
                match $ws_receiver.as_mut().next().now_or_never() {
                    Some(Some(Ok(msg))) => match msg {
                        Message::Text(text) => {
                            match serde_json::from_str::<ClientMessage>(&text) {
                                Ok(ClientMessage::Configure { .. }) => bail!(
                                    $ws_sender,
                                    "Configure sent after session started"
                                ),
                                Ok(ClientMessage::Advance {
                                    timestamp_cs,
                                    context,
                                }) => {
                                    if let Err(e) =
                                        $session.advance(timestamp_cs, context)
                                    {
                                        bail!($ws_sender, "advance failed: {}", e);
                                    };
                                    let time_s = timestamp_cs as f64 / 100.;
                                    info!("advanced to {:.2}s", time_s);
                                }
                                Ok(ClientMessage::EndOfStream) => {
                                    info!("end of audio stream");
                                    $finalized = true;
                                }
                                Err(e) => {
                                    bail!($ws_sender, "cannot parse message: {}", e)
                                }
                            }
                        }
                        Message::Binary(data) => {
                            if let Err(e) = $session.decode_and_append_opus(&data) {
                                bail!($ws_sender, "error decoding Opus: {}", e);
                            }
                        }
                        Message::Ping(data) => {
                            $ws_sender.send(Message::Pong(data)).await?;
                        }
                        Message::Pong(_) | Message::Frame(_) => {}
                        Message::Close(_) => bail!($ws_sender, "connection closed"),
                    },
                    Some(Some(Err(e))) => {
                        bail!($ws_sender, "websocket error: {}", e)
                    }
                    Some(None) => bail!($ws_sender, "connection closed"),
                    None => break,
                }
            }
        };
    }

    // Finally, enter the normal drain-transcribe loop:
    let mut finalized = false;
    loop {
        drain!(ws_receiver, ws_sender, session, finalized);

        // transcribe
        if no_preview.unwrap_or(false) && !finalized {
            ws_receiver.as_mut().peek().await;
            continue;
        }
        match session.transcribe(finalized) {
            Ok(Some(msg)) => {
                let json = serde_json::to_string(&msg)?;
                ws_sender.send(Message::Text(json)).await?;

                // Two-stroke: re-transcribe from second-to-last segment end
                if two_stroke && !finalized {
                    if let ServerMessage::Transcription {
                        ref complete,
                        advance_cs: tx_advance_cs,
                        ..
                    } = msg
                    {
                        if complete.len() >= 2 {
                            let last_index = complete.len() - 1;
                            let second_to_last = &complete[last_index - 1];
                            let last = &complete[last_index];
                            let retranscribe_from_cs = second_to_last.end_cs;

                            // Drain again to pick up any audio that arrived during transcription
                            drain!(ws_receiver, ws_sender, session, finalized);

                            match session.transcribe_from(retranscribe_from_cs, finalized) {
                                Ok(retranscribed_segments) => {
                                    let (exact_match, n_matching_tokens) =
                                        compare_segments(last, &retranscribed_segments);
                                    let suggestion = ServerMessage::AdvanceSuggestion {
                                        advance_cs: tx_advance_cs,
                                        timestamp_cs: last.end_cs,
                                        segments: retranscribed_segments,
                                        original_last_segment: last.clone(),
                                        exact_match,
                                        n_matching_tokens,
                                    };
                                    let json = serde_json::to_string(&suggestion)?;
                                    ws_sender.send(Message::Text(json)).await?;
                                }
                                Err(e) => {
                                    error!("two-stroke retranscription error: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            Ok(None) => {} // not enough audio
            Err(e) => bail!(ws_sender, "Transcription error: {}", e),
        }

        if finalized {
            break;
        }

        ws_receiver.as_mut().peek().await; // block without consuming
    }

    ws_sender.send(Message::Close(None)).await?;
    info!("Session ended");
    Ok(())
}
