#![warn(missing_docs)]
#![allow(rustdoc::bare_urls)]
#![doc = include_str!("../README.md")]
//! # Simple video player example
//! ```
#![doc = include_str!("../examples/main.rs")]
//! ```
extern crate ffmpeg_the_third as ffmpeg;
use anyhow::Result;
use atomic::Atomic;
use bytemuck::NoUninit;
use chrono::{DateTime, Duration, Utc};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::BuildStreamError;

use egui::{Color32, ColorImage};
use libc::EAGAIN;
use ffmpeg::ffi::{AVERROR, AV_TIME_BASE};
use ffmpeg::format::context::input::Input;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Audio;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::{rescale, Packet, Rational, Rescale};
use ffmpeg::{software, ChannelLayout};
use parking_lot::Mutex;
use cpal::traits::StreamTrait;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc};
use std::time::UNIX_EPOCH;
use subtitle::Subtitle;
use std::sync::mpsc;

mod subtitle;
mod player;

pub use player::gui_player::GuiPlayer;
pub use player::player::Player;

#[cfg(feature = "from_bytes")]
use tempfile::NamedTempFile;

#[cfg(feature = "from_bytes")]
use std::io::Write;

fn format_duration(dur: Duration) -> String {
    let dt = DateTime::<Utc>::from(UNIX_EPOCH) + dur;
    if dt.format("%H").to_string().parse::<i64>().unwrap() > 0 {
        dt.format("%H:%M:%S").to_string()
    } else {
        dt.format("%M:%S").to_string()
    }
}

/// The playback device. Needs to be initialized (and kept alive!) for use by a [`Player`].
pub struct AudioDevice {
    sample_format: cpal::SampleFormat,
    sample_rate: u32,
    callback: Arc<Mutex<AudioDeviceCallback>>,
    stream: cpal::Stream,
}

impl AudioDevice {
    /// Get the sample format.
    pub fn get_sample_format(&self) -> cpal::SampleFormat {
        self.sample_format
    }

    /// Get the sample rate.
    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn play(&self) {
        let _ = self.stream.play();
    }

    fn run<T>(callback: Arc<Mutex<AudioDeviceCallback>>, device: &cpal::Device, config: &cpal::StreamConfig) -> Result<cpal::Stream, BuildStreamError>
    where
        T: cpal::SizedSample + cpal::FromSample<f32> + std::iter::Sum<f32>,
    {
        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
        device.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                Self::write_data(callback.clone(), data)
            },
            err_fn,
            None,
        )
    }

    fn write_data<T>(callback: Arc<Mutex<AudioDeviceCallback>>, output: &mut [T])
    where
        T: cpal::Sample + cpal::FromSample<f32> + std::iter::Sum<f32>,
    {
        callback.lock().callback(output);
    }

    /// Create a new [`AudioDevice`]. An [`AudioDevice`] is required for using audio.
    pub fn new() -> Self {
        let host = cpal::default_host();
        let device = host.default_output_device().unwrap(); // TODO: do not unwrap, device might not be available; detect device changes, user might disconnect or change to a different output
        let callback = Arc::new(Mutex::new(AudioDeviceCallback::default()));
        let config = device.default_output_config().unwrap();
        let sample_format = config.sample_format();
        let sample_rate = config.sample_rate().0;
        let config_uw: &cpal::StreamConfig = &config.into();
        let stream = match sample_format {
            cpal::SampleFormat::F32 => Self::run::<f32>(callback.clone(), &device, config_uw),
            // TODO: add more formats (to_stream fn supports more), and don't panic, return an error
            sample_format => panic!("Unsupported sample format '{sample_format}'"),
        }.unwrap();
        AudioDevice {
            sample_format,
            sample_rate,
            callback: callback.clone(),
            stream,
        }
    }
}

enum PlayerMessage {
    StreamCycled(Type),
}

type PlayerMessageSender = std::sync::mpsc::Sender<PlayerMessage>;
type PlayerMessageReciever = std::sync::mpsc::Receiver<PlayerMessage>;

type ApplyVideoFrameFn = Box<dyn FnMut(Video) + Send>;
type SubtitleQueue = Arc<Mutex<VecDeque<Subtitle>>>;

type StreamingAudioChunks = Vec<f32>;
type AudioSampleProducer = mpsc::Sender<StreamingAudioChunks>;
type AudioSampleConsumer = mpsc::Receiver<StreamingAudioChunks>;

/// The possible states of a [`Player`].
#[derive(PartialEq, Clone, Copy, Debug, NoUninit)]
#[repr(u8)]
pub enum PlayerState {
    /// No playback.
    Stopped,
    /// Streams have reached the end of the file.
    EndOfFile,
    /// Stream is seeking.
    SeekingInProgress,
    /// Stream has finished seeking.
    SeekingFinished,
    /// Playback is paused.
    Paused,
    /// Playback is ongoing.
    Playing,
    /// Playback is scheduled to restart.
    Restarting,
}

/// Streams video.
pub struct VideoStreamer {
    video_decoder: ffmpeg::decoder::Video,
    video_stream_index: StreamIndex,
    player_state: Shared<PlayerState>,
    duration_ms: i64,
    input_context: Input,
    video_elapsed_ms: Shared<i64>,
    _audio_elapsed_ms: Shared<i64>,
    apply_video_frame_fn: Option<ApplyVideoFrameFn>,

    output_frame_width: u32,
    output_frame_height: u32,
}

/// Streams audio.
pub struct AudioStreamer {
    video_elapsed_ms: Shared<i64>,
    audio_elapsed_ms: Shared<i64>,
    duration_ms: i64,
    audio_decoder: ffmpeg::decoder::Audio,
    resampler: software::resampling::Context,
    audio_sample_producer: AudioSampleProducer,
    input_context: Input,
    player_state: Shared<PlayerState>,
    audio_stream_indices: VecDeque<StreamIndex>,
}

/// Streams subtitles.
pub struct SubtitleStreamer {
    video_elapsed_ms: Shared<i64>,
    _audio_elapsed_ms: Shared<i64>,
    subtitle_elapsed_ms: Shared<i64>,
    duration_ms: i64,
    subtitle_decoder: ffmpeg::decoder::Subtitle,
    next_packet: Option<Packet>,
    subtitles_queue: SubtitleQueue,
    input_context: Input,
    player_state: Shared<PlayerState>,
    subtitle_stream_indices: VecDeque<StreamIndex>,
}

#[derive(Clone, Debug)]
/// Simple concurrecy of primitive values.
pub struct Shared<T: Copy + bytemuck::NoUninit> {
    raw_value: Arc<Atomic<T>>,
}

impl<T: Copy + bytemuck::NoUninit> Shared<T> {
    /// Set the value.
    pub fn set(&self, value: T) {
        self.raw_value.store(value, atomic::Ordering::Relaxed)
    }
    /// Get the value.
    pub fn get(&self) -> T {
        self.raw_value.load(atomic::Ordering::Relaxed)
    }
    /// Make a new cache.
    pub fn new(value: T) -> Self {
        Self {
            raw_value: Arc::new(Atomic::new(value)),
        }
    }
}

const AV_TIME_BASE_RATIONAL: Rational = Rational(1, AV_TIME_BASE);
const MILLISEC_TIME_BASE: Rational = Rational(1, 1000);

fn timestamp_to_millisec(timestamp: i64, time_base: Rational) -> i64 {
    timestamp.rescale(time_base, MILLISEC_TIME_BASE)
}

fn millisec_to_timestamp(millisec: i64, time_base: Rational) -> i64 {
    millisec.rescale(MILLISEC_TIME_BASE, time_base)
}

#[inline(always)]
fn millisec_approx_eq(a: i64, b: i64) -> bool {
    a.abs_diff(b) < 50
}

fn get_stream_indices_of_type(
    input_context: &Input,
    stream_type: ffmpeg::media::Type,
) -> VecDeque<StreamIndex> {
    input_context
        .streams()
        .filter_map(|s| {
            (s.parameters().medium() == stream_type).then_some(StreamIndex::from(s.index()))
        })
        .collect::<VecDeque<_>>()
}

fn get_decoder_from_stream_index(
    input_context: &Input,
    stream_index: StreamIndex,
) -> Result<ffmpeg::decoder::Decoder> {
    let context = ffmpeg::codec::context::Context::from_parameters(
        input_context.stream(*stream_index).unwrap().parameters(),
    )?;
    Ok(context.decoder())
}

#[derive(PartialEq, Clone, Copy)]
/// The index of the stream.
pub struct StreamIndex(usize);

impl From<usize> for StreamIndex {
    fn from(value: usize) -> Self {
        StreamIndex(value)
    }
}

impl Deref for StreamIndex {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(PartialEq, Clone, Copy)]
struct StreamInfo {
    // Not the actual `StreamIndex` of the stream. This is a user-facing number that starts
    // at `1` and is incrememted when cycling between streams.
    current_stream: usize,
    total_streams: usize,
}

impl StreamInfo {
    fn new() -> Self {
        Self {
            current_stream: 1,
            total_streams: 0,
        }
    }
    fn from_total(total: usize) -> Self {
        let mut slf = Self::new();
        slf.total_streams = total;
        slf
    }
    fn cycle(&mut self) {
        self.current_stream = ((self.current_stream + 1) % (self.total_streams + 1)).max(1);
    }
    fn is_cyclable(&self) -> bool {
        self.total_streams > 1
    }
}

impl std::fmt::Display for StreamInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.current_stream, self.total_streams)
    }
}

/// Streams data.
pub trait Streamer: Send {
    /// The associated type of frame used for the stream.
    type Frame;
    /// The associated type after the frame is processed.
    type ProcessedFrame;
    /// Seek to a location within the stream.
    fn seek(&mut self, seek_frac: f32) {
        let target_ms = (seek_frac as f64 * self.duration_ms() as f64) as i64;
        let seek_completed = millisec_approx_eq(target_ms, self.elapsed_ms().get());
        // stop seeking near target so we dont waste cpu cycles
        if !seek_completed {
            let elapsed_ms = self.elapsed_ms().clone();
            let currently_behind_target = || elapsed_ms.get() < target_ms;

            let seeking_backwards = target_ms < self.elapsed_ms().get();
            let target_ts = millisec_to_timestamp(target_ms, rescale::TIME_BASE);

            // TODO: propagate error
            if self.input_context().seek(target_ts, ..=target_ts).is_ok() {
                self.decoder().flush();
                let mut previous_elapsed_ms = self.elapsed_ms().get();

                // this drop frame loop lets us refresh until current_ts is accurate
                if seeking_backwards {
                    while !currently_behind_target() {
                        let next_elapsed_ms = self.elapsed_ms().get();
                        if next_elapsed_ms > previous_elapsed_ms {
                            break;
                        }
                        previous_elapsed_ms = next_elapsed_ms;
                        if let Err(e) = self.drop_frames() {
                            if is_ffmpeg_eof_error(&e) {
                                break;
                            }
                        }
                    }
                }

                // // this drop frame loop drops frames until we are at desired
                while currently_behind_target() {
                    if let Err(e) = self.drop_frames() {
                        if is_ffmpeg_eof_error(&e) {
                            break;
                        }
                    }
                }

                // frame preview
                if self.is_primary_streamer() {
                    if let Ok(frame) = self.receive_next_packet_until_frame() {
                        self.apply_frame(frame)
                    }
                }
            }
        }
        if self.is_primary_streamer() {
            self.player_state().set(PlayerState::SeekingFinished);
        }
    }
    /// The type of data this stream corresponds to.
    fn stream_type(&self) -> Type;
    /// The primary streamer will control most of the state/syncing.
    fn is_primary_streamer(&self) -> bool;
    /// The stream index.
    fn stream_index(&self) -> StreamIndex;
    /// Move to the next stream index, if possible, and return the new_stream_index.
    fn cycle_stream(&mut self) -> StreamIndex;
    /// The elapsed time of this streamer, in milliseconds.
    fn elapsed_ms(&self) -> &Shared<i64>;
    /// The elapsed time of the primary streamer, in milliseconds.
    fn primary_elapsed_ms(&self) -> &Shared<i64>;
    /// The total duration of the stream, in milliseconds.
    fn duration_ms(&self) -> i64;
    /// The streamer's decoder.
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened;
    /// The streamer's input context.
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input;
    /// The streamer's state.
    fn player_state(&self) -> &Shared<PlayerState>;
    /// Output a frame from the decoder.
    fn decode_frame(&mut self) -> Result<Self::Frame>;
    /// Ignore the remainder of this packet.
    fn drop_frames(&mut self) -> Result<()> {
        if self.decode_frame().is_err() {
            self.receive_next_packet(None)
        } else {
            self.drop_frames()
        }
    }
    /// Receive the next packet of the stream.
    fn receive_next_packet(&mut self, skip_until: Option<i64>) -> Result<()> {
        if let Some(packet) = self.input_context().packets().next() {
            let (stream, packet) = packet?;
            let time_base = stream.time_base();
            if stream.index() == *self.stream_index() {
                match packet.dts() {
                    // Don't try to set elapsed time off of undefined timestamp values
                    Some(ffmpeg::ffi::AV_NOPTS_VALUE) => (),
                    Some(dts) => {
                        self.elapsed_ms().set(timestamp_to_millisec(dts, time_base));

                        if let Some(skip_until) = skip_until {
                            if skip_until > self.elapsed_ms().get() {
                                return Ok(());
                            }
                        }

                        self.decoder().send_packet(&packet)?;
                    }
                    _ => (),
                }
            }
        } else {
            self.decoder().send_eof()?;
        }
        Ok(())
    }
    /// Reset the stream to its initial state.
    fn reset(&mut self) {
        let _ = self.input_context().seek(0, ..);
        self.decoder().flush();
    }
    /// Keep receiving packets until a frame can be decoded.
    fn receive_next_packet_until_frame(&mut self) -> Result<Self::ProcessedFrame> {
        match self.receive_next_frame() {
            Ok(frame_result) => Ok(frame_result),
            Err(e) => {
                // dbg!(&e, is_ffmpeg_incomplete_error(&e));
                if is_ffmpeg_incomplete_error(&e) {
                    self.receive_next_packet(None)?;
                    self.receive_next_packet_until_frame()
                } else {
                    Err(e)
                }
            }
        }
    }
    /// Process a decoded frame.
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame>;
    /// Apply a processed frame
    fn apply_frame(&mut self, _frame: Self::ProcessedFrame) {}
    /// Decode and process a frame.
    fn receive_next_frame(&mut self) -> Result<Self::ProcessedFrame> {
        match self.decode_frame() {
            Ok(decoded_frame) => self.process_frame(decoded_frame),
            Err(e) => Err(e),
        }
    }
}

impl Streamer for VideoStreamer {
    type Frame = Video;
    type ProcessedFrame = Video;
    fn stream_type(&self) -> Type {
        Type::Video
    }
    fn is_primary_streamer(&self) -> bool {
        true
    }
    fn stream_index(&self) -> StreamIndex {
        self.video_stream_index
    }
    fn cycle_stream(&mut self) -> StreamIndex {
        StreamIndex::from(0)
    }
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened {
        &mut self.video_decoder.0
    }
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.input_context
    }
    fn elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn primary_elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn duration_ms(&self) -> i64 {
        self.duration_ms
    }
    fn player_state(&self) -> &Shared<PlayerState> {
        &self.player_state
    }
    fn decode_frame(&mut self) -> Result<Self::Frame> {
        let mut decoded_frame = Video::empty();
        self.video_decoder.receive_frame(&mut decoded_frame)?;
        Ok(decoded_frame)
    }
    fn apply_frame(&mut self, frame: Self::ProcessedFrame) {
        if let Some(apply_video_frame_fn) = self.apply_video_frame_fn.as_mut() {
            apply_video_frame_fn(frame)
        }
    }
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame> {
        let mut rgb_frame = Video::empty();
        let mut scaler = Context::get(
            frame.format(),
            frame.width(),
            frame.height(),
            Pixel::RGBA,
            self.output_frame_width,
            self.output_frame_height,
            Flags::BILINEAR,
        )?;
        scaler.run(&frame, &mut rgb_frame)?;

        Ok(rgb_frame)
    }
}

impl Streamer for AudioStreamer {
    type Frame = Audio;
    type ProcessedFrame = ();
    fn stream_type(&self) -> Type {
        Type::Audio
    }
    fn is_primary_streamer(&self) -> bool {
        false
    }
    fn stream_index(&self) -> StreamIndex {
        self.audio_stream_indices[0]
    }
    fn cycle_stream(&mut self) -> StreamIndex {
        self.audio_stream_indices.rotate_right(1);
        let new_stream_index = self.stream_index();
        let new_decoder = get_decoder_from_stream_index(&self.input_context, new_stream_index)
            .unwrap()
            .audio()
            .unwrap();
        let new_resampler = ffmpeg::software::resampling::context::Context::get2(
            new_decoder.format(),
            new_decoder.ch_layout(),
            new_decoder.rate(),
            self.resampler.output().format,
            ChannelLayout::STEREO,
            self.resampler.output().rate,
        )
        .unwrap();
        self.audio_decoder = new_decoder;
        self.resampler = new_resampler;
        new_stream_index
    }
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened {
        &mut self.audio_decoder.0
    }
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.input_context
    }
    fn elapsed_ms(&self) -> &Shared<i64> {
        &self.audio_elapsed_ms
    }
    fn primary_elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn duration_ms(&self) -> i64 {
        self.duration_ms
    }
    fn player_state(&self) -> &Shared<PlayerState> {
        &self.player_state
    }
    fn decode_frame(&mut self) -> Result<Self::Frame> {
        let mut decoded_frame = Audio::empty();
        self.audio_decoder.receive_frame(&mut decoded_frame)?;
        Ok(decoded_frame)
    }
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame> {
        let mut resampled_frame = ffmpeg::frame::Audio::empty();
        self.resampler.run(&frame, &mut resampled_frame)?;
        let audio_samples = if resampled_frame.is_packed() {
            packed(&resampled_frame)
        } else {
            resampled_frame.plane(0)
        };
        let _ = self.audio_sample_producer.send(audio_samples.to_vec());
        Ok(())
    }
}

impl Streamer for SubtitleStreamer {
    type Frame = (ffmpeg::codec::subtitle::Subtitle, i64);
    type ProcessedFrame = Subtitle;
    fn stream_type(&self) -> Type {
        Type::Subtitle
    }
    fn is_primary_streamer(&self) -> bool {
        false
    }
    fn stream_index(&self) -> StreamIndex {
        self.subtitle_stream_indices[0]
    }
    fn cycle_stream(&mut self) -> StreamIndex {
        self.subtitle_stream_indices.rotate_right(1);
        self.subtitle_decoder.flush();
        let new_stream_index = self.stream_index();
        let new_decoder = get_decoder_from_stream_index(&self.input_context, new_stream_index)
            .unwrap()
            .subtitle()
            .unwrap();
        self.next_packet = None;
        // bandaid: subtitle decoder is always ahead of video decoder, so we need to seek it back to the
        // video decoder's location in order so that we don't miss possible subtitles when switching streams
        self.seek(self.primary_elapsed_ms().get() as f32 / self.duration_ms as f32);
        self.subtitles_queue.lock().clear();
        self.subtitle_decoder = new_decoder;
        new_stream_index
    }
    fn decoder(&mut self) -> &mut ffmpeg::decoder::Opened {
        &mut self.subtitle_decoder.0
    }
    fn input_context(&mut self) -> &mut ffmpeg::format::context::Input {
        &mut self.input_context
    }
    fn elapsed_ms(&self) -> &Shared<i64> {
        &self.subtitle_elapsed_ms
    }
    fn primary_elapsed_ms(&self) -> &Shared<i64> {
        &self.video_elapsed_ms
    }
    fn duration_ms(&self) -> i64 {
        self.duration_ms
    }
    fn player_state(&self) -> &Shared<PlayerState> {
        &self.player_state
    }
    fn receive_next_packet(&mut self, _: Option<i64>) -> Result<()> {
        if let Some(packet) = self.input_context().packets().next() {
            let (stream, packet) = packet?;
            let time_base = stream.time_base();
            if stream.index() == *self.stream_index() {
                if let Some(dts) = packet.dts() {
                    self.elapsed_ms().set(timestamp_to_millisec(dts, time_base));
                }
                self.next_packet = Some(packet);
            }
        } else {
            self.decoder().send_eof()?;
        }
        Ok(())
    }
    fn decode_frame(&mut self) -> Result<Self::Frame> {
        if let Some(packet) = self.next_packet.take() {
            let mut decoded_frame = ffmpeg::Subtitle::new();
            self.subtitle_decoder.decode(&packet, &mut decoded_frame)?;
            Ok((decoded_frame, packet.duration()))
        } else {
            Err(ffmpeg::Error::from(AVERROR(EAGAIN)).into())
        }
    }
    fn process_frame(&mut self, frame: Self::Frame) -> Result<Self::ProcessedFrame> {
        // TODO: manage the case when frame rects len > 1
        let (frame, duration) = frame;
        if let Some(rect) = frame.rects().next() {
            Subtitle::from_ffmpeg_rect(rect).map(|s| {
                if s.remaining_duration_ms == 0 {
                    s.with_duration_ms(duration)
                } else {
                    s
                }
            })
        } else {
            anyhow::bail!("no subtitle")
        }
    }
    fn apply_frame(&mut self, frame: Self::ProcessedFrame) {
        let mut queue = self.subtitles_queue.lock();
        queue.push_back(frame)
    }
}

type FfmpegAudioFormat = ffmpeg::format::Sample;
type FfmpegAudioFormatType = ffmpeg::format::sample::Type;
trait AsFfmpegSample {
    fn to_ffmpeg_sample(&self) -> FfmpegAudioFormat;
}

impl AsFfmpegSample for cpal::SampleFormat {
    fn to_ffmpeg_sample(&self) -> FfmpegAudioFormat {
        match self {
            cpal::SampleFormat::U8 => FfmpegAudioFormat::U8(FfmpegAudioFormatType::Packed),
            cpal::SampleFormat::I8 => panic!("unsupported audio format"),
            cpal::SampleFormat::U16 => panic!("unsupported audio format"),
            cpal::SampleFormat::I16 => FfmpegAudioFormat::I16(FfmpegAudioFormatType::Packed),
            cpal::SampleFormat::U32 => panic!("unsupported audio format"),
            cpal::SampleFormat::I32 => FfmpegAudioFormat::I32(FfmpegAudioFormatType::Packed),
            cpal::SampleFormat::U64 => panic!("unsupported audio format"),
            cpal::SampleFormat::I64 => FfmpegAudioFormat::I64(FfmpegAudioFormatType::Packed),
            cpal::SampleFormat::F32 => FfmpegAudioFormat::F32(FfmpegAudioFormatType::Packed),
            cpal::SampleFormat::F64 => FfmpegAudioFormat::F64(FfmpegAudioFormatType::Packed),
            _ => panic!("unsupported audio format"),
        }
    }
}

/// Pipes audio samples to cpal.
#[derive(Default)]
pub struct AudioDeviceCallback {
    sample_streams: Vec<AudioSampleStream>,
    seeking: Option<Shared<bool>>,
}

struct ChunkSampler {
    chunks: StreamingAudioChunks,
    processed: usize,
}

impl ChunkSampler {
    fn new(chunk: StreamingAudioChunks) -> Self {
        ChunkSampler {
            chunks: chunk,
            processed: 0,
        }
    }

    fn get_sample(&mut self) -> f32 {
        if !self.finished() {
            let sample = self.chunks[self.processed];
            self.processed += 1;
            return sample
        }
        0.0
    }

    fn finished(&self) -> bool {
        self.processed >= self.chunks.len()
    }
}

struct AudioSampleStream {
    sample_consumer: AudioSampleConsumer,
    audio_volume: Shared<f32>,
    chunks: Option<ChunkSampler>,
}

impl AudioSampleStream {
    fn get_sample(&mut self) -> f32 {
        if self.chunks.is_none() || self.chunks.as_ref().unwrap().finished() {
            match self.sample_consumer.try_recv() {
                Ok(result) => self.chunks = Some(ChunkSampler::new(result)),
                Err(_) => (),
            }
        }
        if self.chunks.is_some() {
            self.chunks.as_mut().unwrap().get_sample()
        } else {
            0.0
        }
    }
}

impl AudioDeviceCallback {
    fn callback<T>(&mut self, output: &mut [T])
    where
        T: cpal::Sample + cpal::FromSample<f32> + std::iter::Sum<f32>,
    {
        if self.seeking.is_some() && self.seeking.as_ref().unwrap().get() {
            for stream in self.sample_streams.iter() {
                // clear until there's nothing left
                while let Ok(_) = stream.sample_consumer.try_recv() {
                    //println!("draining audio receiver...");
                }
            }
            self.seeking.as_ref().unwrap().set(false);
        }

        for x in output.iter_mut() {
            *x = self
                .sample_streams
                .iter_mut()
                .map(|s| {
                    s.get_sample() * s.audio_volume.get()
                })
                .sum()
        }
    }
}

#[inline]
// Thanks https://github.com/zmwangx/rust-ffmpeg/issues/72 <3
// Interpret the audio frame's data as packed (alternating channels, 12121212, as opposed to planar 11112222)
fn packed<T: ffmpeg::frame::audio::Sample>(frame: &ffmpeg::frame::Audio) -> &[T] {
    if !frame.is_packed() {
        panic!("data is not packed");
    }

    if !<T as ffmpeg::frame::audio::Sample>::is_valid(
        frame.format(),
        frame.ch_layout().channels() as u16,
    ) {
        panic!("unsupported type");
    }

    unsafe {
        std::slice::from_raw_parts(
            (*frame.as_ptr()).data[0] as *const T,
            frame.samples() * frame.ch_layout().channels() as usize,
        )
    }
}

fn is_ffmpeg_eof_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<ffmpeg::Error>(),
        Some(ffmpeg::Error::Eof)
    )
}

fn is_ffmpeg_incomplete_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<ffmpeg::Error>(),
        Some(ffmpeg::Error::Other { errno } ) if *errno == EAGAIN
    )
}

pub fn video_frame_to_image(frame: Video) -> ColorImage {
    let size = [frame.width() as usize, frame.height() as usize];
    let data = frame.data(0);
    let stride = frame.stride(0);
    let pixel_size_bytes = 4;
    let byte_width: usize = pixel_size_bytes * frame.width() as usize;
    let height: usize = frame.height() as usize;
    let mut pixels = vec![];
    pixels.reserve(size[0] * size[1]);
    for line in 0..height {
        let begin = line * stride;
        let end = begin + byte_width;
        let data_line = &data[begin..end];
        let pixel_line: &[Color32] = bytemuck::cast_slice(data_line);
        pixels.extend_from_slice(pixel_line);
    }
    ColorImage { size, pixels }
}
