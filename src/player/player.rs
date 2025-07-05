use anyhow::Result;
use chrono::{Duration};

use ffmpeg::format::{input};
use ffmpeg::media::Type;
use ffmpeg::{ChannelLayout};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use crate::subtitle::Subtitle;
use timer::{Guard, Timer};
use std::sync::mpsc;
use egui::{ColorImage, Vec2};
use crate::{get_decoder_from_stream_index, get_stream_indices_of_type, is_ffmpeg_eof_error, timestamp_to_millisec, AsFfmpegSample, AudioDevice, AudioSampleStream, AudioStreamer, PlayerMessage, PlayerMessageReciever, PlayerMessageSender, PlayerState, Shared, StreamIndex, StreamInfo, Streamer, StreamingAudioChunks, SubtitleQueue, SubtitleStreamer, VideoStreamer, AV_TIME_BASE_RATIONAL};

/// Configurable aspects of a [`crate::Player`].
#[derive(Clone, Debug)]
pub struct PlayerOptions {
    /// Should the stream loop if it finishes?
    pub looping: bool,
    /// The volume of the audio stream.
    pub audio_volume: Shared<f32>,
    /// The maximum volume of the audio stream.
    pub max_audio_volume: f32,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self {
            looping: false,
            max_audio_volume: 1.,
            audio_volume: Shared::new(0.5),
        }
    }
}

pub type OnNewFrame = Box<dyn Fn(ColorImage) + Send + Sync>;

impl PlayerOptions {
    /// Set the maximum player volume, and scale the actual player volume to the
    /// same current ratio.
    pub fn set_max_audio_volume(&mut self, volume: f32) {
        self.audio_volume
            .set(self.audio_volume.get() * (volume / self.max_audio_volume));
        self.max_audio_volume = volume;
    }

    /// Set the player volume, clamped in `0.0..=max_audio_volume`.
    pub fn set_audio_volume(&mut self, volume: f32) {
        self.audio_volume
            .set(volume.clamp(0., self.max_audio_volume));
    }
}

/// The [`Player`] processes and controls streams of video/audio. This is what you use to show a video file.
/// Initialize once, and use the [`Player::ui`] or [`Player::ui_at()`] functions to show the playback.
pub struct Player {
    /// The video streamer of the player.
    pub video_streamer: Arc<Mutex<VideoStreamer>>,
    /// The audio streamer of the player. Won't exist unless [`Player::with_audio`] is called and there exists
    /// a valid audio stream in the file.
    pub audio_streamer: Option<Arc<Mutex<AudioStreamer>>>,
    /// The subtitle streamer of the player. Won't exist unless [`Player::with_subtitles`] is called and there exists
    /// a valid subtitle stream in the file.
    pub subtitle_streamer: Option<Arc<Mutex<SubtitleStreamer>>>,
    /// The state of the player.
    pub player_state: Shared<PlayerState>,
    /// The size of the video stream.
    pub size: Vec2,
    /// The total duration of the stream, in milliseconds.
    pub duration_ms: i64,
    /// The framerate of the video stream, in frames per second.
    pub framerate: f64,
    /// Configures certain aspects of this [`Player`].
    pub options: PlayerOptions,
    pub audio_stream_info: StreamInfo,
    pub subtitle_stream_info: StreamInfo,

    on_new_frame: Option<Arc<OnNewFrame>>,

    message_sender: PlayerMessageSender,
    message_receiver: PlayerMessageReciever,
    video_timer: Timer,
    audio_timer: Timer,
    subtitle_timer: Timer,
    audio_thread: Option<Guard>,
    video_thread: Option<Guard>,
    subtitle_thread: Option<Guard>,
    // ctx_ref: egui::Context,
    last_seek_ms: Option<i64>,
    preseek_player_state: Option<PlayerState>,
    #[cfg(feature = "from_bytes")]
    temp_file: Option<NamedTempFile>,
    video_elapsed_ms: Shared<i64>,
    audio_elapsed_ms: Shared<i64>,
    subtitle_elapsed_ms: Shared<i64>,
    seeking_signal: Shared<bool>,
    video_elapsed_ms_override: Option<i64>,
    subtitles_queue: SubtitleQueue,
    current_subtitles: Vec<Subtitle>,
    input_path: String,
}

impl Player {
    /// Create a new [`Player`].
    pub fn new(input_path: &String) -> Result<Self> {
        let input_context = input(&input_path)?;
        let video_stream = input_context
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = StreamIndex::from(video_stream.index());

        let video_elapsed_ms = Shared::new(0);
        let audio_elapsed_ms = Shared::new(0);
        let seeking_signal = Shared::new(false);
        let player_state = Shared::new(PlayerState::Stopped);

        let video_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let video_decoder = video_context.decoder().video()?;
        let framerate = (video_stream.avg_frame_rate().numerator() as f64)
            / video_stream.avg_frame_rate().denominator() as f64;
        let (width, height) = (video_decoder.width(), video_decoder.height());
        let size = Vec2::new(width as f32, height as f32);
        let duration_ms = timestamp_to_millisec(input_context.duration(), AV_TIME_BASE_RATIONAL); // in sec

        let stream_decoder = VideoStreamer {
            apply_video_frame_fn: None,
            duration_ms,
            video_decoder,
            video_stream_index,
            _audio_elapsed_ms: audio_elapsed_ms.clone(),
            video_elapsed_ms: video_elapsed_ms.clone(),
            input_context,
            player_state: player_state.clone(),
        };

        let options = PlayerOptions::default();
        let (message_sender, message_receiver) = std::sync::mpsc::channel();

        let streamer = Self {
            input_path: input_path.clone(),
            audio_streamer: None,
            subtitle_streamer: None,
            video_streamer: Arc::new(Mutex::new(stream_decoder)),
            subtitle_stream_info: StreamInfo::new(),
            audio_stream_info: StreamInfo::new(),
            framerate,
            video_timer: Timer::new(),
            audio_timer: Timer::new(),
            subtitle_timer: Timer::new(),
            subtitle_elapsed_ms: Shared::new(0),
            preseek_player_state: None,
            video_thread: None,
            subtitle_thread: None,
            audio_thread: None,
            player_state,
            message_sender,
            message_receiver,
            video_elapsed_ms,
            audio_elapsed_ms,
            seeking_signal,
            size,
            last_seek_ms: None,
            duration_ms,
            options,
            video_elapsed_ms_override: None,
            subtitles_queue: Arc::new(Mutex::new(VecDeque::new())),
            current_subtitles: Vec::new(),
            #[cfg(feature = "from_bytes")]
            temp_file: None,

            on_new_frame: None,
        };

        Ok(streamer)
    }

    /// Resets player to initial state
    pub fn reset(&mut self) {
        self.last_seek_ms = None;
        self.video_elapsed_ms_override = None;
        self.video_elapsed_ms.set(0);
        self.audio_elapsed_ms.set(0);
        self.video_streamer.lock().reset();
        if let Some(audio_decoder) = self.audio_streamer.as_mut() {
            audio_decoder.lock().reset();
        }
    }

    /// The elapsed duration of the stream, in milliseconds. This value won't be truly accurate to the decoders
    /// while seeking, and will instead be overridden with the target seek location (for visual representation purposes).
    pub fn elapsed_ms(&self) -> i64 {
        self.video_elapsed_ms_override
            .as_ref()
            .map(|i| *i)
            .unwrap_or(self.video_elapsed_ms.get())
    }

    /// Pause the stream.
    pub fn pause(&mut self) {
        self.set_state(PlayerState::Paused)
    }

    /// Resume the stream from a paused state.
    pub fn resume(&mut self) {
        self.set_state(PlayerState::Playing)
    }

    /// Stop the stream.
    pub fn stop(&mut self) {
        self.set_state(PlayerState::Stopped);
        self.video_thread = None;
        self.audio_thread = None;
        self.reset()
    }

    /// Seek to a location in the stream.
    pub fn seek(&mut self, seek_frac: f32) {
        let current_state = self.player_state.get();
        if !matches!(current_state, PlayerState::SeekingInProgress) {
            match current_state {
                PlayerState::Stopped | PlayerState::EndOfFile => {
                    self.preseek_player_state = Some(PlayerState::Paused);
                    self.start();
                }
                PlayerState::Paused | PlayerState::Playing => {
                    self.preseek_player_state = Some(current_state);
                }
                _ => (),
            }

            let video_streamer = self.video_streamer.clone();
            let mut audio_streamer = self.audio_streamer.clone();
            let mut subtitle_streamer = self.subtitle_streamer.clone();
            let subtitle_queue = self.subtitles_queue.clone();

            self.last_seek_ms = Some((seek_frac as f64 * self.duration_ms as f64) as i64);
            self.set_state(PlayerState::SeekingInProgress);

            if let Some(audio_streamer) = audio_streamer.take() {
                std::thread::spawn(move || {
                    audio_streamer.lock().seek(seek_frac);
                });
            };
            if let Some(subtitle_streamer) = subtitle_streamer.take() {
                self.current_subtitles.clear();
                std::thread::spawn(move || {
                    subtitle_queue.lock().clear();
                    subtitle_streamer.lock().seek(seek_frac);
                });
            };
            std::thread::spawn(move || {
                video_streamer.lock().seek(seek_frac);
            });
        }
    }

    /// Start the stream.
    pub fn start(&mut self) {
        self.stop();
        self.spawn_timers();
        self.resume();
    }

    /// Process player state updates. This function must be called for proper function
    /// of the player.
    pub fn process_state(&mut self) {
        let mut reset_stream = false;

        match self.player_state.get() {
            PlayerState::EndOfFile => {
                if self.options.looping {
                    reset_stream = true;
                } else {
                    self.player_state.set(PlayerState::Stopped);
                }
            }
            PlayerState::Playing => {
                for subtitle in self.current_subtitles.iter_mut() {
                    subtitle.remaining_duration_ms -= (1000. / self.framerate) as i64;
                }
                self.current_subtitles
                    .retain(|s| s.remaining_duration_ms > 0);
                if let Some(mut queue) = self.subtitles_queue.try_lock() {
                    if queue.len() > 1 {
                        self.current_subtitles.push(queue.pop_front().unwrap());
                    }
                }
            }
            state @ (PlayerState::SeekingInProgress | PlayerState::SeekingFinished) => {
                if self.last_seek_ms.is_some() {
                    let last_seek_ms = *self.last_seek_ms.as_ref().unwrap();
                    if matches!(state, PlayerState::SeekingFinished) {
                        if let Some(previous_player_state) = self.preseek_player_state {
                            self.set_state(previous_player_state)
                        }
                        self.video_elapsed_ms_override = None;
                        self.last_seek_ms = None;
                    } else {
                        self.seeking_signal.set(true);
                        self.video_elapsed_ms_override = Some(last_seek_ms);
                    }
                } else {
                    self.video_elapsed_ms_override = None;
                }
            }
            PlayerState::Restarting => reset_stream = true,
            _ => (),
        }
        if let Ok(message) = self.message_receiver.try_recv() {
            match message {
                PlayerMessage::StreamCycled(stream_type) => match stream_type {
                    Type::Audio => self.audio_stream_info.cycle(),
                    Type::Subtitle => {
                        self.current_subtitles.clear();
                        self.subtitle_stream_info.cycle();
                    }
                    _ => unreachable!(),
                },
            }
        }
        if reset_stream {
            self.reset();
            self.resume();
        }
    }

    /// Initializes the audio stream (if there is one), required for making a [`Player`] output audio.
    /// Will stop and reset the player's state.
    pub fn add_audio(&mut self, audio_device: &mut AudioDevice) -> Result<()> {
        let audio_input_context = input(&self.input_path)?;
        let audio_stream_indices = get_stream_indices_of_type(&audio_input_context, Type::Audio);

        let audio_streamer = if !audio_stream_indices.is_empty() {
            let audio_decoder =
                get_decoder_from_stream_index(&audio_input_context, audio_stream_indices[0])?
                    .audio()?;

            let (audio_sample_producer, audio_sample_consumer) = mpsc::channel::<StreamingAudioChunks>();
            let audio_resampler = ffmpeg::software::resampling::context::Context::get2(
                audio_decoder.format(),
                audio_decoder.ch_layout(),
                audio_decoder.rate(),
                audio_device.get_sample_format().to_ffmpeg_sample(),
                ChannelLayout::STEREO,
                audio_device.get_sample_rate(),
            )?;

            audio_device
                .callback
                .lock()
                .sample_streams
                .push(AudioSampleStream {
                    sample_consumer: audio_sample_consumer,
                    audio_volume: self.options.audio_volume.clone(),
                    chunks: None,
                });

            audio_device
                .callback
                .lock()
                .seeking = Some(self.seeking_signal.clone());

            audio_device.play();

            self.stop();
            self.audio_stream_info = StreamInfo::from_total(audio_stream_indices.len());
            Some(AudioStreamer {
                duration_ms: self.duration_ms,
                player_state: self.player_state.clone(),
                video_elapsed_ms: self.video_elapsed_ms.clone(),
                audio_elapsed_ms: self.audio_elapsed_ms.clone(),
                audio_sample_producer,
                input_context: audio_input_context,
                audio_decoder,
                resampler: audio_resampler,
                audio_stream_indices,
            })
        } else {
            None
        };
        self.audio_streamer = audio_streamer.map(|s| Arc::new(Mutex::new(s)));
        Ok(())
    }

    /// Initializes the subtitle stream (if there is one), required for making a [`Player`] display subtitles.
    /// Will stop and reset the player's state.
    pub fn add_subtitles(&mut self) -> Result<()> {
        let subtitle_input_context = input(&self.input_path)?;
        let subtitle_stream_indices =
            get_stream_indices_of_type(&subtitle_input_context, Type::Subtitle);

        let subtitle_streamer = if !subtitle_stream_indices.is_empty() {
            let subtitle_decoder =
                get_decoder_from_stream_index(&subtitle_input_context, subtitle_stream_indices[0])?
                    .subtitle()?;

            self.stop();
            self.subtitle_stream_info = StreamInfo::from_total(subtitle_stream_indices.len());
            Some(SubtitleStreamer {
                next_packet: None,
                duration_ms: self.duration_ms,
                player_state: self.player_state.clone(),
                video_elapsed_ms: self.video_elapsed_ms.clone(),
                _audio_elapsed_ms: self.audio_elapsed_ms.clone(),
                subtitle_elapsed_ms: self.subtitle_elapsed_ms.clone(),
                input_context: subtitle_input_context,
                subtitles_queue: self.subtitles_queue.clone(),
                subtitle_decoder,
                subtitle_stream_indices,
            })
        } else {
            None
        };
        self.subtitle_streamer = subtitle_streamer.map(|s| Arc::new(Mutex::new(s)));
        Ok(())
    }

    /// Switches to the next subtitle stream.
    pub fn cycle_subtitle_stream(&mut self) {
        self.cycle_stream(self.subtitle_streamer.as_ref());
    }

    /// Switches to the next audio stream.
    pub fn cycle_audio_stream(&mut self) {
        self.cycle_stream(self.audio_streamer.as_ref());
    }

    /// Enables using [`Player::add_audio`] with the builder pattern.
    pub fn with_audio(mut self, audio_device: &mut AudioDevice) -> Result<Self> {
        self.add_audio(audio_device)?;
        Ok(self)
    }

    /// Enables using [`Player::add_subtitles`] with the builder pattern.
    pub fn with_subtitles(mut self) -> Result<Self> {
        self.add_subtitles()?;
        Ok(self)
    }

    /// Create a callback to be called when new video frame is available
    pub fn set_on_new_frame(mut self, on_new_frame: OnNewFrame) -> Self {
        self.on_new_frame = Some(Arc::new(on_new_frame));

        self
    }

    /// Get thumbnail of a video
    pub fn get_thumbnail(&mut self) -> ColorImage {
        loop {
            if let Ok(thumbnail) = self.video_streamer.lock().receive_next_packet_until_frame() {
                return thumbnail;
            }
        }
    }

    fn set_state(&mut self, new_state: PlayerState) {
        self.player_state.set(new_state)
    }

    fn spawn_timers(&mut self) {
        // let mut texture_handle = self.texture_handle.clone();
        // let texture_options = self.options.texture_options;
        // let ctx = self.ctx_ref.clone();
        let nanos = 1e9 / self.framerate;
        let wait_duration = Duration::nanoseconds(nanos as i64);

        fn play<T: Streamer>(streamer: &Weak<Mutex<T>>) {
            if let Some(streamer) = streamer.upgrade() {
                if let Some(mut streamer) = streamer.try_lock() {
                    if (streamer.player_state().get() == PlayerState::Playing)
                        && streamer.primary_elapsed_ms().get() >= streamer.elapsed_ms().get()
                    {
                        match streamer.receive_next_packet_until_frame() {
                            Ok(frame) => streamer.apply_frame(frame),
                            Err(e) => {
                                if is_ffmpeg_eof_error(&e) && streamer.is_primary_streamer() {
                                    streamer.player_state().set(PlayerState::EndOfFile)
                                }
                            }
                        }

                        // if desync passes n threshold, try to sync again by skipping frames
                        let max_desync = streamer.primary_elapsed_ms().get() - 10;
                        while streamer.elapsed_ms().get() < max_desync {
                            let _ = streamer.receive_next_packet(Some(max_desync));
                        }
                    }
                }
            }
        }

        let on_new_frame = self.on_new_frame.clone();
        if let Some(on_new_frame) = on_new_frame {
            let on_new_frame = on_new_frame.clone();
            self.video_streamer.lock().apply_video_frame_fn = Some(Box::new(move |frame| {
                on_new_frame(frame);
            }));
        }

        let video_streamer_ref = Arc::downgrade(&self.video_streamer);
        let video_timer_guard = self.video_timer.schedule_repeating(wait_duration, move || {
            play(&video_streamer_ref);
        });

        self.video_thread = Some(video_timer_guard);

        if let Some(audio_decoder) = self.audio_streamer.as_ref() {
            let audio_decoder_ref = Arc::downgrade(audio_decoder);
            let audio_timer_guard = self
                .audio_timer
                // sleep to free resources for other tasks, otherwise, due to sync logic, audio will take all cpu time, and video won't stream
                .schedule_repeating(Duration::milliseconds(1), move || play(&audio_decoder_ref));
            self.audio_thread = Some(audio_timer_guard);
        }

        if let Some(subtitle_decoder) = self.subtitle_streamer.as_ref() {
            let subtitle_decoder_ref = Arc::downgrade(subtitle_decoder);
            let subtitle_timer_guard = self
                .subtitle_timer
                .schedule_repeating(wait_duration, move || play(&subtitle_decoder_ref));
            self.subtitle_thread = Some(subtitle_timer_guard);
        }
    }

    fn cycle_stream<T: Streamer + 'static>(&self, mut streamer: Option<&Arc<Mutex<T>>>) {
        if let Some(streamer) = streamer.take() {
            let message_sender = self.message_sender.clone();
            let streamer = streamer.clone();
            std::thread::spawn(move || {
                let mut streamer = streamer.lock();
                streamer.cycle_stream();
                message_sender.send(PlayerMessage::StreamCycled(streamer.stream_type()))
            });
        };
    }
}
