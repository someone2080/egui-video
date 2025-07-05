use std::sync::{Arc, Mutex as StdMutex};
use anyhow::Result;
use chrono::Duration;
use egui::{vec2, Align2, Color32, ColorImage, CornerRadius, FontId, Image, Pos2, Rect, Response, Sense, Shadow, Spinner, TextureHandle, TextureOptions, Ui, Vec2};
use egui::emath::RectTransform;
use egui::load::SizedTexture;
use ffmpeg::media::Type;
use crate::{format_duration, AudioDevice, PlayerState};
use crate::player::player::Player;
use crate::subtitle::Subtitle;

/// The [`crate::Player`] processes and controls streams of video/audio. This is what you use to show a video file.
/// Initialize once, and use the [`crate::Player::ui`] or [`crate::Player::ui_at()`] functions to show the playback.
pub struct GuiPlayer {
    /// The player's texture handle.
    pub texture_handle: Arc<StdMutex<TextureHandle>>,
    /// Player
    pub player: Player,
    /// Player's texture options
    pub texture_options: Arc<TextureOptions>,

    ctx_ref: Arc<egui::Context>,
    current_subtitles: Vec<Subtitle>,
}

impl GuiPlayer {
    /// Create a new [`GuiPlayer`].
    pub fn new(ctx: &egui::Context, input_path: &String) -> Result<Self> {
        let texture_options = Arc::new(TextureOptions::default());
        let texture_handle = Arc::new(StdMutex::new(ctx.load_texture("vidstream", ColorImage::example(), *texture_options)));

        let player = Player::new(input_path)?;

        // frame listener
        let c_texture_options = texture_options.clone();
        let c_texture_handle = texture_handle.clone();
        let c_ctx = ctx.clone();
        let player = player.set_on_new_frame(Box::new(move |frame| {
            c_texture_handle.lock().unwrap().set(frame, *c_texture_options);
            c_ctx.request_repaint();
        }));

        let mut gui_player = Self {
            texture_handle,
            ctx_ref: Arc::new(ctx.clone()),
            texture_options,
            player,
            current_subtitles: Vec::new(),
        };

        let thumbnail = gui_player.player.get_thumbnail();
        gui_player.texture_handle.lock().unwrap().set(thumbnail, *gui_player.texture_options);

        Ok(gui_player)
    }

    /// A formatted string for displaying the duration of the video stream.
    pub fn duration_text(&mut self) -> String {
        format!(
            "{} / {}",
            format_duration(Duration::milliseconds(self.player.elapsed_ms())),
            format_duration(Duration::milliseconds(self.player.duration_ms))
        )
    }

    /// Pause the stream.
    pub fn pause(&mut self) {
        self.player.pause();
    }

    /// Resume the stream from a paused state.
    pub fn resume(&mut self) {
        self.player.resume();
    }

    /// Stop the stream.
    pub fn stop(&mut self) {
        self.player.stop();
    }

    /// Seek to a location in the stream.
    pub fn seek(&mut self, seek_frac: f32) {
        self.player.seek(seek_frac);
    }

    /// Start the stream.
    pub fn start(&mut self) {
        self.player.start();
    }

    /// Process player state updates. This function must be called for proper function
    /// of the player. This function is already included in  [`crate::Player::ui`] or
    /// [`crate::Player::ui_at`].
    pub fn process_state(&mut self) {
        self.player.process_state();
    }

    /// Create the [`egui::Image`] for the video frame.
    pub fn generate_frame_image(&self, size: Vec2) -> Image {
        Image::new(SizedTexture::new(self.texture_handle.lock().unwrap().id(), size)).sense(Sense::click())
    }

    /// Draw the video frame with a specific rect (without controls). Make sure to call [`crate::Player::process_state`].
    pub fn render_frame(&self, ui: &mut Ui, size: Vec2) -> Response {
        ui.add(self.generate_frame_image(size))
    }

    /// Draw the video frame (without controls). Make sure to call [`crate::Player::process_state`].
    pub fn render_frame_at(&self, ui: &mut Ui, rect: Rect) -> Response {
        ui.put(rect, self.generate_frame_image(rect.size()))
    }

    /// Draw the video frame and player controls and process state changes.
    pub fn ui(&mut self, ui: &mut Ui, size: Vec2) -> egui::Response {
        let frame_response = self.render_frame(ui, size);
        self.render_controls(ui, &frame_response);
        self.render_subtitles(ui, &frame_response);
        self.process_state();
        frame_response
    }

    /// Draw the video frame and player controls with a specific rect, and process state changes.
    pub fn ui_at(&mut self, ui: &mut Ui, rect: Rect) -> egui::Response {
        let frame_response = self.render_frame_at(ui, rect);
        self.render_controls(ui, &frame_response);
        self.render_subtitles(ui, &frame_response);
        self.process_state();
        frame_response
    }

    /// Draw the subtitles, if any. Only works when a subtitle streamer has been already created with
    /// [`crate::Player::add_subtitles`] or [`crate::Player::with_subtitles`] and a valid subtitle stream exists.
    pub fn render_subtitles(&mut self, ui: &mut Ui, frame_response: &Response) {
        let original_rect_center_bottom = Pos2::new(self.player.size.x / 2., self.player.size.y);
        let mut last_bottom = self.player.size.y;
        for subtitle in self.current_subtitles.iter() {
            let transform = RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, self.player.size),
                frame_response.rect,
            );
            let text_rect = ui.painter().text(
                subtitle
                    .position
                    .map(|p| transform.transform_pos(p))
                    .unwrap_or_else(|| {
                        //TODO incorporate left/right margin
                        let mut center_bottom = original_rect_center_bottom;
                        center_bottom.y = center_bottom.y.min(last_bottom) - subtitle.margin.bottom as f32;
                        transform.transform_pos(center_bottom)
                    }),
                subtitle.alignment,
                &subtitle.text,
                FontId::proportional(transform.transform_pos(Pos2::new(subtitle.font_size, 0.)).x),
                subtitle.primary_fill,
            );
            last_bottom = transform.inverse().transform_pos(text_rect.center_top()).y;
        }
    }

    /// Draw the player controls. Make sure to call [`crate::Player::process_state()`]. Unless you are explicitly
    /// drawing something in between the video frames and controls, it is probably better to use
    /// [`crate::Player::ui`] or [`crate::Player::ui_at`].
    pub fn render_controls(&mut self, ui: &mut Ui, frame_response: &Response) {
        let hovered = ui.rect_contains_pointer(frame_response.rect);
        let player_state = self.player.player_state.get();
        let currently_seeking = matches!(
            player_state,
            PlayerState::SeekingInProgress | PlayerState::SeekingFinished
        );
        let is_stopped = matches!(player_state, PlayerState::Stopped);
        let is_paused = matches!(player_state, PlayerState::Paused);
        let animation_time = 0.2;
        let seekbar_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_anim"),
            hovered || currently_seeking || is_paused || is_stopped,
            animation_time,
        );

        if seekbar_anim_frac <= 0. {
            return;
        }

        let seekbar_width_offset = 20.;
        let fullseekbar_width = frame_response.rect.width() - seekbar_width_offset;

        let seekbar_width = fullseekbar_width * self.duration_frac();

        let seekbar_offset = 20.;
        let seekbar_pos =
            frame_response.rect.left_bottom() + vec2(seekbar_width_offset / 2., -seekbar_offset);
        let seekbar_height = 3.;
        let mut fullseekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(fullseekbar_width, seekbar_height));

        let mut seekbar_rect =
            Rect::from_min_size(seekbar_pos, vec2(seekbar_width, seekbar_height));
        let seekbar_interact_rect = fullseekbar_rect.expand(10.);

        let seekbar_response = ui.interact(
            seekbar_interact_rect,
            frame_response.id.with("seekbar"),
            Sense::click_and_drag(),
        );

        let seekbar_hovered = seekbar_response.hovered();
        let seekbar_hover_anim_frac = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seekbar_hover_anim"),
            seekbar_hovered || currently_seeking,
            animation_time,
        );

        if seekbar_hover_anim_frac > 0. {
            let new_top = fullseekbar_rect.top() - (3. * seekbar_hover_anim_frac);
            fullseekbar_rect.set_top(new_top);
            seekbar_rect.set_top(new_top);
        }

        let seek_indicator_anim = ui.ctx().animate_bool_with_time(
            frame_response.id.with("seek_indicator_anim"),
            currently_seeking,
            animation_time,
        );

        if currently_seeking {
            let seek_indicator_shadow = Shadow {
                offset: [10, 20],
                blur: 15,
                spread: 0,
                color: Color32::from_black_alpha(96).linear_multiply(seek_indicator_anim),
            };
            let spinner_size = 20. * seek_indicator_anim;
            ui.painter()
                .add(seek_indicator_shadow.as_shape(frame_response.rect, CornerRadius::ZERO));
            ui.put(
                Rect::from_center_size(frame_response.rect.center(), Vec2::splat(spinner_size)),
                Spinner::new().size(spinner_size),
            );
        }

        if seekbar_hovered || currently_seeking {
            if let Some(hover_pos) = seekbar_response.hover_pos() {
                if seekbar_response.clicked() || seekbar_response.dragged() {
                    let seek_frac = ((hover_pos - frame_response.rect.left_top()).x
                        - seekbar_width_offset / 2.)
                        .max(0.)
                        .min(fullseekbar_width)
                        / fullseekbar_width;
                    seekbar_rect.set_right(
                        hover_pos
                            .x
                            .min(fullseekbar_rect.right())
                            .max(fullseekbar_rect.left()),
                    );
                    if is_stopped {
                        self.start()
                    }
                    self.seek(seek_frac);
                }
            }
        }
        let text_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        let pause_icon = if is_paused {
            "▶"
        } else if is_stopped {
            "◼"
        } else if currently_seeking {
            "↔"
        } else {
            "⏸"
        };
        let audio_volume_frac = self.player.options.audio_volume.get() / self.player.options.max_audio_volume;
        let sound_icon = if audio_volume_frac > 0.7 {
            "🔊"
        } else if audio_volume_frac > 0.4 {
            "🔉"
        } else if audio_volume_frac > 0. {
            "🔈"
        } else {
            "🔇"
        };

        let icon_font_id = FontId {
            size: 16.,
            ..Default::default()
        };

        let subtitle_icon = "💬";
        let stream_icon = "🔁";
        let icon_margin = 5.;
        let text_y_offset = -7.;
        let sound_icon_offset = vec2(-5., text_y_offset);
        let sound_icon_pos = fullseekbar_rect.right_top() + sound_icon_offset;

        let stream_index_icon_offset = vec2(-30., text_y_offset + 1.);
        let stream_icon_pos = fullseekbar_rect.right_top() + stream_index_icon_offset;

        let contraster_alpha: u8 = 100;
        let pause_icon_offset = vec2(3., text_y_offset);
        let pause_icon_pos = fullseekbar_rect.left_top() + pause_icon_offset;

        let duration_text_offset = vec2(25., text_y_offset);
        let duration_text_pos = fullseekbar_rect.left_top() + duration_text_offset;
        let duration_text_font_id = FontId {
            size: 14.,
            ..Default::default()
        };

        let shadow = Shadow {
            offset: [10, 20],
            blur: 15,
            spread: 0,
            color: Color32::from_black_alpha(25).linear_multiply(seekbar_anim_frac),
        };

        let mut shadow_rect = frame_response.rect;
        shadow_rect.set_top(shadow_rect.bottom() - seekbar_offset - 10.);

        let fullseekbar_color = Color32::GRAY.linear_multiply(seekbar_anim_frac);
        let seekbar_color = Color32::WHITE.linear_multiply(seekbar_anim_frac);

        ui.painter()
            .add(shadow.as_shape(shadow_rect, CornerRadius::ZERO));

        ui.painter().rect_filled(
            fullseekbar_rect,
            CornerRadius::ZERO,
            fullseekbar_color.linear_multiply(0.5),
        );
        ui.painter()
            .rect_filled(seekbar_rect, CornerRadius::ZERO, seekbar_color);
        ui.painter().text(
            pause_icon_pos,
            Align2::LEFT_BOTTOM,
            pause_icon,
            icon_font_id.clone(),
            text_color,
        );

        ui.painter().text(
            duration_text_pos,
            Align2::LEFT_BOTTOM,
            self.duration_text(),
            duration_text_font_id,
            text_color,
        );

        if seekbar_hover_anim_frac > 0. {
            ui.painter().circle_filled(
                seekbar_rect.right_center(),
                7. * seekbar_hover_anim_frac,
                seekbar_color,
            );
        }

        if frame_response.clicked() {
            let mut reset_stream = false;
            let mut start_stream = false;

            match self.player.player_state.get() {
                PlayerState::Stopped => start_stream = true,
                PlayerState::EndOfFile => reset_stream = true,
                PlayerState::Paused => self.player.player_state.set(PlayerState::Playing),
                PlayerState::Playing => self.player.player_state.set(PlayerState::Paused),
                _ => (),
            }

            if reset_stream {
                self.player.reset();
                self.resume();
            } else if start_stream {
                self.start();
            }
        }

        let is_audio_cyclable = self.player.audio_stream_info.is_cyclable();
        let is_subtitle_cyclable = self.player.audio_stream_info.is_cyclable();

        if is_audio_cyclable || is_subtitle_cyclable {
            let stream_icon_rect = ui.painter().text(
                stream_icon_pos,
                Align2::RIGHT_BOTTOM,
                stream_icon,
                icon_font_id.clone(),
                text_color,
            );
            let stream_icon_hovered = ui.rect_contains_pointer(stream_icon_rect);
            let mut stream_info_hovered = false;
            let mut cursor = stream_icon_rect.right_top() + vec2(0., 5.);
            let cursor_offset = vec2(3., 15.);
            let stream_anim_id = frame_response.id.with("stream_anim");
            let mut stream_anim_frac: f32 = ui
                .ctx()
                .memory_mut(|m| *m.data.get_temp_mut_or_default(stream_anim_id));

            let mut draw_row = |stream_type: Type| {
                let text = match stream_type {
                    Type::Audio => format!("{} {}", sound_icon, self.player.audio_stream_info),
                    Type::Subtitle => format!("{} {}", subtitle_icon, self.player.subtitle_stream_info),
                    _ => unreachable!(),
                };

                let text_position = cursor - cursor_offset;
                let text_galley =
                    ui.painter()
                        .layout_no_wrap(text.clone(), icon_font_id.clone(), text_color);

                let background_rect =
                    Rect::from_min_max(text_position - text_galley.size(), text_position)
                        .expand(5.);

                let background_color =
                    Color32::from_black_alpha(contraster_alpha).linear_multiply(stream_anim_frac);

                ui.painter()
                    .rect_filled(background_rect, CornerRadius::same(5), background_color);

                if ui.rect_contains_pointer(background_rect.expand(5.)) {
                    stream_info_hovered = true;
                }

                if ui
                    .interact(
                        background_rect,
                        frame_response.id.with(&text),
                        Sense::click(),
                    )
                    .clicked()
                {
                    match stream_type {
                        Type::Audio => self.cycle_audio_stream(),
                        Type::Subtitle => self.cycle_subtitle_stream(),
                        _ => unreachable!(),
                    };
                };

                let text_rect = ui.painter().text(
                    text_position,
                    Align2::RIGHT_BOTTOM,
                    text,
                    icon_font_id.clone(),
                    text_color.linear_multiply(stream_anim_frac),
                );

                cursor.y = text_rect.top();
            };

            if stream_anim_frac > 0. {
                if is_audio_cyclable {
                    draw_row(Type::Audio);
                }
                if is_subtitle_cyclable {
                    draw_row(Type::Subtitle);
                }
            }

            stream_anim_frac = ui.ctx().animate_bool_with_time(
                stream_anim_id,
                stream_icon_hovered || (stream_info_hovered && stream_anim_frac > 0.),
                animation_time,
            );

            ui.ctx()
                .memory_mut(|m| m.data.insert_temp(stream_anim_id, stream_anim_frac));
        }

        if self.player.audio_streamer.is_some() {
            let sound_icon_rect = ui.painter().text(
                sound_icon_pos,
                Align2::RIGHT_BOTTOM,
                sound_icon,
                icon_font_id.clone(),
                text_color,
            );
            if ui
                .interact(
                    sound_icon_rect,
                    frame_response.id.with("sound_icon_sense"),
                    Sense::click(),
                )
                .clicked()
            {
                if self.player.options.audio_volume.get() != 0. {
                    self.player.options.audio_volume.set(0.)
                } else {
                    self.player.options
                        .audio_volume
                        .set(self.player.options.max_audio_volume / 2.)
                }
            }

            let sound_slider_outer_height = 75.;

            let mut sound_slider_rect = sound_icon_rect;
            sound_slider_rect.set_bottom(sound_icon_rect.top() - icon_margin);
            sound_slider_rect.set_top(sound_slider_rect.top() - sound_slider_outer_height);

            let sound_slider_interact_rect = sound_slider_rect.expand(icon_margin);
            let sound_hovered = ui.rect_contains_pointer(sound_icon_rect);
            let sound_slider_hovered = ui.rect_contains_pointer(sound_slider_interact_rect);
            let sound_anim_id = frame_response.id.with("sound_anim");
            let mut sound_anim_frac: f32 = ui
                .ctx()
                .memory_mut(|m| *m.data.get_temp_mut_or_default(sound_anim_id));
            sound_anim_frac = ui.ctx().animate_bool_with_time(
                sound_anim_id,
                sound_hovered || (sound_slider_hovered && sound_anim_frac > 0.),
                0.2,
            );
            ui.ctx()
                .memory_mut(|m| m.data.insert_temp(sound_anim_id, sound_anim_frac));
            let sound_slider_bg_color =
                Color32::from_black_alpha(contraster_alpha).linear_multiply(sound_anim_frac);
            let sound_bar_color =
                Color32::from_white_alpha(contraster_alpha).linear_multiply(sound_anim_frac);
            let mut sound_bar_rect = sound_slider_rect;
            sound_bar_rect
                .set_top(sound_bar_rect.bottom() - audio_volume_frac * sound_bar_rect.height());

            ui.painter()
                .rect_filled(sound_slider_rect, CornerRadius::same(5), sound_slider_bg_color);

            ui.painter()
                .rect_filled(sound_bar_rect, CornerRadius::same(5), sound_bar_color);
            let sound_slider_resp = ui.interact(
                sound_slider_rect,
                frame_response.id.with("sound_slider_sense"),
                Sense::click_and_drag(),
            );
            if sound_anim_frac > 0. && sound_slider_resp.clicked() || sound_slider_resp.dragged() {
                if let Some(hover_pos) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                    let sound_frac = 1.
                        - ((hover_pos - sound_slider_rect.left_top()).y
                        / sound_slider_rect.height())
                        .clamp(0., 1.);
                    self.player.options
                        .audio_volume
                        .set(sound_frac * self.player.options.max_audio_volume);
                }
            }
        }
    }

    #[cfg(feature = "from_bytes")]
    /// Create a new [`Player`] from input bytes.
    pub fn from_bytes(ctx: &egui::Context, input_bytes: &[u8]) -> Result<Self> {
        let mut file = tempfile::Builder::new().tempfile()?;
        file.write_all(input_bytes)?;
        let path = file.path().to_string_lossy().to_string();
        let mut slf = Self::new(ctx, &path)?;
        slf.temp_file = Some(file);
        Ok(slf)
    }

    /// Switches to the next subtitle stream.
    pub fn cycle_subtitle_stream(&mut self) {
        self.player.cycle_subtitle_stream();
    }

    /// Switches to the next audio stream.
    pub fn cycle_audio_stream(&mut self) {
        self.player.cycle_audio_stream();
    }

    /// Enables using [`Player::add_audio`] with the builder pattern.
    pub fn with_audio(mut self, audio_device: &mut AudioDevice) -> Result<Self> {
        self.player.add_audio(audio_device)?;

        Ok(self)
    }

    /// Enables using [`Player::add_subtitles`] with the builder pattern.
    pub fn with_subtitles(mut self) -> Result<Self> {
        self.player.add_subtitles()?;

        Ok(self)
    }

    fn duration_frac(&mut self) -> f32 {
        self.player.elapsed_ms() as f32 / self.player.duration_ms as f32
    }
}
