use eframe::NativeOptions;
use egui::{CentralPanel, DragValue, Grid, Sense, Slider, TextEdit, Window};
use egui_video::{AudioDevice, GuiPlayer};

fn main() {
    let _ = eframe::run_native(
        "app",
        NativeOptions::default(),
        Box::new(|_| Ok(Box::new(App::default()))),
    );
}
struct App {
    audio_device: AudioDevice,
    gui_player: Option<GuiPlayer>,

    media_path: String,
    stream_size_scale: f32,
    seek_frac: f32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            audio_device: AudioDevice::new(),
            media_path: String::new(),
            stream_size_scale: 1.,
            seek_frac: 0.,
            gui_player: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
                    if ui.button("load").clicked() {
                        match GuiPlayer::new(ctx, &self.media_path.replace("\"", "")).and_then(|p| {
                            p.with_audio(&mut self.audio_device)
                                .and_then(|p| p.with_subtitles())
                        }) {
                            Ok(player) => {
                                self.gui_player = Some(player);
                            }
                            Err(e) => println!("failed to make stream: {e}"),
                        }
                    }
                });
                ui.add_enabled_ui(!self.media_path.is_empty(), |ui| {
                    if ui.button("clear").clicked() {
                        self.gui_player = None;
                    }
                });

                let tedit_resp = ui.add_sized(
                    [ui.available_width(), ui.available_height()],
                    TextEdit::singleline(&mut self.media_path)
                        .hint_text("click to set path")
                        .interactive(false),
                );

                if ui
                    .interact(
                        tedit_resp.rect,
                        tedit_resp.id.with("click_sense"),
                        Sense::click(),
                    )
                    .clicked()
                {
                    if let Some(path_buf) = rfd::FileDialog::new()
                        .add_filter("videos", &["mp4", "gif", "webm", "mkv", "ogg"])
                        .pick_file()
                    {
                        self.media_path = path_buf.as_path().to_string_lossy().to_string();
                    }
                }
            });
            ui.separator();
            if let Some(gui_player) = self.gui_player.as_mut() {
                Window::new("info").show(ctx, |ui| {
                    Grid::new("info_grid").show(ui, |ui| {
                        ui.label("frame rate");
                        ui.label(gui_player.player.framerate.to_string());
                        ui.end_row();

                        ui.label("size");
                        ui.label(format!("{}x{}", gui_player.player.size.x, gui_player.player.size.y));
                        ui.end_row();

                        ui.label("elapsed / duration");
                        ui.label(gui_player.duration_text());
                        ui.end_row();

                        ui.label("state");
                        ui.label(format!("{:?}", gui_player.player.player_state.get()));
                        ui.end_row();

                        ui.label("has audio?");
                        ui.label(gui_player.player.audio_streamer.is_some().to_string());
                        ui.end_row();

                        ui.label("has subtitles?");
                        ui.label(gui_player.player.subtitle_streamer.is_some().to_string());
                        ui.end_row();
                    });
                });
                Window::new("controls").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("seek to:").clicked() {
                            gui_player.seek(self.seek_frac);
                        }
                        ui.add(
                            DragValue::new(&mut self.seek_frac)
                                .speed(0.05)
                                .range(0.0..=1.0),
                        );
                        ui.checkbox(&mut gui_player.player.options.looping, "loop");
                    });
                    ui.horizontal(|ui| {
                        ui.label("size scale");
                        ui.add(Slider::new(&mut self.stream_size_scale, 0.0..=2.));
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("play").clicked() {
                            gui_player.start()
                        }
                        if ui.button("unpause").clicked() {
                            gui_player.resume();
                        }
                        if ui.button("pause").clicked() {
                            gui_player.pause();
                        }
                        if ui.button("stop").clicked() {
                            gui_player.stop();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("volume");
                        let mut volume = gui_player.player.options.audio_volume.get();
                        if ui
                            .add(Slider::new(
                                &mut volume,
                                0.0..=gui_player.player.options.max_audio_volume,
                            ))
                            .changed()
                        {
                            gui_player.player.options.audio_volume.set(volume);
                        };
                    });
                });

                gui_player.ui(ui, gui_player.player.size * self.stream_size_scale);
            }
        });
    }
}
