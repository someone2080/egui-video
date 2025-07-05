# 0.10.0

- New `GuiPlayer` can be used to embed video into `egui`
- BREAKING: `Player` now acts as a headless video player

# 0.9.0
 - newType `StreamIndex`
 - removed `Player::stop_direct` (just use `Player::stop` now)
 - added `PlayerOptions::set_audio_volume` and `PlayerOptions::set_max_audio_volume` for convenience
 - fix [soundness issue](https://github.com/n00kii/egui-video/pull/19)
 - fix [possible overflow from elapsed time calculation](https://github.com/n00kii/egui-video/issues/20)
