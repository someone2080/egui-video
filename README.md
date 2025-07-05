# egui-video-headless, a video playing library for [`egui`](https://github.com/emilk/egui) or for `headless streaming`

[//]: # ([![crates.io]&#40;https://img.shields.io/crates/v/egui-video&#41;]&#40;https://crates.io/crates/egui-video&#41;)

[//]: # ([![docs]&#40;https://docs.rs/egui-video/badge.svg&#41;]&#40;https://docs.rs/egui-video/latest/egui_video/&#41;)

[//]: # ([![license]&#40;https://img.shields.io/badge/license-MIT-blue.svg&#41;]&#40;https://github.com/n00kii/egui-video/blob/main/README.md&#41;)

_This is a fork from original https://github.com/n00kii/egui-video_

https://github.com/n00kii/egui-video/assets/57325298/c618ff0a-9ad2-4cf0-b14a-dda65dc54b23

- Plays videos in egui from file path or from bytes
- Can be used to extract video frames from file or bytes

## Dependencies:

- requires ffmpeg 7. follow the build instructions [here](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building)

## Usage:
```rust
/* called once (top level initialization) */

{ // if using audio...
    let audio_device = egui_video::AudioDevice::new()?;
    
    // don't let audio_device drop out of memory! (or else you lose audio)

    add_audio_device_to_state_somewhere(audio_device);
}
```
```rust
/* called once (creating a player) */

let mut player = egui_video::Player::new(ctx, my_media_path)?;

{ // if using audio...
    player = player.with_audio(&mut my_state.audio_device)
}
```
```rust
/* called every frame (showing the player) */
player.ui(ui, player.size);
```
## contributions
are welcome :)

### current caveats
 - need to compile in `release` or `opt-level=3` otherwise limited playback performance
